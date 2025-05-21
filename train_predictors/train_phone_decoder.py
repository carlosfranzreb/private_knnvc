"""
Train the phone decoder with LibriSpeech data. The CTC loss is used to ensure that the
sequence is correct. The ce loss is used to ensure that the duration of the phones is
correct.

As data we use LS-train-clean-100 and the aligned phones used in SUPERB. They are
stored in the file `files/aligned-phones_ls-train-clean-100.txt`.
"""

import os
import argparse
from time import time
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.functional import ctc_loss, cross_entropy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from conv_decoder import ConvDecoder
from wavlm_loader import load_wavlm
from train_utils import get_phones, get_split_datasets


class Trainer:
    def __init__(self, data_dir: str, batch_size: int, max_epochs: int):

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        # create the experiment folder
        exp_folder = os.path.join("logs", "train_phone_decoder", str(int(time())))
        os.makedirs(exp_folder)
        self.logger = open(os.path.join(exp_folder, "log.txt"), "w")
        self.tb_logger = SummaryWriter(log_dir=exp_folder)
        self.batch_size = batch_size
        self.ce_weight = 1.0
        self.ctc_weight = 1.0

        # dump args and commit hash to exp folder
        with open(os.path.join(exp_folder, "args.txt"), "w") as f:
            f.write(f"Data directory: {data_dir}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Max epochs: {max_epochs}\n")
            f.write(f"CE weight: {self.ce_weight}\n")
            f.write(f"CTC weight: {self.ctc_weight}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Commit hash: {os.popen('git rev-parse HEAD').read().strip()}\n")

        phone_map, self.pad_idx, self.phones_true = get_phones()
        self.datasets = get_split_datasets(data_dir, self.phones_true)

        # load the models
        self.wavlm = load_wavlm(self.device)
        self.model = ConvDecoder(
            encoder_embed_dim=1024, output_dim=len(phone_map) + 1
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # train the model
        best_val_loss = float("inf")
        for epoch_nr in range(max_epochs):
            self.logger.write(f"Epoch {epoch_nr}\n")
            for split in ["train", "val"]:
                total_loss = self.run_epoch(epoch_nr, split)

            # save the model
            self.logger.flush()
            torch.save(self.model.state_dict(), os.path.join(exp_folder, "last.pt"))
            if total_loss < best_val_loss:
                best_val_loss = total_loss
                torch.save(self.model.state_dict(), os.path.join(exp_folder, "best.pt"))

        # test the model
        self.model.eval()
        self.run_epoch(0, "test")
        self.logger.close()

    def run_epoch(self, epoch_nr: int, split: str) -> float:
        self.model.train() if split == "train" else self.model.eval()
        dl = DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            num_workers=10 if self.device == "cuda" else 0,
            shuffle=True if split == "train" else False,
            collate_fn=lambda x: [
                pad_sequence([y[0] for y in x], batch_first=True),
                [y[1] for y in x],
            ],
        )
        epoch_total_loss = 0
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc=split):
            audio, files = batch

            # remove files for which we don't have the phones
            indices = list()
            fnames = list()
            for idx, file in enumerate(files):
                fname = os.path.splitext(os.path.basename(file))[0]
                if fname in self.phones_true:
                    indices.append(idx)
                    fnames.append(fname)
                else:
                    self.logger.write(f"\tMissing ground truth for {fname}\n")
            audio = audio[indices].to(self.device)

            with torch.no_grad():
                feats = self.wavlm.extract_features(audio)[0]
            phones_x = self.model(feats)

            # truncate labels if there are not as many predictions
            phones_all = [self.phones_true[fname] for fname in fnames]
            for sample_idx in range(phones_x.shape[0]):
                pred_len = phones_x[sample_idx].shape[0]
                if pred_len < phones_all[sample_idx].shape[0]:
                    phones_all[sample_idx] = phones_all[sample_idx][:pred_len]

            phones_all_lens = torch.tensor([p.shape[0] for p in phones_all])
            phones_ce = torch.cat(phones_all).to(self.device)

            # truncate predictions if there are not as many labels
            phones_x_flat = torch.cat(
                [phones_x[idx, :len] for idx, len in enumerate(phones_all_lens)]
            )

            ce_loss_val = cross_entropy(phones_x_flat, phones_ce)

            phones_ctc = [torch.unique_consecutive(p) for p in phones_all]
            phones_ctc_lens = torch.tensor([p.shape[0] for p in phones_ctc])
            phones_ctc = pad_sequence(
                phones_ctc, batch_first=True, padding_value=self.pad_idx
            ).to(self.device)
            ctc_loss_val = ctc_loss(
                phones_x.transpose(0, 1), phones_ctc, phones_all_lens, phones_ctc_lens
            )

            total_loss_val = (
                self.ce_weight * ce_loss_val + self.ctc_weight * ctc_loss_val
            )
            epoch_total_loss += total_loss_val.item()

            if split == "train":
                self.optimizer.zero_grad()
                total_loss_val.backward()
                self.optimizer.step()
                step_idx = epoch_nr * len(dl) + batch_idx
                self.tb_logger.add_scalar(
                    "train/loss/total", total_loss_val.item(), step_idx
                )
                self.tb_logger.add_scalar("train/loss/ce", ce_loss_val.item(), step_idx)
                self.tb_logger.add_scalar(
                    "train/loss/ctc", ctc_loss_val.item(), step_idx
                )

        avg_loss = epoch_total_loss / len(dl)
        self.logger.write(f"\t{split} loss: {avg_loss}\n")
        self.tb_logger.add_scalar(f"{split}/Avg_epoch_loss", avg_loss, epoch_nr)

        return avg_loss


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        help="Path to the extracted phone durations file.",
        default="/Users/cafr02/datasets/LibriSpeech/train-clean-100",
    )
    ap.add_argument("--bs", type=int, default=2, help="Batch size.")
    ap.add_argument(
        "--max_epochs", type=int, default=15, help="Maximum number of epochs."
    )
    args = ap.parse_args()

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    Trainer(args.data_dir, args.bs, args.max_epochs)
