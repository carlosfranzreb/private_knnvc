"""
Train the duration predictor with the extracted phone durations from LJSpeech.

The datafile is a file with one phone-duration pair per line and a blank line between
speech samples.
"""

import os
import argparse
from time import time
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from conv_decoder import ConvDecoder


N_PHONES = 41


class Trainer:
    def __init__(self, datafile: str, batch_size: int, max_epochs: int):

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.mps.is_available():
            self.device = torch.device("mps")

        # create the experiment folder
        exp_folder = os.path.join("logs", "train_dur_predictor", str(int(time())))
        os.makedirs(exp_folder)
        self.logger = open(os.path.join(exp_folder, "log.txt"), "w")
        self.tb_logger = SummaryWriter(log_dir=exp_folder)
        self.batch_size = batch_size

        # dump args and commit hash to exp folder
        with open(os.path.join(exp_folder, "args.txt"), "w") as f:
            f.write(f"Datafile: {datafile}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Max epochs: {max_epochs}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Commit hash: {os.popen('git rev-parse HEAD').read().strip()}\n")

        # load the data as a huge tensor
        data = [[]]
        for line in open(datafile):
            try:
                phone_idx, count = line.strip().split()
                data[-1].append([int(phone_idx), int(count)])
            except ValueError:
                data[-1] = torch.tensor(data[-1], dtype=torch.float32)
                data.append([])
        data = pad_sequence(data[:-1], batch_first=True).to(self.device)
        # ! TODO: use pad_idx instead of 0? It's part of the phone predictor as well

        # split the data into train, val and test sets and create the dataloader
        n_tr = int(data.shape[0] * 0.8)
        n_val = int(data.shape[0] * 0.1)
        self.data_split = {
            "train": data[:n_tr],
            "val": data[n_tr : n_tr + n_val],
            "test": data[n_tr + n_val :],
        }

        # load the model
        self.model = ConvDecoder(emb_dim=N_PHONES).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

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

        # test the model and store the predictions
        self.model.eval()
        self.run_epoch(
            0, "test", dump_to=os.path.join(exp_folder, "test_predictions.txt")
        )
        self.logger.close()

    def run_epoch(self, epoch_nr: int, split: str, dump_to: str = None) -> float:
        self.model.train() if split == "train" else self.model.eval()
        dl = DataLoader(
            self.data_split[split],
            batch_size=self.batch_size,
            shuffle=True if split == "train" else False,
        )
        total_loss = 0
        for batch in tqdm(dl, desc=split):
            batch = batch.to(self.device)
            phones, durations = batch[:, :, 0], batch[:, :, 1]
            pred_durations = self.model(phones.to(torch.int32))
            mask = phones != 0
            loss = mse_loss(pred_durations[mask], durations[mask])

            if split == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()

            if dump_to:
                with open(dump_to, "a") as f:
                    for sample_idx in range(phones.shape[0]):
                        for phone_idx, dur_true, dur_pred in zip(
                            phones[sample_idx],
                            durations[sample_idx],
                            pred_durations[sample_idx],
                        ):
                            if phone_idx != 0:
                                f.write(
                                    f"{phone_idx.item()} {dur_true.item()} {dur_pred.item()} \n"
                                )
                        f.write("\n")
                    f.write("\n")

        avg_loss = total_loss / len(dl)
        self.logger.write(f"\t{split} loss: {avg_loss}\n")
        self.tb_logger.add_scalar(f"{split}/Avg_epoch_loss", avg_loss, epoch_nr)

        return total_loss


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "datafile", type=str, help="Path to the extracted phone durations file."
    )
    ap.add_argument("--bs", type=int, default=512, help="Batch size.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs."
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    Trainer(args.datafile, args.bs, args.max_epochs)
