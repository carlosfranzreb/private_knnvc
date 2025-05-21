"""
Train the phone decoder with LibriSpeech data. Ensure that the seed is the same
as the one used for training. Then, the test set will be the same and there will
be no leakage of information.
"""

import os
import argparse
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import editdistance

from conv_decoder import load_model as load_conv_decoder
from wavlm_loader import load_wavlm
from train_utils import get_phones, get_split_datasets


def test(exp_folder: str):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    phone_map, pad_idx, phones_true = get_phones()

    with open(os.path.join(exp_folder, "args.txt")) as f:
        lines = f.readlines()
        data_dir = lines[0].split(":")[1].strip()

    dl = DataLoader(
        get_split_datasets(data_dir, phones_true)["test"],
        batch_size=8,
        num_workers=10 if device == "cuda" else 0,
        collate_fn=lambda x: [
            pad_sequence([y[0] for y in x], batch_first=True),
            [y[1] for y in x],
        ],
    )

    # load the models
    wavlm = load_wavlm(device)
    model = load_conv_decoder(os.path.join(exp_folder, "best.pt"), device)

    writer = open(os.path.join(exp_folder, "test_results.txt"), "w")
    n_correct, n_edits, n_phones_all, n_phones_unique = 0, 0, 0, 0
    tp, fp, fn = [0] * len(phone_map), [0] * len(phone_map), [0] * len(phone_map)

    for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
        audio, files = batch

        # remove files for which we don't have the phones
        indices, fnames = list(), list()
        for file_idx, file in enumerate(files):
            fname = os.path.splitext(os.path.basename(file))[0]
            if fname in phones_true:
                indices.append(file_idx)
                fnames.append(fname)
            else:
                print(f"Missing ground truth for {fname}")

        with torch.no_grad():
            audio = audio[indices].to(device)
            feats = wavlm.extract_features(audio)[0]
            x = model(feats)

        for sample_idx in range(x.shape[0]):

            # truncate labels if there are not as many predictions
            fname = fnames[sample_idx]
            y_all = phones_true[fname].to(device)
            x_all = x[sample_idx].argmax(dim=-1)
            x_all = x_all[x_all != pad_idx]
            if x_all.shape[0] < y_all.shape[0]:
                y_all = y_all[: x_all.shape[0]]
            elif x_all.shape[0] > y_all.shape[0]:
                x_all = x_all[: y_all.shape[0]]

            # compute the confusion matrix
            for x_p, y_p in zip(x_all, y_all):
                if x_p == y_p:
                    tp[x_p] += 1
                else:
                    fp[x_p] += 1
                    fn[y_p] += 1

            # compute the accuracy and PER
            n_correct_sample = (x_all == y_all).sum().item()
            acc = n_correct_sample / x_all.shape[0]
            n_correct += n_correct_sample
            n_phones_all += x_all.shape[0]

            x_unique = torch.unique_consecutive(x_all)
            y_unique = torch.unique_consecutive(y_all)
            n_edits_sample = editdistance.eval(y_unique, x_unique)
            per = n_edits_sample / len(y_unique)
            n_edits += n_edits_sample
            n_phones_unique += len(y_unique)

            # print the results
            strings = dict()
            p_len = 3
            for key, data in zip(["Y", "X"], [y_all, x_all]):
                strings[key] = ""
                for p_idx in range(data.shape[0]):
                    p = phone_map[data[p_idx]]
                    if len(p) < p_len:
                        p += " " * (p_len - len(p))
                    strings[key] += p + " "

            writer.write(f"{fname} {acc:.4f} {per:.4f}\n")
            writer.write(f"Y: {strings['Y']}\n")
            writer.write(f"X: {strings['X']}\n\n")
            writer.flush()

    writer.close()

    # compute and dump the overall metrics
    acc = n_correct / n_phones_all
    per = n_edits / n_phones_unique

    class_f1 = [0] * len(phone_map)
    for p_idx in range(len(phone_map)):
        precision = (
            tp[p_idx] / (tp[p_idx] + fp[p_idx]) if tp[p_idx] + fp[p_idx] > 0 else 0
        )
        recall = tp[p_idx] / (tp[p_idx] + fn[p_idx]) if tp[p_idx] + fn[p_idx] > 0 else 0
        class_f1[p_idx] = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

    macro_f1 = sum(class_f1) / len(phone_map)

    with open(os.path.join(exp_folder, "test_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"PER: {per:.4f}\n\n")
        f.write(f"n_correct: {n_correct}\n")
        f.write(f"n_edits: {n_edits}\n")
        f.write(f"n_phones_all: {n_phones_all}\n")
        f.write(f"n_phones_unique: {n_phones_unique}\n\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write("Class F1 scores:\n")
        for p_idx, phone in phone_map.items():
            f.write(f"{phone}: {class_f1[p_idx]:.4f}\n")


if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "exp_folder",
        type=str,
        help="Path to the experiment folder to be evaluated.",
    )
    args = ap.parse_args()
    test(args.exp_folder)
