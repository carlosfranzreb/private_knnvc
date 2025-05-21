"""
Iterate through the speech samples and:

1. Compute their WavLM features.
2. Predict the phone for each WavLM feature with the SUPERB phone predictor.
3. Merge SIL and SPN, and trim the silence.
4. Store the phone and its duration (in WavLM frames).

All predictions are stored in one file. Each phone-duration pair is stored in a
separate line and there is a blank line between speech samples.
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from dataset import SpeechDataset
from conv_decoder import load_model as load_phone_predictor
from wavlm_loader import load_wavlm


@torch.inference_mode()
def main(data_dir: str, dump_path: str, max_dur: int = -1, f_format: str = "wav"):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    # Load the models
    phone_predictor = load_phone_predictor(
        "checkpoints/phone_decoder.pt",
        device,
    )
    wavlm = load_wavlm(device)

    # Load the dataset
    dl = DataLoader(
        SpeechDataset(data_dir, 16000, format=f_format),
        batch_size=8,
        num_workers=10 if device == "cuda" else 0,
        collate_fn=lambda x: pad_sequence([y[0] for y in x], batch_first=True),
    )

    # Predict the phones and store their durations
    writer = open(dump_path, "w")
    for batch in tqdm(dl):
        with torch.no_grad():
            batch = batch.to(device)
            feats = wavlm.extract_features(batch)[0]
            phones = phone_predictor(feats).argmax(dim=-1)
        phones[phones == 0] = 1

        # count the number of repetitions of each phone
        for sample_idx in range(batch.shape[0]):
            unique_phones, counts = torch.unique_consecutive(
                phones[sample_idx], return_counts=True
            )
            if unique_phones[0] == 0:
                unique_phones = unique_phones[1:]
                counts = counts[1:]
            if unique_phones[-1] == 0:
                unique_phones = unique_phones[:-1]
                counts = counts[:-1]
            if max_dur > 0:
                counts[counts > max_dur] = max_dur
            for phone, count in zip(unique_phones, counts):
                writer.write(f"{phone.item()} {count.item()}\n")
            writer.write("\n")
        writer.flush()
    writer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "data_dir", type=str, help="The directory containing the audio files."
    )
    ap.add_argument(
        "dump_path", type=str, help="The path to store the phone predictions."
    )
    ap.add_argument(
        "--max_dur", type=int, default=-1, help="Maximum duration of each phone."
    )
    ap.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the dump file"
    )

    args = ap.parse_args()
    if not args.force and os.path.exists(args.dump_path):
        raise ValueError(f"{args.dump_path} already exists. Use -f to overwrite it.")
    main(args.data_dir, args.dump_path, args.max_dur)
