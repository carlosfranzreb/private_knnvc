"""
Create target speakers out of LS' train-clean-100, grouping the features
by their true phone labels.
"""

import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.cluster import KMeans

from train_utils import get_phones
from wavlm_loader import load_wavlm
from dataset import SpeechDataset


def dump_spk_feats(spk: str, feats: list[list[Tensor]], dump_dir: str):
    """
    Dump the features of this speaker. Also, cluster thema and dump the
    cluster centers. Speaker IDs are written to a file.
    """
    for phone_idx in range(len(phone_map)):
        if len(feats[phone_idx]) > 0:
            feats[phone_idx] = torch.stack(feats[phone_idx])
        else:
            feats[phone_idx] = torch.tensor([])

    os.makedirs(os.path.join(dump_dir, "all_feats"), exist_ok=True)

    spk_file = os.path.join(dump_dir, "speakers.txt")
    spk_idx = len(open(spk_file).readlines()) if os.path.exists(spk_file) else 0
    torch.save(feats, os.path.join(dump_dir, "all_feats", f"{spk_idx}.pt"))
    with open(spk_file, "a") as f:
        f.write(f"{spk}\n")

    for n_clusters in [32, 16, 8, 2]:
        clustered_feats = list()
        for phone_idx, phone_feats in enumerate(feats):
            if phone_feats.shape[0] == 0:
                phone_feats = torch.zeros((n_clusters, 1024))
            elif phone_feats.shape[0] < n_clusters:
                phone_feats = torch.cat(
                    [phone_feats] * (n_clusters // phone_feats.shape[0] + 1),
                    dim=0,
                )
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
                phone_feats
            )
            clustered_feats.append(
                torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            )
        clustered_feats = torch.stack(clustered_feats)
        dump_dir_cluster = os.path.join(dump_dir, f"{n_clusters}_clusters")
        os.makedirs(dump_dir_cluster, exist_ok=True)
        torch.save(clustered_feats, os.path.join(dump_dir_cluster, f"{spk_idx}.pt"))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "/ds/audio/LibriSpeech/train-clean-100"
    dump_dir = "./logs/ls-targets-truephones"

    phone_map, pad_idx, phones_true = get_phones()
    wavlm = load_wavlm(device)

    dataset = SpeechDataset(data_dir, 16000, ".flac")
    dl = DataLoader(
        dataset,
        batch_size=8,
        num_workers=10 if device == "cuda" else 0,
        collate_fn=lambda x: [
            pad_sequence([y[0] for y in x], batch_first=True),
            torch.tensor([y[0].shape[0] for y in x]),
            [y[1] for y in x],
        ],
    )

    current_spk = ""
    spk_feats = [list() for _ in range(len(phone_map))]
    for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
        audios, lens, fnames = batch
        audios = audios.to(device)
        with torch.no_grad():
            feats = wavlm.extract_features(audios, output_layer=6)[0].to("cpu")

        for fname, utt_feats in zip(fnames, feats):
            fname = os.path.splitext(os.path.basename(fname))[0]
            if fname not in phones_true:
                print(f"{fname} not in phones_true")
                continue

            spk = fname.split("-")[0]
            if spk != current_spk:
                if len(spk_feats[0]) > 0:
                    dump_spk_feats(spk, spk_feats, dump_dir)
                current_spk = spk
                spk_feats = [list() for _ in range(len(phone_map))]

            for phone, feat in zip(phones_true[fname], utt_feats):
                spk_feats[phone.item()].append(feat)

    if len(spk_feats[0]) > 0:
        dump_spk_feats(spk, spk_feats, dump_dir)
