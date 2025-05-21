import os
import json

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.cluster import KMeans

from private_knnvc.conv_decoder import load_model as load_conv_decoder
from wavlm.wavlm_model import WavLM, WavLMConfig
from hifigan import Generator, AttrDict


PHONE_DECODER_CKPT = "./checkpoints/phone_decoder.pt"
DURATION_DECODER_CKPT = "./checkpoints/duration_decoder.pt"
WAVLM_CKPT = "./checkpoints/WavLM-Large.pt"
HIFIGAN_CKPT = "./checkpoints/knnvc/hifigan.pt"
HIFIGAN_CFG = "./checkpoints/knnvc/hifigan.json"


class Converter:
    def __init__(self, device: str, dur_w: float, n_phone_clusters: int) -> None:
        """
        Initialize the converter.
        """
        self.device = device

        # load the WavLM model
        ckpt = torch.load(WAVLM_CKPT, map_location=device, weights_only=False)
        wavlm_cfg = WavLMConfig(ckpt["cfg"])
        self.wavlm = WavLM(wavlm_cfg)
        self.wavlm.load_state_dict(ckpt["model"])
        self.wavlm.to(device)
        self.wavlm.eval()

        # load the vocoder
        self.hifigan = Generator(AttrDict(json.load(open(HIFIGAN_CFG))))
        state_dict = torch.load(HIFIGAN_CKPT, map_location=device, weights_only=False)
        self.hifigan.load_state_dict(state_dict["generator"])
        self.hifigan.to(device)
        self.hifigan.eval()
        self.hifigan.remove_weight_norm()

        # load the phone and duration predictors
        self.phone_lexicon = (
            open("./private_knnvc/phone_lexicon.txt", "r").read().splitlines()
        )
        self.dur_w = dur_w
        self.n_phone_clusters = n_phone_clusters
        self.phone_predictor = load_conv_decoder(PHONE_DECODER_CKPT, device)
        self.duration_predictor = load_conv_decoder(DURATION_DECODER_CKPT, device)

        # load the WavLM features of the targets
        self.target_feats = list()
        target_feats_dir = "./target_feats"
        for spk_idx in range(len(os.listdir(target_feats_dir))):
            self.target_feats.append(
                torch.load(os.path.join(target_feats_dir, f"{spk_idx}.pt"))
            )

        # cluster the target features if needed
        if self.n_phone_clusters > 0:
            for spk_idx in tqdm(
                range(len(self.target_feats)), desc="Clustering the target feats"
            ):
                for phone, feats in enumerate(self.target_feats[spk_idx]):
                    if feats.nelement() == 0:
                        self.target_feats[spk_idx][phone] = torch.zeros(
                            (self.n_phone_clusters, 1024)
                        )
                        continue

                    if feats.shape[0] < self.n_phone_clusters:
                        feats = torch.cat(
                            [feats] * (self.n_phone_clusters // feats.shape[0] + 1),
                            dim=0,
                        )
                    kmeans = KMeans(
                        n_clusters=self.n_phone_clusters, n_init="auto"
                    ).fit(feats)
                    self.target_feats[spk_idx][phone] = torch.tensor(
                        kmeans.cluster_centers_
                    )

                self.target_feats[spk_idx] = torch.stack(self.target_feats[spk_idx]).to(
                    torch.float
                )

    @torch.inference_mode()
    def run(self, audio: Tensor, target_idx: int) -> str:
        """
        Selects the target speakers for the given batch if needed and converts the batch
        to those targets.

        Args:
            batch: a list with a tensor comprising spectrograms in first position.
        """
        # get the features, source and target speakers
        with torch.no_grad():
            wavlm_layers = [6, 24]
            last_feats, all_feats = self.wavlm.extract_features(
                audio,
                ret_layer_results=True,
                output_layer=wavlm_layers[-1],
            )[0]
            src_feats = list()
            for layer in wavlm_layers:
                src_feats.append(all_feats[layer][0].transpose(0, 1))
            src_feats = torch.stack(src_feats)

        # anonymize source features and synthesize them
        tgt_feats = self.target_feats[target_idx]
        converted_feats = self.convert_vecs(
            src_feats, [src_feats.shape[2]], [tgt_feats]
        )[0]
        return self.hifigan(converted_feats.unsqueeze(0)).squeeze()

    def convert_vecs(self, src_vecs: Tensor, src_lens: Tensor, tgt_vecs: list) -> list:
        """
        Given the WavLM vecs of the source and target audios, convert them with the
        KnnVC matching algorithm.

        Args:
            src_vecs: tensor of shape (2, batch_size, n_vecs_s, vec_dim)
            src_lens: tensor of shape (batch_size,) with the number of vecs in each src
            tgt_vecs: list with tensors of shape (n_phones, n_vecs_t, vec_dim)
                if self.n_phone_clusters > 0 else (n_vecs_t, vec_dim)

        Returns:
            list with the converted wavLM vectors for each batch element
        """
        vc_feats, phone_feats = src_vecs
        batch_phones = self.phone_predictor(phone_feats).argmax(dim=2)
        phones = list()  # the distinct predicted phones for the src feats
        phone_durations = list()  # the actual duration of each phone in the src feats
        for src_idx in range(batch_phones.shape[0]):
            unique, counts = torch.unique_consecutive(
                batch_phones[src_idx, : src_lens[src_idx]], return_counts=True
            )
            phones.append(unique)
            phone_durations.append(counts)

        phone_lens = torch.tensor([len(p) for p in phones], device=self.device)
        phones = pad_sequence(phones, batch_first=True).to(self.device)
        phone_durations = pad_sequence(phone_durations, batch_first=True).to(
            self.device
        )

        # interpolate the actual durations with the predicted ones
        pred_durations = self.duration_predictor(phones)
        interpolated_durations = (
            self.dur_w * pred_durations + (1 - self.dur_w) * phone_durations
        )
        n_frames = torch.round(interpolated_durations).to(torch.int64)
        n_frames[n_frames <= 0] = 1

        # duplicate the features according to the durations
        converted_feats = list()
        for src_idx in range(vc_feats.shape[0]):
            src_feats_dur = list()
            src_phones_dur = list()
            feat_idx_start = 0
            for distinct_idx in range(phone_lens[src_idx]):
                phone_dur = phone_durations[src_idx][distinct_idx].item()
                feat_idx_end = feat_idx_start + phone_dur - 1

                # get `n_frames[distinct_idx]` between `feat_idx_start` and `feat_idx_end`
                feat_indices = torch.linspace(
                    feat_idx_start,
                    feat_idx_end,
                    n_frames[src_idx][distinct_idx].item(),
                    dtype=torch.int64,
                )
                src_feats_dur.append(vc_feats[src_idx, feat_indices])
                src_phones_dur.append(
                    torch.ones(
                        n_frames[src_idx][distinct_idx],
                        dtype=torch.int64,
                        device=self.device,
                    )
                    * batch_phones[src_idx, feat_idx_start]
                )
                feat_idx_start = feat_idx_end + 1

            src_feats_dur = torch.cat(src_feats_dur, dim=0)
            src_phones_dur = torch.cat(src_phones_dur, dim=0)

            # compute the similarities between the source and target feats
            # each source feat is only compared to the target feats of the same phone
            if self.n_phone_clusters > 0:
                tgt_feats = tgt_vecs[src_idx][src_phones_dur]
                dot_p = torch.bmm(
                    src_feats_dur.unsqueeze(1), tgt_feats.transpose(1, 2)
                ).squeeze(1)
                src_norm = torch.norm(src_feats_dur, dim=-1)
                tgt_norm = torch.norm(tgt_feats, dim=-1)
                quotient = src_norm.unsqueeze(1) * tgt_norm
                cos_sim = torch.div(dot_p, quotient)

                # get the indices of the most similar target feats
                max_indices = torch.argmax(cos_sim, dim=1)
                conv_feats = tgt_feats[torch.arange(tgt_feats.shape[0]), max_indices]
            else:
                conv_feats = torch.empty_like(src_feats_dur)
                for feat_idx, phone in enumerate(src_phones_dur):
                    src_feat = src_feats_dur[feat_idx]
                    tgt_feats = tgt_vecs[src_idx][phone]
                    if tgt_feats.shape[0] == 0:
                        conv_feats[feat_idx] = torch.zeros_like(src_feat)
                    else:
                        cos_sim = cosine_similarity(src_feat.unsqueeze(0), tgt_feats)
                        conv_feats[feat_idx] = tgt_feats[cos_sim.argmax()]

            converted_feats.append(conv_feats)

        return converted_feats


def cosine_similarity(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    """
    Compute the cosine similarity among all vectors in `tensor_a` and `tensor_b`.

    Args:
        tensor_a: tensor of shape (n_vecs_a, vec_dim)
        tensor_b: tensor of shape (n_vecs_b, vec_dim)

    Returns:
        cosine similarity tensor: tensor of shape (n_vecs_a, n_vecs_b)
    """
    dot_product = torch.matmul(tensor_a, tensor_b.transpose(-1, -2))
    source_norm = torch.norm(tensor_a, dim=-1)
    target_norm = torch.norm(tensor_b, dim=-1)
    cos_sim = dot_product / torch.outer(source_norm, target_norm)
    return cos_sim
