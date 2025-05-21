"""
Selects targets based on speaker embeddings.
"""

import os
import json
import logging
import importlib

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from tqdm import tqdm

from spkanon_eval.setup_module import setup as setup_module
from spkanon_eval.datamodules.dataloader import eval_dataloader

from .conv_decoder import load_model as load_conv_decoder

LOGGER = logging.getLogger("progress")


class Converter:
    def __init__(self, config: OmegaConf, device: str) -> None:
        """
        Initialize the converter.

        It requires a datafile for the targets in the path data/targets.txt under
        the `exp_folder`, which is present in the config object.

        You can also pass a directory with the WavLM features of the targets
        (config.target_feats), as dumped by a previous run. If this directory is not
        given, the features are extracted here. They are also dumped, so that they can
        be reused in future runs.

        Args:
            config: we assume that all samples of each speaker are consecutive.
            device: the device where the model should run.
        """
        self.config = config
        self.device = device
        self.target_selection = None  # initialized later (see init_target_selection)
        self.target_feats = list()
        self.target_is_male = list()
        target_df = os.path.join(config.exp_folder, "data", "targets.txt")

        # load the phone and duration predictors
        self.phone_lexicon = (
            open(config.converter_params.phone_lexicon, "r").read().splitlines()
        )
        self.dur_w = config.converter_params.get("duration_prediction_weight", 0.0)
        self.n_phone_clusters = config.converter_params.get("n_phone_clusters", 0)
        self.phone_predictor = load_conv_decoder(
            config.converter_params.phone_predictor_ckpt, device
        )
        self.duration_predictor = load_conv_decoder(
            config.converter_params.duration_predictor_ckpt, device
        )

        # if possible, load the WavLM features of the targets
        target_feats_dir = config.converter_params.get("target_feats", None)
        if target_feats_dir is not None:
            LOGGER.info(f"Loading target features from {target_feats_dir}")
            for spk_idx in range(len(os.listdir(target_feats_dir))):
                self.target_feats.append(
                    torch.load(os.path.join(target_feats_dir, f"{spk_idx}.pt"))
                )

            # ensure that the target features are in the correct format
            if self.n_phone_clusters > 0:
                assert (
                    self.target_feats[0].ndim == 3  # n_phones, n_clusters, feat_dim
                    and self.target_feats[0].shape[1] == self.n_phone_clusters
                )

            # gather the genders of the target speakers
            LOGGER.info("Loading the genders of the target speakers")
            self.target_is_male = torch.zeros(len(self.target_feats), dtype=torch.bool)
            with open(target_df, "r") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    spk = obj["speaker_id"]
                    if spk < len(self.target_feats):
                        self.target_is_male[spk] = obj["gender"] == "M"

            return

        # otherwise compute the WavLM features and concatenate them for each speaker
        LOGGER.info("Extracting target features")
        wavlm = setup_module(config.wavlm, device)
        dl = eval_dataloader(config.wavlm_dl, target_df, wavlm, max_ratio=0.5)
        for _, batch, data in tqdm(dl):
            feats, feat_lengths = wavlm.run(batch).values()
            vc_feats, phone_feats = feats
            phones = self.phone_predictor(phone_feats).argmax(dim=2).cpu()
            vc_feats = vc_feats.to("cpu")

            for sample_idx in range(batch[0].shape[0]):

                # expand the target_feats list to the current speaker
                spk = data[sample_idx]["speaker_id"]
                while len(self.target_feats) <= spk:
                    self.target_feats.append(
                        [list() for _ in range(len(self.phone_lexicon))]
                    )
                    self.target_is_male.append(None)
                if self.target_is_male[spk] is None:
                    self.target_is_male[spk] = data[sample_idx]["gender"] == "M"

                # append the features to the corresponding phone
                for feat_idx in range(feat_lengths[sample_idx]):
                    if feat_idx >= phones.shape[1]:
                        break
                    phone = phones[sample_idx, feat_idx]
                    self.target_feats[spk][phone].append(vc_feats[sample_idx, feat_idx])

        # vectorize the lists
        self.target_is_male = torch.tensor(self.target_is_male)
        for spk_idx in range(len(self.target_feats)):
            for phone_idx in range(len(self.target_feats[spk_idx])):
                if len(self.target_feats[spk_idx][phone_idx]) > 0:
                    self.target_feats[spk_idx][phone_idx] = torch.stack(
                        self.target_feats[spk_idx][phone_idx]
                    )
                else:
                    self.target_feats[spk_idx][phone_idx] = torch.tensor([])

        # cluster the features if needed
        if self.n_phone_clusters > 0:
            for spk_idx in tqdm(
                range(len(self.target_feats)), desc="Clustering the target feats"
            ):
                for phone, feats in enumerate(self.target_feats[spk_idx]):
                    if feats.nelement() == 0:
                        LOGGER.warning(f"Phone {phone} not found for target {spk_idx}.")
                        self.target_feats[spk_idx][phone] = torch.zeros(
                            (self.n_phone_clusters, vc_feats.shape[-1])
                        )
                        continue

                    if feats.shape[0] < self.n_phone_clusters:
                        LOGGER.warning(
                            f"Phone {phone} has {feats.shape[0]} occurrences for target {spk_idx}."
                        )
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

        # dump the features
        dump_folder = os.path.join(config.exp_folder, "target_feats")
        os.makedirs(dump_folder)
        for idx in range(len(self.target_feats)):
            torch.save(self.target_feats[idx], os.path.join(dump_folder, f"{idx}.pt"))

    def init_target_selection(self, cfg: OmegaConf, *args):
        """
        Initialize the target selection algorithm. This method is called by the
        anonymizer, passing it config and the arguments that the defined algorithm
        requires. These are passed directly to the algorithm, along with the target
        features computed in the constructor.
        """
        self.target_is_male = self.target_is_male.to(self.device)
        module_str, cls_str = cfg.cls.rsplit(".", 1)
        module = importlib.import_module(module_str)
        cls = getattr(module, cls_str)
        self.target_selection = cls(
            self.target_feats, cfg, target_is_male=self.target_is_male, *args
        )

    def run(self, batch: list) -> dict:
        """
        Selects the target speakers for the given batch if needed and converts the batch
        to those targets.

        Args:
            batch: a list with a tensor comprising spectrograms in first position.
        """
        # get the features, source and target speakers
        feats = batch[self.config.input.feats]
        n_feats = batch[self.config.input.n_feats].to("cpu")
        source = batch[self.config.input.source].to("cpu")
        source_is_male = batch[self.config.input.source_is_male]
        target = self.target_selection.select(feats[0], source, source_is_male)

        # run the conversion
        tgt_feats = [self.target_feats[tgt] for tgt in target]
        if self.n_phone_clusters > 0:
            tgt_feats = [t.to(self.device) for t in tgt_feats]
        else:
            tgt_feats = [[p.to(self.device) for p in t] for t in tgt_feats]

        converted_feats = self.convert_vecs(feats, n_feats, tgt_feats)
        n_converted_feats = torch.tensor([f.shape[0] for f in converted_feats])
        converted_batch = pad_sequence(converted_feats, batch_first=True)

        return {
            "wavlm": converted_batch.to(self.device),
            "target": target,
            "n_feats": n_converted_feats,
        }

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
        n_frames[n_frames <= 0] = 1  # TODO: I don't think this is needed

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
