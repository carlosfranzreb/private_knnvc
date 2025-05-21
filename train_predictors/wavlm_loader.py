import torch

from wavlm.wavlm_model import WavLM, WavLMConfig


def load_wavlm(device: str):
    ckpt = torch.load(
        "checkpoints/WavLM-Large.pt",
        map_location=device,
        weights_only=False,
    )
    model_cfg = WavLMConfig(ckpt["cfg"])
    model = WavLM(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
