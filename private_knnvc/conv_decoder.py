import torch
from torch import nn


class ConvDecoder(nn.Module):
    """
    FastSpeech2 duration predictor (fairseq implementation)
    Copied from fairseq/models/text_to_speech/fastspeech2.py with the following changes:

    1. Changed how arguments are passed.
    2. Replaced the custom dropout module with the one from torch.
    3. Added an embedding layer.
    4. Added default values from the paper.
    5. Output can be either a scalar or a log-probability distribution.
    """

    def __init__(
        self,
        encoder_embed_dim: int = 256,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5,
        output_dim: int = 1,
        emb_dim: int = -1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        if emb_dim > 0:
            self.emb = nn.Embedding(emb_dim, encoder_embed_dim, padding_idx=0)
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                encoder_embed_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_dim, output_dim)

    @torch.inference_mode()
    def forward(self, x):
        # Input: B x T; Output: B x T
        if self.emb_dim > 0:
            x = self.emb(x)

        x = self.conv1(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.dropout1(self.ln1(x))
        x = self.conv2(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.dropout2(self.ln2(x))
        x = self.proj(x).squeeze(dim=2)

        if self.proj.out_features > 1:
            x = x.log_softmax(dim=-1)

        return x


def load_model(ckpt_path: str, device: str) -> ConvDecoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    emb_dim = -1 if "emb.weight" not in ckpt else ckpt["emb.weight"].shape[0]
    model = ConvDecoder(
        encoder_embed_dim=ckpt["conv1.0.weight"].shape[1],
        hidden_dim=ckpt["conv1.0.weight"].shape[0],
        kernel_size=ckpt["conv1.0.weight"].shape[2],
        dropout=0,
        output_dim=ckpt["proj.weight"].shape[0],
        emb_dim=emb_dim,
    ).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    return model
