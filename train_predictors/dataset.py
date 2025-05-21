"""
Loads the audio files from a directory and resamples them if needed.
Returns the waveform as a 1D tensor and the name of the audio file.
! We expect the sample rate to be the same for all the audio files.
"""

import os

from torch import Tensor
from torch.utils.data import Dataset
import torchaudio


class SpeechDataset(Dataset):
    def __init__(self, data_dir: str, sample_rate: int, format: str) -> None:
        """
        Args:
            data_dir: The directory containing the audio files.
            sample_rate: The sampling rate to which the audio should be resampled.
            format: The format of the audio files.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.audiofiles = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(format):
                    self.audiofiles.append(os.path.join(root, f))

        self.folder = data_dir
        self.resampler = None

    def __len__(self) -> int:
        return len(self.audiofiles)

    def __getitem__(self, sample_idx: int) -> tuple[Tensor, str]:
        audiofile = self.audiofiles[sample_idx]
        audio = self.load_audio(os.path.join(self.folder, audiofile))
        return audio, audiofile

    def load_audio(self, audio_path: str) -> Tensor:
        """
        Load the audio from the given path. If the sampling rate is different from
        given sampling rate, resample the audio. Return the waveform as a 1D tensor.
        If the audio is stereo, returns the mean across channels.
        """

        audio, sr = torchaudio.load(audio_path, normalize=True)
        if sr != self.sample_rate:
            if self.resampler is None:
                self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = self.resampler(audio)
        audio = audio.squeeze()
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        return audio
