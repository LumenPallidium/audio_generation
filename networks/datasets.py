import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torchaudio
import torch
import numpy as np
from torch.utils.data import Dataset

# PORTED FROM HERE:
# https://pytorch.org/audio/stable/_modules/torchaudio/datasets/commonvoice.html#COMMONVOICE
# needed to fix some things
def load_commonvoice_item(
    line: List[str], header: List[str], path: str, folder_audio: str, ext_audio: str
) -> Tuple[torch.Tensor, int, Dict[str, str]]:
    # Each line as the following data:
    # client_id, path, sentence, up_votes, down_votes, age, gender, accent

    if header[1] != "path":
        raise ValueError(f"expect `header[1]` to be 'path', but got {header[1]}")
    fileid = line[1]
    filename = os.path.join(path, folder_audio, fileid)
    if not filename.endswith(ext_audio):
        filename += ext_audio
    waveform, sample_rate = torchaudio.load(filename)

    dic = dict(zip(header, line))

    return waveform, sample_rate, dic

class COMMONVOICE(Dataset):
    """*CommonVoice* :cite:`ardila2020common` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is located.
             (Where the ``tsv`` file is present.)
        tsv (str, optional):
            The name of the tsv file used to construct the metadata, such as
            ``"train.tsv"``, ``"test.tsv"``, ``"dev.tsv"``, ``"invalidated.tsv"``,
            ``"validated.tsv"`` and ``"other.tsv"``. (default: ``"train.tsv"``)
    """

    _ext_txt = ".txt"
    _ext_audio = ".wav"
    _folder_audio = "clips"

    def __init__(self, root: Union[str, Path], tsv: str = "train.tsv") -> None:

        # Get string representation of 'root' in case Path object is passed
        self._path = os.fspath(root)
        self._tsv = os.path.join(self._path, tsv)

        with open(self._tsv, "r", encoding = "utf-8") as tsv_:
            walker = csv.reader(tsv_, delimiter="\t")
            self._header = next(walker)
            self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, Dict[str, str]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            Dict[str, str]:
                Dictionary containing the following items from the corresponding TSV file;

                * ``"client_id"``
                * ``"path"``
                * ``"sentence"``
                * ``"up_votes"``
                * ``"down_votes"``
                * ``"age"``
                * ``"gender"``
                * ``"accent"``
        """
        line = self._walker[n]
        return load_commonvoice_item(line, self._header, self._path, self._folder_audio, self._ext_audio)


    def __len__(self) -> int:
        return len(self._walker)
    
def overtones(shape, freq, n_overtones=5, sample_rate=16000, max = 0.125, normalize = True):
    interval = torch.arange(shape[-1], dtype=torch.float32) / sample_rate
    interval = interval * (freq * 2 * torch.pi)
    overtones = torch.sin(interval.unsqueeze(0) * torch.arange(1, n_overtones + 1).unsqueeze(1)).mean(0)
    overtones = overtones / overtones.abs().max() * max
    return overtones

def introduce_disharmony(waveform, 
                         magnitude_scale=0.5, 
                         offset_range = (1, 80),
                         n_segments = None,
                         segment_length = None):
    spectrum = torch.fft.fft(waveform)
    magnitude_spectrum = torch.abs(spectrum)

    max_magnitude = magnitude_spectrum.max()

    if segment_length is not None:
        n_segments = waveform.shape[-1] // segment_length

    b, c, l = waveform.shape
    if n_segments is not None:
        segment_length = l // n_segments
        spectrum = spectrum.view(b, c, n_segments, segment_length)
        magnitude_spectrum = magnitude_spectrum.view(b, c, n_segments, segment_length)

    # max along the segments
    _, dominant_idx = torch.max(magnitude_spectrum, dim=-1)



    offset = torch.randint(*offset_range, size=dominant_idx.shape)
    disharmonic_idx = dominant_idx + offset

    # dummy indices for help
    b_indices = torch.arange(b)[:, None, None]
    c_indices = torch.arange(c)[None, :, None]
    n_indices = torch.arange(n_segments)[None, None, :]

    spectrum[b_indices, c_indices, n_indices, disharmonic_idx] += magnitude_scale * max_magnitude

    # coerce and return to time basis
    spectrum = spectrum.view(b, c, l)
    modified_waveform = torch.fft.ifft(spectrum).real

    return modified_waveform


#TODO : doesn't seem to have modified audio in segments?
if __name__ == "__main__":
    from IPython.display import Audio
    om = torchaudio.load(r"om.wav")[0].unsqueeze(0)
    tmp = introduce_disharmony(om, magnitude_scale=100, n_segments=4).squeeze(0)
    Audio(tmp.detach().cpu().numpy(), rate = 16000)

    
