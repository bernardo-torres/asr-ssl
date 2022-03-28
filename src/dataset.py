from pathlib import Path
import re
import librosa
import pandas as pd
from torch.utils.data import Dataset


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'


def remove_special_characters(s):
    return re.sub(chars_to_ignore_regex, '', s.lower() + " ")


def list_collate(batch):
    results = [[] for _ in batch[0]]
    for item in batch:
        for i, x in enumerate(item):
            results[i].append(x)
    return results


class CommonVoiceDataset(Dataset):
    def __init__(self, root_dir, lang, split, sr=16000, augment=None):
        root_dir = Path(root_dir)
        self.lang = lang
        self.lang_dir = root_dir / lang
        self.index_path = self.lang_dir / (split + ".tsv")
        self.index = pd.read_csv(str(self.index_path), delimiter='\t')

        self.sr = sr
        self.augment = augment

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        path = self.lang_dir / "clips" / row['path']
        waveform, _ = librosa.load(str(path), sr=self.sr)
        if self.augment is not None:
            waveform = self.augment(waveform, sample_rate=self.sr)
        text = remove_special_characters(row['sentence'])
        return waveform, text

    def caption(self, idx):
        row = self.index.iloc[idx]
        text = remove_special_characters(row['sentence'])
        return text
