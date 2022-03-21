from torch.utils.data import Dataset
import librosa


def id(x):
    return x


class FilesDataset(Dataset):
    def __init__(self, files, sr=16000, augment=None):
        self.files = files
        self.sr = sr
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, _ = librosa.load(self.files[idx], sr=self.sr)
        if self.augment is not None:
            waveform = self.augment(waveform, sample_rate=self.sr)
        return waveform