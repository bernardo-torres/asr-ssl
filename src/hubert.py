from pathlib import Path
from transformers import Wav2Vec2Processor, HubertModel, HubertForCTC
import torch
from torch.utils.data import DataLoader
import soundfile as sf
from audiomentations import Compose, AddGaussianSNR, PitchShift
from matplotlib import pyplot as plt

from src.dataset import FilesDataset, id


if __name__ == "__main__":
    data_path = Path('data/french/clips')
    fs = list(str(x) for x in data_path.glob('*.mp3'))
    augment = Compose([
        AddGaussianSNR(min_snr_in_db=15., max_snr_in_db=25., p=0.8),
        PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
    ])
    ds = FilesDataset(fs, sr=16000, augment=augment)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=id)

    batch = next(iter(loader))
    wav = batch[0]
    plt.plot(wav)
    plt.show()

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

    x = processor(batch, sampling_rate=16000, padding=True, return_tensors='pt')
    with torch.no_grad():
        out = model(x.input_values, attention_mask=x.attention_mask)
    
    pred_ids = torch.argmax(out.logits, dim=-1)
    texts = processor.batch_decode(pred_ids)

    for i, (wav, text) in enumerate(zip(batch, texts)):
        path = "test{}.wav".format(i)
        sf.write(path, wav, 16000)
        print("Transcription of {}: {}".format(path, text))
