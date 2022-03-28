from transformers import Wav2Vec2Processor, HubertForCTC
import torch
from torch.utils.data import DataLoader
import soundfile as sf
from audiomentations import Compose, AddGaussianSNR, PitchShift
from matplotlib import pyplot as plt

from dataset import CommonVoiceDataset, list_collate


if __name__ == "__main__":
    augment = Compose([
        AddGaussianSNR(min_snr_in_db=15., max_snr_in_db=25., p=0.8),
        PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
    ])
    ds = CommonVoiceDataset("../data", "fr", "10min",
                            sr=16000, augment=augment)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2,
                        collate_fn=list_collate)

    wavs, gt_texts = next(iter(loader))
    plt.plot(wavs[0])
    plt.show()

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

    x = processor(wavs, sampling_rate=16000, padding=True, return_tensors='pt')
    with torch.no_grad():
        out = model(x.input_values, attention_mask=x.attention_mask)

    pred_ids = torch.argmax(out.logits, dim=-1)
    pred_texts = processor.batch_decode(pred_ids)

    for i, (wav, gt_text, pred_text) in enumerate(zip(wavs, gt_texts, pred_texts)):
        path = "test{}.wav".format(i)
        sf.write(path, wav, 16000)
        print("GT transcription of {}: {}".format(path, gt_text))
        print("Predicted transcription of {}: {}".format(path, pred_text))
