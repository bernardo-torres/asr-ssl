from pathlib import Path
from transformers import Wav2Vec2Processor, WavLMModel, WavLMConfig, WavLMForCTC
import torch
from torch.utils.data import DataLoader
import soundfile as sf
from audiomentations import Compose, AddGaussianSNR, PitchShift
from matplotlib import pyplot as plt

from dataset import FilesDataset, id


if __name__ == "__main__":
    data_path = Path('./data/english/clips')
    fs = list(str(x) for x in data_path.glob('*.mp3'))
    augment = Compose([
        AddGaussianSNR(min_snr_in_db=15., max_snr_in_db=25., p=0.8),
        PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
    ])
    ds = FilesDataset(fs, sr=16000, augment=augment)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=id)

    batch = next(iter(loader))
    wav = batch[0]
    #plt.plot(wav)   
    #plt.show()

    

    # WavLM base
    configuration = WavLMConfig()
    model1 = WavLMModel(configuration)
    configuration = model1.config      # Accessing the model configuration
    print(configuration)

    # WavLM with CTC
    pretrained_model = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
    # This model is a fine-tuned version of microsoft/wavlm-base-plus on the LIBRISPEECH_ASR - CLEAN 
    #pretrained_model = "microsoft/wavlm-large"
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
    print('Loaded processor')
    model2 = WavLMForCTC.from_pretrained(pretrained_model)
    print('Loaded model')
    print(model2)

    x = processor(batch, sampling_rate=16000, padding=True, return_tensors='pt')
    with torch.no_grad():
        out1 = model1(x.input_values, output_hidden_states=True)
        out2 = model2(x.input_values, attention_mask=x.attention_mask)

      
    print("Last hidden state shape (batch_size, sequence_length, hidden_size): ", list(out1.last_hidden_state.shape))
    print("Hidden states shapes: ", [list(hid_state.shape) for hid_state in out1.hidden_states])
    print("Extracted features shape (batch_size, sequence_length, conv_dim[-1]):", list(out1.extract_features.shape))
    
    pred_ids = torch.argmax(out2.logits, dim=-1)
    texts = processor.batch_decode(pred_ids)

    for i, (wav, text) in enumerate(zip(batch, texts)):
        path = "test{}.wav".format(i)
        sf.write(path, wav, 16000)
        print("Transcription of {}: {}".format(path, text))
