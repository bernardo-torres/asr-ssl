import json
import kenlm
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor, Wav2Vec2ProcessorWithLM)
from pyctcdecode import build_ctcdecoder


def processor_factory(ds, lm_path=None):
    captions = [ds.caption(i) for i in range(len(ds))]
    all_text = " ".join(captions)
    vocab_list = list(set(all_text))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    path = ds.lang + "_vocab.json"
    with open(path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer(
        "./vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=ds.sr,
        padding_value=0.,
        do_normalize=True,
        return_attention_mask=True,
    )

    if lm_path is not None:
        kenlm_model = kenlm.Model(lm_path)
        decoder = build_ctcdecoder(
            vocab_list,
            kenlm_model,
        )
        processor = Wav2Vec2ProcessorWithLM(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            decoder=decoder,
        )
    else:
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

    return processor
