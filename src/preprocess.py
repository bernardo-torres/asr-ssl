from datasets import ClassLabel
import torch
import re
from datasets import Audio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random
import pandas as pd
from IPython.display import display, HTML
import json
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}



def preprocess(dataset, 
                language,
                ds_name='',
                filter_length= None,
                custom_vocab=None,
                processor=None, 
                tokenizer=None, 
                feature_extractor=None,
                phoneme_language=None, 
                chars_to_ignore_regex=None, 
                verbose=True):

    if chars_to_ignore_regex is None:
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\ʿ\+\)\(]'


    def remove_special_characters(batch):
        batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
        return batch

    # Remove special characters
    dataset = dataset.map(remove_special_characters)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    if custom_vocab is None:
        vocab_ds = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names)
    else:
        vocab_ds = custom_vocab

    if tokenizer is None:

      vocab_list = list(set(vocab_ds['vocab'][0]))
      vocab_dict = {v: k for k, v in enumerate(vocab_list)}
      #print(list)
      vocab_dict["|"] = vocab_dict[" "]
      del vocab_dict[" "]
      vocab_dict["[UNK]"] = len(vocab_dict)
      vocab_dict["[PAD]"] = len(vocab_dict)

      if verbose:
          print(vocab_dict)


      path = language +'_' + ds_name + "_vocab.json"
      with open(path, 'w') as vocab_file:
          json.dump(vocab_dict, vocab_file)
      #processor_factory(dataset, lm_path=lm_path)

      tokenizer = Wav2Vec2CTCTokenizer(
          path,
          unk_token="[UNK]",
          pad_token="[PAD]",
          word_delimiter_token="|",
      )

    if feature_extractor is None:
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.,
            do_normalize=True,
            return_attention_mask=True,
        )

    if processor is None:
        processor = Wav2Vec2Processor(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
            )

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language
            
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"], **additional_kwargs).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=1)

    if filter_length is not None:
        max_input_length_in_sec = filter_length
        dataset = dataset.filter(lambda x: len(x) < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_values"])
        if verbose:
            print(f"Limited dataset to {max_input_length_in_sec} seconds max")
    return dataset, processor, tokenizer, feature_extractor