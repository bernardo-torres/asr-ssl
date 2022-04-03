import torch
from transformers import Wav2Vec2ForCTC
import numpy as np
from datasets import load_dataset, load_metric
import random
import pandas as pd
from IPython.display import display, HTML

#!/usr/bin/env python3
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch


from model import model_factory
from preprocess import preprocess

from transformers import (
    HfArgumentParser,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    set_seed,
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    language: str = field(
        metadata={"help": "language of trained model"}
    )
    phoneme_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "The target language that should be used be"
            " passed to the tokenizer for tokenization. Note that"
            " this is only relevant if the model classifies the"
            " input audio to a sequence of phoneme sequences."
            "https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md"

        },
    )
    cache_dir: Optional[str] = field(
        default="/content/downloads",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    
    device: str = field(
        default="cuda",
        metadata={"help": "device for tensors/model"}
    )

    
@dataclass
class DataTestingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    script_dir: Optional[str] = field(
        default="/content/asr-ssl/src/common_voice_hf_loader.py", metadata={"help": "Common voice loading script (local)"}
    )
    data_dir: Optional[str] = field(
        default="/content/drive/MyDrive/asr_data/data", metadata={"help": "Common voice data dir (local)"}
    )
    num_test_elements: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of elements to perform testing."
        },
    )
    
    
def main():
    parser = HfArgumentParser((ModelArguments, DataTestingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args= parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args= parser.parse_args_into_dataclasses()

    device = torch.device(model_args.device)
    path = model_args.model_name_or_path#'/content/drive/MyDrive/asr_results/wav2vec2-hu10h' #model argument
    language = model_args.language#'hu' # model argument
    phoneme_language = model_args.phoneme_language
    processor = Wav2Vec2Processor.load(path)
    model =  Wav2Vec2ForCTC.load(path).to(device)

    def map_to_result(batch):
      with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
        logits = model(input_values).logits

      pred_ids = torch.argmax(logits, dim=-1)
      batch["pred_str"] = processor.batch_decode(pred_ids)[0]
      batch["text"] = processor.decode(batch["labels"], group_tokens=False)
      #batch["text"] = phonemize(batch["text"])
      return batch

    wer_metric = load_metric("wer")


    def show_random_elements(dataset, num_examples=10):
        assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
        picks = []
        for _ in range(num_examples):
            pick = random.randint(0, len(dataset)-1)
            while pick in picks:
                pick = random.randint(0, len(dataset)-1)
            picks.append(pick)
        
        df = pd.DataFrame(dataset[picks])
        #display(HTML(df.to_html()))
        display(df)  # HTML does not work when running from terminal
    
    data_dir = data_args.data_dir#'/content/drive/MyDrive/asr_data/data' # data argument
    script_dir = data_args.script_dir#"/content/asr-ssl/src/common_voice_hf_loader.py" #data argument
    #language = 'hu'  # fr, en, hu # model argument
    
    cache_dir = model_args.cache_dir
    
    ds_test = load_dataset(script_dir, language, split='test', data_dir=data_dir, cache_dir=cache_dir,  download_mode='force_redownload') # __data argument__
    
    ds_test = ds_test.remove_columns([columns for columns in ds_test.column_names if not ['path', 'audio', 'sentence']])
  
    if phoneme_language is None:
        print(f'Processing data in ASR mode (not phoneme), language {language}')
    else:
        print(f'Processing data in phoneme mode, configuration {phoneme_language}')
    print('Processing test data done')
    dataset_test, _, _, _ = preprocess(ds_test, 
                                        language, 
                                        custom_vocab=None, 
                                        processor=processor, 
                                        tokenizer=processor.tokenizer, 
                                        feature_extractor=processor.feature_extractor,
                                       phoneme_language=phoneme_language, # __model argument__
                                        verbose=True)
    print('Processing test data done')
    
    
    if data_args.num_test_elements is None:
        results = dataset_test.map(map_to_result, remove_columns=dataset_test.column_names)
        print(f"Computing test results for full test data")
    else:
        results = dataset_test.select(list(range(data_args.num_test_elements))).map(map_to_result, remove_columns=dataset_test.select(list(range(data_args.num_test_elements))).column_names)
        print(f"Computing test results for {data_args.num_test_elements} elements")

    print("\n\n======== Test Results==========\n")

    print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

    print("\nExemples of predictions\n")
    #counter = 0
    #for row in results:
    #  counter+=1
    #  print(row)
    #  if counter == 10:
    #    break
    show_random_elements(results)
    
    
if __name__ == "__main__":
    main()