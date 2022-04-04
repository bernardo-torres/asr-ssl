#!/usr/bin/env python3
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
from nbformat import write
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from packaging import version
from torch import nn

from model import model_factory

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from preprocess import preprocess, DataCollatorCTCWithPadding


if is_apex_available():
    from apex import amp


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default='wav2vec',
        metadata={"help": "Model type, wav2vec, wavlm or hubert"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    freeze_base_model: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze all of the base model except the classification head."}
    )
    attention_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."},
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
        },
    )
    mask_feature_prob: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The"
            "masking procecure generates ''mask_feature_prob*len(feature_axis)/mask_time_length'' independent masks over"
            "the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector"
            "span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap"
            "may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is"
            "True`."
        },
    )
    layerdrop: Optional[float] = field(default=0.0, metadata={"help": "The LayerDrop probability."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    script_dir: Optional[str] = field(
        default="/content/asr-ssl/src/common_voice_hf_loader.py", metadata={"help": "Common voice loading script (local)"}
    )
    data_dir: Optional[str] = field(
        default="/content/drive/MyDrive/asr_data/data", metadata={"help": "Common voice data dir (local)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library). Equivalent to language"}
    )
    train_split_name: Optional[str] = field(
        default="1h",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to '1h'"
        },
    )
    train_filter_length: Optional[float] = field(
        default=None,
        metadata={
            "help": "Number of seconds to limit training samples. Defaults to 'None'"
        },
    )
    test_filter_length: Optional[float] = field(
        default=None,
        metadata={
            "help": "Number of seconds to limit test samples. Defaults to 'None'"
        },
    )
    download_mode: str = field(
        default='reuse_cache_if_exists', 
        metadata={"help": "reuse_cache_if_exists (default), force_redownload, reuse_dataset_if_exists."}
    )
    overwrite_cache: Optional[str] = field(
        default='False', metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    chars_to_ignore: List[str] = list_field(
        default=[",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�"],
        metadata={"help": "A list of characters to remove from the transcripts."},
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
    trim_train_clips: Optional[str] = field(
        default=False,
        metadata={
            "help": "Controls if the silent parts of train clips should be trimmed or not."
        }
    )
    trim_test_clips: Optional[str] = field(
        default=False,
        metadata={
            "help": "Controls if the silent parts of test clips should be trimmed or not."
        }
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    #writer = SummaryWriter(training_args.log_dir)  # tensorboard


    # Get the datasets:

    train_dataset = datasets.load_dataset(
        data_args.script_dir, data_args.dataset_config_name, split=data_args.train_split_name, data_dir=data_args.data_dir, cache_dir=model_args.cache_dir, download_mode=data_args.download_mode
    )
    eval_dataset = datasets.load_dataset(
        data_args.script_dir, data_args.dataset_config_name, split="test", data_dir=data_args.data_dir, cache_dir=model_args.cache_dir, download_mode=data_args.download_mode
    )

    print('------------------------------')

    assert len(train_dataset) > 0
    print('Succesfully loaded train dataset ' + data_args.train_split_name + ' for language '+data_args.dataset_config_name)
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(eval_dataset)}')
    print('------------------------------')
    # Create and save tokenizer
    """chars_to_ignore_regex = f'[{"".join(data_args.chars_to_ignore)}]'

    def remove_special_characters(batch):
        batch["text"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).lower() + " "
        return batch

    train_dataset = train_dataset.map(remove_special_characters, remove_columns=["sentence"])
    eval_dataset = eval_dataset.map(remove_special_characters, remove_columns=["sentence"])

    def extract_all_chars(batch):
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocab_train = train_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_dataset.column_names,
    )
    vocab_test = train_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=eval_dataset.column_names,
    )

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open("vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)"""

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
        print(f'Using {data_args.max_train_samples} train samples')

    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        print(f'Using {data_args.max_val_samples} test samples')

    phoneme_language = data_args.phoneme_language

    train_dataset, processor, tokenizer, feature_extractor = preprocess(train_dataset, 
                                                                    data_args.dataset_config_name, 
                                                                    ds_name=data_args.train_split_name,
                                                                    filter_length=data_args.train_filter_length,
                                                                    trim=data_args.trim_train_clips,
                                                                    custom_vocab=None,
                                                                    phoneme_language=phoneme_language,
                                                                    #custom_vocab=vocab_fr, 
                                                                    verbose=True)
    print('Preprocessing train data done')
    print('------------------------------')
    eval_dataset, _, _, _ = preprocess(eval_dataset, 
                                        data_args.dataset_config_name, 
                                        custom_vocab=None, 
                                        phoneme_language=phoneme_language,
                                        filter_length=data_args.test_filter_length,
                                        trim=data_args.trim_test_clips,
                                        processor=processor, 
                                        tokenizer=tokenizer, 
                                        feature_extractor=feature_extractor,
                                        verbose=True)
    print('Preprocessing test data done')

    print('------------------------------')


    model = model_factory(model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            activation_dropout=model_args.activation_dropout,
            attention_dropout=model_args.attention_dropout,
            hidden_dropout=model_args.hidden_dropout,
            feat_proj_dropout=model_args.feat_proj_dropout,
            mask_time_prob=model_args.mask_time_prob,
            mask_feature_prob=model_args.mask_feature_prob,
            gradient_checkpointing=training_args.gradient_checkpointing,
            layerdrop=model_args.layerdrop,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            model_type=model_args.model_type)

    """model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
        gradient_checkpointing=training_args.gradient_checkpointing,
        layerdrop=model_args.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )"""

    print('Loaded model '+ model_args.model_name_or_path)

    # Preprocessing the datasets.
    # We need to read the aduio files as arrays and tokenize the targets.
    """def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = resampler(speech_array).squeeze().numpy()
        batch["sampling_rate"] = 16_000
        batch["target_text"] = batch["sentence"]
        return batch

    train_dataset = train_dataset.map(
        speech_file_to_array_fn,
        remove_columns=train_dataset.column_names,
        num_proc=data_args.preprocessing_num_workers,
    )
    eval_dataset = eval_dataset.map(
        speech_file_to_array_fn,
        remove_columns=eval_dataset.column_names,
        num_proc=data_args.preprocessing_num_workers,
    )"""

    """def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(batch["array"], sampling_rate=batch["sampling_rate"][0]).input_values
        # Setup the processor for targets
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=train_dataset.column_names,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )
    eval_dataset = eval_dataset.map(
        prepare_dataset,
        remove_columns=eval_dataset.column_names,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )"""

    # Metric
    wer_metric = datasets.load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        if phoneme_language is None:    
            return {"wer": wer}
        else:
            return {"per": wer}

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()
        print('Freezed feature extractor')
    else:
        print('Did not freeze feature extractor')

    if model_args.freeze_base_model:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = True
        print('Only training classification head')
    else:
        print('Fine tuning model')
    # model.gradient_checkpointing_enable()
    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Initialize our Trainer
    """trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )"""

    trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=processor.feature_extractor,
)

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        # Save the feature_extractor and the tokenizer
        if is_main_process(training_args.local_rank):
            processor.save_pretrained(training_args.output_dir)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return results


if __name__ == "__main__":
    main()