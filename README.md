# asr-ssl

## Preparation

#### Prerequisites

- See `requirements.txt`.

#### Dataset

1. The data structure should look like:

   ```bash
   /data/
   ├── fr.zip
   |   ├── 10m.tsv
   |   ├── 1h.tsv
   |   ├── 10h.tsv
   |   ├── test.tsv
   |   ├── clips
   |     ├── file1.mp3
   |     ├── file2.mp3
   │     ...
   ├── en.zip
   |   ├── 10m.tsv
   |   ├── 1h.tsv
   |   ├── 10h.tsv
   |   ├── test.tsv
   |   ├── clips
   |     ├── file1.mp3
   |     ├── file2.mp3
   │     ...
   ├── _language_.zip
   |   ├── 10m.tsv
   |   ├── 1h.tsv
   |   ├── 10h.tsv
   |   ├── test.tsv
   |   ├── clips
   |     ├── file1.mp3
   |     ├── file2.mp3
   │     ...
   ...
       
   ```
   #### Training example
   

Finetuning wav2vec on Automatic Phoneme Recognition (APR) on 1h data partition of hungarian dataset using espeak backend to convert to hungarian phonemes. 
```python train.py \
    --model_name_or_path="facebook/wav2vec2-base" \
    --model_type="wav2vec" \
    --data_dir="path/data" \
    --cache_dir="/train_cache" \
    --script_dir="common_voice_hf_loader.py" \
    --phoneme_language="hu" \
    --dataset_config_name="hu"\
    --train_split_name="1h" \
    --train_filter_length=7.0 \
    --output_dir=$output_dir \
    --num_train_epochs="200" \
    --freeze_feature_extractor=True \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --gradient_accumulation_steps='2'\
    --evaluation_strategy="steps" \
    --learning_rate="1e-4" \
    --warmup_steps="500" \
    --fp16 \
    --freeze_feature_extractor \
    --save_steps="200" \
    --eval_steps="200" \
    --save_total_limit="2" \
    --logging_steps="100" \
    --group_by_length \
    --gradient_checkpointing \
    --do_train --do_eval \
    --fp16=True \
    --max_val_samples=1000 \
    --overwrite_output_dir \
    --load_best_model_at_end=True \
    --report_to="tensorboard"
   ```
   
   Finetuning WavLM on Automatic Speech Recognition (ASR) on 10m data partition of english (no phoneme transcription). 
```python train.py \
    --model_name_or_path="patrickvonplaten/wavlm-libri-clean-100h-base-plus" \
    --model_type="wavlm" \
    --data_dir="path/data" \
    --cache_dir="/train_cache" \
    --script_dir="common_voice_hf_loader.py" \
    --dataset_config_name="en"\
    --train_split_name="10m" \
    --train_filter_length=7.0 \
    --output_dir="myresults/wavlm-en-10m" \
    --num_train_epochs="200" \
    --freeze_feature_extractor=True \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --gradient_accumulation_steps='2'\
    --evaluation_strategy="steps" \
    --learning_rate="1e-4" \
    --warmup_steps="500" \
    --fp16 \
    --freeze_feature_extractor \
    --save_steps="200" \
    --eval_steps="200" \
    --save_total_limit="2" \
    --logging_steps="100" \
    --group_by_length \
    --gradient_checkpointing \
    --do_train --do_eval \
    --fp16=True \
    --max_val_samples=2000 \
    --overwrite_output_dir \
    --load_best_model_at_end=True \
    --report_to="tensorboard"
   ```
   
   Adding flag `--phoneme_language="en-us"` would run APR instead.
   
   #### Evaluation
   
   Evaluating ASR on full test data.
   
   ```
   python test.py \
    --model_name_or_path='/myresults/wavlm-en-10m' \
    --language='en' \
    --cache_dir="/content/downloads" \
    --device='cuda' \
    --script_dir="common_voice_hf_loader.py" \
    --data_dir='/path/data' \
   ```
   
   Adding flag `--phoneme_language="en-us"` would run APR evaluation instead. This takes longer.
   
   #### TODO
   - Add support for APR
   - Add support for choosing the model
   - Change training step function and add support for logging loss curves
   - 
