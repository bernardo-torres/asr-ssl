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
   
   #### TODO
   - Add support for APR
   - Add support for choosing the model
   - Change training step function and add support for logging loss curves
   - 
