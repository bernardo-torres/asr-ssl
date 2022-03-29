# asr-ssl

## Preparation

#### Prerequisites

- See `requirements.txt`.

#### Dataset

1. The data structure should look like:

   ```bash
   /data/
   ├── fr
   |   ├── 10m.tsv
   |   ├── 1h.tsv
   |   ├── 10h.tsv
   |   ├── test.tsv
   |   ├── clips
   |     ├── file1.mp3
   |     ├── file2.mp3
   │     ...
   ├── en
   |   ├── 10m.tsv
   |   ├── 1h.tsv
   |   ├── 10h.tsv
   |   ├── test.tsv
   |   ├── clips
   |     ├── file1.mp3
   |     ├── file2.mp3
   │     ...
   ├── _language_
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
