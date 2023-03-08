# Neural Data-to-Text Generation Based on Small Datasets: Comparing the Added Value of Two Semi-Supervised Learning Approaches on Top of a Large Language Model

The code for the paper *Neural Data-to-Text Generation Based on Small Datasets: Comparing the Added Value of Two Semi-Supervised Learning Approaches on Top of a Large Language Model* which can be found [here](https://arxiv.org/pdf/2207.06839). This paper introduces two methods for semi-supervised learning, alongside an extensive evaluation, done using both automated metrics as well as a human evaluation. The results of the human evaluation can be found on [Figshare](https://figshare.com/s/3959076f2d69d1381ccc), while this GitHub page reports the results of the automatic metrics, the code used to obtain these results, and the semi-supervised learning scripts.

<h2>The Semi-Supervised Learning Approaches</h2>

Data Augmentation and Pseudo-Labeling were used as the semi-supervised learning methods. Both can be found (with their own documentation) in the `Data_Augmentation` and `Pseudo_Labeling` subfolders.

<h2>Data-to-Text generation</h2>

The data-to-text model used for this study, was [Any2Some](https://github.com/ThiagoCF05/Any2Some). The input files can be found in the `Any2Some` subfolder.

<h2>Evaluation</h2>
The script used to evaluate the results of the human evaluation can be found in the `Evaluation` subfolder.

For automatic text quality metrics, [Any2Some](https://github.com/ThiagoCF05/Any2Some) scripts were used.
For automatic text diversity metrics, [MeasureDiversity](https://github.com/evanmiltenburg/MeasureDiversity) scripts were used.

<h2>Results (Diversity)</h2>

| LM | Dataset     | Train type | ASL   | SDSL | Types | TTR1 | TTR2 | %Novel | Cov  | Nov  | Loc1 |
|----|-------------|------------|-------|------|-------|------|------|--------|------|------|------|
| T5 | CACAPO (en) | base       | 17.26 | 8.29 | 3502  | 0.66 | 0.93 | 98.24  | 0.58 | 0.20 | 0.53 |
| T5 | CACAPO (en) | dat_aug    | 17.39 | 8.54 | 3797  | 0.68 | 0.95 | 99.80  | 0.61 | 0.24 | 0.51 |
| T5 | CACAPO (en) | sem_par    | 17.56 | 9.21 | 3709  | 0.67 | 0.93 | 98.50  | 0.60 | 0.22 | 0.57 |
| T5 | CACAPO (nl) | base       | 14.65 | 7.70 | 2748  | 0.58 | 0.85 | 98.86  | 0.56 | 0.20 | 0.52 |
| T5 | CACAPO (nl) | dat_aug    | 14.31 | 6.09 | 2828  | 0.63 | 0.91 | 98.93  | 0.56 | 0.22 | 0.57 |
| T5 | CACAPO (nl) | sem_par    | 14.99 | 6.30 | 3176  | 0.65 | 0.91 | 93.54  | 0.64 | 0.24 | 0.66 |
| T5 | E2E         | base       | 28.58 | 7.66 | 120   | 0.34 | 0.50 | 100.00 | 0.11 | 0.00 | 0.11 |
| T5 | E2E         | dat_aug    | 34.42 | 7.73 | 223   | 0.38 | 0.55 | 100.00 | 0.16 | 0.03 | 0.10 |
| T5 | E2E         | sem_par    | 23.22 | 5.26 | 115   | 0.26 | 0.38 | 100.00 | 0.07 | 0.03 | 0.08 |
| T5 | WebNLG      | base       | 16.01 | 6.71 | 2136  | 0.43 | 0.68 | 79.78  | 0.69 | 0.02 | 0.69 |
| T5 | WebNLG      | dat_aug    | 15.87 | 6.91 | 2311  | 0.43 | 0.71 | 97.71  | 0.62 | 0.15 | 0.49 |
| T5 | WebNLG      | sem_par    | 16.42 | 6.70 | 2404  | 0.45 | 0.72 | 81.13  | 0.73 | 0.08 | 0.66 |

<h2>Results (Text Quality)</h2>

| LM | Dataset     | Train type | BLEU  | NIST | BertScore | METEOR | ROUGE-L |
|----|-------------|------------|-------|------|-----------|--------|---------|
| T5 | CACAPO (en) | base       | 30.5  | 6.77 | 59.51     | 56.05  | 51.34   |
| T5 | CACAPO (en) | dat_aug    | 24.37 | 6.3  | 52.15     | 48.8   | 45.89   |
| T5 | CACAPO (en) | sem_par    | 36.21 | 7.55 | 63.83     | 59.93  | 56.55   |
| T5 | CACAPO (nl) | base       | 33.94 | 6.77 | 84.6      | 52.91  | 51.97   |
| T5 | CACAPO (nl) | dat_aug    | 38.3  | 7.56 | 86.86     | 59.01  | 58.31   |
| T5 | CACAPO (nl) | sem_par    | 54.25 | 9.3  | 89.84     | 68.74  | 68.05   |
| T5 | E2E         | base       | 66.05 | 7.08 | 79.4      | 80.21  | 44.97   |
| T5 | E2E         | dat_aug    | 28.41 | 4.15 | 56.41     | 62.49  | 33.4    |
| T5 | E2E         | sem_par    | 50.51 | 4.65 | 63.12     | 60.39  | 38.48   |
| T5 | WebNLG      | base       | 47.91 | 8.74 | 71.3      | 71.57  | 59.88   |
| T5 | WebNLG      | dat_aug    | 27.71 | 5.95 | 52.75     | 53.23  | 45.33   |
| T5 | WebNLG      | sem_par    | 44.55 | 8.32 | 67.82     | 68.74  | 56.82   |

| LM   | Dataset     | Train type | BLEU  | NIST | BertScore | METEOR | ROUGE-L |
|------|-------------|------------|-------|------|-----------|--------|---------|
| BART | CACAPO (en) | base       | 32.96 | 7.33 | 60.31     | 57.33  | 52.34   |
| BART | CACAPO (en) | dat_aug    | 23.65 | 6.29 | 49.38     | 47.98  | 44.58   |
| BART | CACAPO (en) | sem_par    | 37.37 | 7.87 | 64.12     | 60.6   | 56.68   |
| BART | CACAPO (nl) | base       | 42.47 | 7.77 | 84.82     | 60.61  | 59.29   |
| BART | CACAPO (nl) | dat_aug    | 37.57 | 7.49 | 86.22     | 58.47  | 57      |
| BART | CACAPO (nl) | sem_par    | 52.47 | 9.12 | 89.46     | 67.56  | 66.46   |
| BART | E2E         | base       | 66.03 | 7.14 | 79.25     | 79.44  | 45.09   |
| BART | E2E         | dat_aug    | 26.95 | 3.9  | 51.82     | 59.16  | 32.22   |
| BART | E2E         | sem_par    | 47.78 | 3.9  | 61.98     | 59.52  | 39.57   |
| BART | WebNLG      | base       | 47.28 | 8.69 | 71.19     | 71.33  | 59.35   |
| BART | WebNLG      | dat_aug    | 21.21 | 4.95 | 45.61     | 46.03  | 39.51   |
| BART | WebNLG      | sem_par    | 30.88 | 6.2  | 51.38     | 53.98  | 44.13   |

| LM   | Dataset     | Train type | BLEU  | NIST | BertScore | METEOR | ROUGE-L |
|------|-------------|------------|-------|------|-----------|--------|---------|
| GPT2 | CACAPO (en) | base       | 22.08 | 5.7  | 47.63     | 46.72  | 41.87   |
| GPT2 | CACAPO (en) | dat_aug    | 19.54 | 5.49 | 42.77     | 43.27  | 40.01   |
| GPT2 | CACAPO (en) | sem_par    | 30.88 | 7.03 | 58.4      | 55.91  | 51.68   |
| GPT2 | CACAPO (nl) | base       | 34.94 | 6.74 | 84.85     | 53.68  | 52.95   |
| GPT2 | CACAPO (nl) | dat_aug    | 34.15 | 6.89 | 84.9      | 55.28  | 53.54   |
| GPT2 | CACAPO (nl) | sem_par    | 46.78 | 8.41 | 88.29     | 63.88  | 63.34   |
| GPT2 | E2E         | base       | 65.26 | 7    | 78.77     | 80.24  | 44.8    |
| GPT2 | E2E         | dat_aug    | 28.76 | 3.84 | 53.61     | 62.21  | 32.33   |
| GPT2 | E2E         | sem_par    | 51.59 | 4.17 | 64        | 61.36  | 40.11   |
| GPT2 | WebNLG      | base       | 44.21 | 8.22 | 67.11     | 68.03  | 56.88   |
| GPT2 | WebNLG      | dat_aug    | 22.07 | 4.98 | 44.56     | 45.69  | 39.66   |
| GPT2 | WebNLG      | sem_par    | 32.89 | 6.61 | 54.8      | 57.47  | 46.15   |
