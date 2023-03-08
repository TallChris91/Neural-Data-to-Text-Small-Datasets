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

| LM | Dataset     | Train type | ASL   | SDSL | Types | TTR1 | TTR2 | %Novel | Cov  | Nov  | Loc5 |
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
