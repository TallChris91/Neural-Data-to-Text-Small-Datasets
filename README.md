# Neural Data-to-Text Generation Based on Small Datasets: Comparing the Added Value of Two Semi-Supervised Learning Approaches on Top of a Large Language Model

The code for the paper *Neural Data-to-Text Generation Based on Small Datasets: Comparing the Added Value of Two Semi-Supervised Learning Approaches on Top of a Large Language Model* which can be found [here](https://arxiv.org/pdf/2207.06839). This paper introduces two methods for semi-supervised learning, alongside an extensive evaluation, done using both automated metrics as well as a human evaluation. The results of the human evaluation can be found on [Figshare](https://figshare.com/s/3959076f2d69d1381ccc), while this GitHub page reports the results of the automatic metrics, the code used to obtain these results, and the semi-supervised learning scripts.

<h2>The Semi-Supervised Learning Approaches</h2>

Data Augmentation and Pseudo-Labeling were used as the semi-supervised learning methods. Both can be found (with their own documentation) in the `Data_Augmentation` and `Pseudo_Labeling` subfolders.

<h2>Data-to-Text generation</h2>

The data-to-text model used for this study, was [Any2Some](https://github.com/ThiagoCF05/Any2Some). The input files can be found in the `Any2Some` subfolder.

<h2>Evaluation</h3>
The script used to evaluate the results of the human evaluation can be found in the `Evaluation` subfolder.

For automatic text quality metrics, [Any2Some](https://github.com/ThiagoCF05/Any2Some) scripts were used.
For automatic text diversity metrics, [MeasureDiversity](https://github.com/evanmiltenburg/MeasureDiversity) scripts were used.
