# Intrusion Detection System (IDS) Project

<!-- OVERVIEW_START -->

## Overview

I built an academic Intrusion Detection System (IDS) project in Google Colab using **NSL-KDD** and **CICIDS2017** datasets. The goal is to evaluate how well traditional machine learning models detect attacks in a **binary classification** setting (**0 = Normal**, **1 = Attack**) under both **intra-dataset** and **cross-dataset** testing. I implemented **Decision Tree (DT)**, **Random Forest (RF)**, and **Logistic Regression (LR)**, saved all metrics as JSON, and generated figures such as confusion matrices, ROC curves (where applicable), and error histograms. The full report is maintained in the project outputs folder. 

<!-- OVERVIEW_END -->

<!-- INSTALLATION_START -->

## Installation

This project was designed and executed in **Google Colab**, which simplifies dependency management and provides sufficient computational resources for handling the CICIDS2017 dataset. Running the experiments locally is possible, but additional memory management may be required.

### Prerequisites
- **Python 3.8 or later**
- Adequate memory, especially when working with **CICIDS2017**, which is significantly larger than NSL-KDD
- Sufficient disk space for datasets, generated figures, and saved JSON metrics

### Environment setup
I recommend using **Google Colab** for reproducibility and ease of execution. All experiments in this project were conducted in Colab, with dataset sampling applied where necessary to control memory usage.

### Dependencies
The required Python packages can be installed directly inside a Google Colab notebook cell using:

pip install -r requirements.txt

### Main libraries used
- **pandas** for data loading and preprocessing
- **numpy** for numerical computations
- **scikit-learn** for preprocessing pipelines, machine learning models, and evaluation metrics
- **imbalanced-learn** for applying SMOTE strictly on training data
- **matplotlib** for visualizing confusion matrices, ROC curves, and performance distributions

<!-- INSTALLATION_END -->

<!-- PROJECT_STRUCTURE_START -->

## Project Structure

The project is organized using a clear and reproducible structure commonly adopted in cybersecurity and machine learning research. Datasets, notebooks, and experimental outputs are separated to support both intra-dataset and cross-dataset evaluation.

```
IDS_Project/
├── data/
│   └── raw/
│       ├── NSL-KDD/            # NSL-KDD dataset files
│       └── CICIDS2017/         # CICIDS2017 dataset files
├── notebooks/                 # Google Colab notebooks used for experiments
├── outputs_article_analysis/  # Full article-oriented experimental outputs
│   ├── NSL-KDD/               # Intra-dataset results for NSL-KDD
│   ├── CICIDS2017/            # Intra-dataset results for CICIDS2017
│   ├── cross_dataset_CICIDT/  # Cross-dataset results (Decision Tree)
│   ├── cross_dataset_CICIRF/  # Cross-dataset results (Random Forest)
│   ├── cross_dataset_CICILR/  # Cross-dataset results (Logistic Regression, if generated)
│   └── article_report.md      # Automatically generated academic report
└── README.md                  # Project documentation (this file)
```

All evaluation metrics are stored as JSON files, while visual outputs such as confusion matrices, ROC curves, and F1-score histograms are saved as PNG images inside their respective result directories.

<!-- PROJECT_STRUCTURE_END -->

<!-- USAGE_START -->

## Usage

All experiments in this project are executed through **Google Colab notebooks**. The workflow is designed so that running the notebooks sequentially reproduces the full experimental pipeline, including preprocessing, model training, and evaluation.

### Typical workflow
1. Load the NSL-KDD and CICIDS2017 datasets from the `data/raw/` directory.
2. Apply dataset-specific preprocessing:
   - Encoding and scaling for NSL-KDD.
   - Feature scaling for CICIDS2017.
3. Split each dataset into training and testing sets.
4. Apply **SMOTE only on the training data** to address class imbalance.
5. Train the selected machine learning models (Decision Tree, Random Forest, and Logistic Regression).
6. Perform intra-dataset evaluation and cross-dataset evaluation.

### Output generation
During execution, the notebooks produce evaluation metrics and visualizations that are saved to the corresponding subdirectories inside `outputs_article_analysis/`. These outputs are then used for analysis and comparison across datasets and models.

<!-- USAGE_END -->

<!-- RESULTS_START -->

## Results

### Intra-dataset performance

| Dataset | Model | Accuracy | Precision (Attack) | Recall (Attack) | F1 (Attack) | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | Decision Tree | 99.47% | 99.44% | 99.46% | 99.45% | 0.9979 |
| NSL-KDD | Random Forest | 99.61% | 99.67% | 99.52% | 99.60% | 0.9997 |
| NSL-KDD | intra_lr_metrics | 95.49% | 95.94% | 94.63% | 95.28% | 0.9901 |
| CICIDS2017 | intra_dt_metrics | 99.85% | 99.54% | 99.59% | 99.57% | 0.9979 |
| CICIDS2017 | intra_rf_metrics | 99.86% | 99.56% | 99.63% | 99.59% | 0.9997 |
| CICIDS2017 | intra_lr_metrics | 89.55% | 62.35% | 96.18% | 75.66% | 0.9732 |

### Cross-dataset performance

| Train | Test | Model | Accuracy | Precision (Attack) | Recall (Attack) | F1 (Attack) | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | cross_dataset_CICILR | 51.88% | 0.00% | 0.00% | 0.00% | 0.4905 |
|  |  | cross_dataset_CICILR | 16.94% | 16.94% | 100.00% | 28.97% | 0.5392 |
|  |  | cross_dataset_CICIRF | 68.53% | 11.97% | 13.50% | 12.69% | 0.4923 |
|  |  | cross_dataset_CICIRF | 51.88% | 0.00% | 0.00% | 0.00% | 0.4989 |
|  |  | cross_dataset_CICIDT | 52.15% | 100.00% | 0.55% | 1.10% | 0.5028 |
|  |  | cross_dataset_CICIDT | 16.94% | 16.94% | 100.00% | 28.97% | 0.5000 |
<!-- RESULTS_END -->

<!-- GENERATED_OUTPUTS_START -->

## Generated Outputs

All experimental outputs produced during this project are organized inside the `outputs_article_analysis/` directory. These outputs support quantitative analysis, visual inspection, and academic reporting.

### CSV tables
The following CSV files summarize key experimental results and dataset characteristics:

- **dataset_comparison.csv** — overview and comparison of dataset properties
- **performance_comparison.csv** — consolidated model performance metrics
- **feature_analysis.csv** — feature distributions and statistical analysis

### Figures
A set of figures is plotted to visualize class imbalance, model performance, and cross-dataset behavior:

- Class distribution and performance comparison plots
- Cross-dataset transfer visualizations for Random Forest, Logistic Regression, and Decision Tree models
- Decision Tree visualizations for both NSL-KDD and CICIDS2017

In addition, the file **article_report.md** contains the complete article-style report that integrates tables, figures, and written analysis derived from the experimental results.
<!-- GENERATED_OUTPUTS_END -->

<!-- KEY_FINDINGS_START -->

## Key Findings

Based on the conducted experiments, several high-level observations can be drawn regarding model behavior and dataset characteristics. These findings are intended to summarize trends observed across experiments rather than assert absolute performance rankings.

- Model performance varies noticeably between datasets, highlighting the influence of dataset composition, feature distributions, and class imbalance on intrusion detection.
- Intra-dataset evaluations generally show stronger performance than cross-dataset evaluations, indicating that models benefit from being trained and tested on data drawn from the same distribution.
- Cross-dataset experiments reveal challenges in generalization, suggesting that learned decision boundaries do not always transfer cleanly between different intrusion detection datasets.
- Differences in behavior across models point to trade-offs between fitting capacity and robustness, which become more apparent when moving from intra-dataset to cross-dataset testing.
- Feature representation and preprocessing choices play a significant role in shaping model outcomes, particularly when aligning datasets with different original feature spaces.

These observations motivate a more detailed analysis of model generalization and robustness, which is explored further in the accompanying article report.
<!-- KEY_FINDINGS_END -->

<!-- TECHNICAL_DETAILS_START -->

## Technical Details

I implemented this project using traditional machine learning techniques for binary intrusion detection, with a strong emphasis on reproducibility and fair evaluation across datasets. All experiments were conducted in Google Colab.

### Datasets
I used two widely studied intrusion detection datasets: **NSL-KDD** and **CICIDS2017**. Both datasets were reformulated into a binary classification task, where normal traffic is labeled as 0 and attack traffic as 1.

### Preprocessing
I applied dataset-specific preprocessing strategies. For NSL-KDD, I used a combination of categorical encoding and feature scaling within a preprocessing pipeline. For CICIDS2017, I focused on numerical feature scaling after cleaning and sampling the data to manage memory constraints.

To address class imbalance, I applied **SMOTE exclusively on the training data**, ensuring that no synthetic samples leaked into the test sets.

### Models
I evaluated three traditional machine learning models: Decision Tree, Random Forest, and Logistic Regression. Model hyperparameters were kept consistent within each experiment to enable fair comparison across datasets and evaluation settings.

### Evaluation setup
I conducted both intra-dataset and cross-dataset evaluations. In intra-dataset experiments, models were trained and tested on the same dataset. In cross-dataset experiments, models were trained on one dataset and evaluated on the other.

Because NSL-KDD and CICIDS2017 have different feature spaces, I performed feature alignment using a fixed-dimensional hashing approach to enable cross-dataset evaluation.

### Metrics and outputs
I evaluated model performance using accuracy, precision, recall, F1-score for the attack class, and ROC-AUC where applicable. All metrics were saved as JSON files, and visual outputs such as confusion matrices, ROC curves, and performance plots were saved as PNG files for further analysis.
<!-- TECHNICAL_DETAILS_END -->

<!-- INTERPRETATION_START -->

## Interpretation

I interpret the results of this project by considering both the numerical performance metrics and the visual evidence provided by the plotted figures. Rather than focusing on individual scores in isolation, I examine how model behavior changes across datasets and evaluation settings.

From the intra-dataset experiments, I observe how each model adapts to the statistical properties of a single dataset. These results reflect how well the models can learn patterns when the training and testing data come from the same distribution.

The cross-dataset experiments provide a more challenging perspective. When models trained on one dataset are evaluated on a different dataset, performance differences highlight the difficulty of generalizing intrusion detection systems across heterogeneous network environments. I interpret these differences as evidence of dataset bias and feature distribution mismatch rather than simple model failure.

Visualizations such as confusion matrices and cross-dataset transfer plots help me better understand error patterns, including false positives and false negatives. These plots complement the quantitative metrics by revealing how prediction behavior shifts under different conditions.

Overall, I interpret the results as emphasizing the importance of robust evaluation strategies for intrusion detection. While intra-dataset results demonstrate learning capability, cross-dataset results underline the practical challenges faced when deploying models in real-world scenarios.
<!-- INTERPRETATION_END -->

<!-- LIMITATIONS_START -->

## Limitations

I recognize several limitations in this study that are important for correctly contextualizing the results and their applicability.

The experiments are conducted using **benchmark intrusion detection datasets**, specifically NSL-KDD and CICIDS2017. While these datasets are well-established in academic research, they represent controlled environments and may not fully reflect the diversity and dynamics of live network traffic.

The intrusion detection task is formulated as **binary classification**, distinguishing only between normal and attack traffic. This abstraction simplifies analysis but does not capture differences between individual attack categories, which may be relevant in practical deployment scenarios.

Cross-dataset evaluation requires aligning datasets with inherently different feature spaces. Although the adopted feature alignment strategy enables comparative analysis, it may reduce the expressiveness of original features and influence cross-dataset performance.

In addition, this work focuses on a limited set of traditional machine learning models. Other approaches, such as deep learning architectures or hybrid methods, are not explored within this study.
<!-- LIMITATIONS_END -->

<!-- FUTURE_WORK_START -->

## Future Work

1. I plan to extend the current binary classification setup to a multi-class intrusion detection framework in order to differentiate between specific attack categories.
2. I intend to investigate additional feature engineering techniques to better capture temporal and statistical characteristics of network traffic.
3. I aim to explore alternative feature alignment strategies to further improve cross-dataset generalization between heterogeneous intrusion detection datasets.
4. I plan to evaluate more advanced learning approaches, including deep learning models, and compare their behavior with traditional machine learning methods.
5. I intend to study the impact of different imbalance-handling strategies beyond SMOTE on detection performance.
6. I plan to assess the robustness of the models under more realistic and dynamic traffic conditions to better approximate real-world deployment scenarios.
<!-- FUTURE_WORK_END -->

<!-- REFERENCES_START -->

## References

1. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. *Proceedings of the IEEE Symposium on Computational Intelligence for Security and Defense Applications*.

2. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *Proceedings of the International Conference on Information Systems Security and Privacy*.

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

4. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
<!-- REFERENCES_END -->
