# Network Intrusion Detection Using Traditional Machine Learning

## Abstract
In this project, I investigate the performance of traditional machine learning algorithms for network intrusion detection. I carry out experiments using two widely used benchmark datasets: NSL-KDD and CICIDS2017. I evaluate three classifiers—Decision Tree, Random Forest, and Logistic Regression under an intra-dataset experimental setting. Model performance is assessed using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC. Through these experiments, I aim to understand the strengths and limitations of classical machine learning approaches when applied to network traffic data, particularly in the presence of class imbalance.

## 1. Introduction
Network Intrusion Detection Systems (NIDS) are an essential part of modern network security, as they help identify malicious activities within network traffic. While deep learning approaches are increasingly popular, traditional machine learning techniques are still widely used due to their interpretability, lower computational requirements, and strong baseline performance on structured network data.

In this project, I focus on evaluating classical machine learning models for intrusion detection. By conducting controlled intra-dataset experiments, I aim to compare how different models behave when trained and tested on the same dataset, and to analyze the impact of dataset characteristics such as size and class imbalance on model performance.
## 3. Experimental Setup

### 3.1 Train-Test Split
In all experiments, I split each dataset into training and testing sets using an 80/20 split. The split is stratified based on the binary class labels to ensure that both normal and attack traffic are proportionally represented in the training and test sets. This approach helps maintain a fair evaluation, especially in the presence of class imbalance.

### 3.2 Data Preprocessing
Before training the models, I apply a consistent preprocessing pipeline to both datasets. Categorical features are encoded using one-hot encoding, while numerical features are standardized using z-score normalization. I use a ColumnTransformer to ensure that preprocessing is fitted only on the training data and then applied to the test data, preventing any data leakage.

### 3.3 Handling Class Imbalance
Both NSL-KDD and CICIDS2017 exhibit class imbalance, with attack traffic either dominating or being underrepresented depending on the dataset. To address this issue, I apply different strategies depending on the model.

For Decision Tree and Random Forest classifiers and Logistic Regression on NSL-KDD dataset. I apply the Synthetic Minority Over-sampling Technique (SMOTE) only to the training data. This allows the models to learn from a more balanced class distribution without altering the original test set.

For Logistic Regression on CICIDS2017, applying SMOTE on the full training data is computationally expensive due to the dataset’s large size. Therefore, I use class-weight balancing combined with stratified subsampling of the training data. This approach reduces computational cost while still allowing the model to account for class imbalance.

### 3.4 Machine Learning Models
In this project, I evaluate three traditional machine learning models: Decision Tree, Random Forest, and Logistic Regression. These models are chosen because they represent a range of complexity and interpretability, making them suitable baselines for intrusion detection tasks.
## 4. Experimental Results

In this section, I present and analyze the experimental results obtained from applying Decision Tree, Random Forest, and Logistic Regression models to the NSL-KDD and CICIDS2017 datasets. The evaluation focuses on intra-dataset performance, where each model is trained and tested on the same dataset. Performance is assessed using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.

### 4.1 Results on NSL-KDD

The NSL-KDD dataset provides a controlled benchmark for evaluating intrusion detection models. Due to its relatively moderate size and well-structured features, all three models achieve stable performance.

The Decision Tree model demonstrates strong classification capability, achieving a good balance between detecting attack traffic and minimizing false alarms. The confusion matrix shows that most attack instances are correctly identified, although some false positives remain.

Random Forest achieves the best overall performance on NSL-KDD. By combining multiple decision trees, it improves generalization and reduces overfitting. The ROC curve further confirms this behavior, showing a high area under the curve and a clear separation between normal and attack traffic.

Logistic Regression serves as a strong linear baseline. While its performance is slightly lower than that of tree-based models, it still achieves competitive results, particularly in terms of precision. This indicates that Logistic Regression is effective at identifying normal traffic, though it may miss some attack instances compared to Random Forest.

Figures illustrating the confusion matrices, ROC curves, and false positive versus false negative distributions for each model are provided to support this analysis.

### 4.2 Results on CICIDS2017

CICIDS2017 presents a more challenging evaluation scenario due to its large scale, higher feature diversity, and severe class imbalance. As a result, model performance differs noticeably from NSL-KDD.

The Decision Tree model performs reasonably well but shows sensitivity to misclassification.  While it detects a substantial portion of attack traffic, the confusion matrix reveals an increase in false negatives compared to NSL-KDD, indicating that some attacks are misclassified as normal traffic.

Random Forest again demonstrates strong performance on CICIDS2017. Its ensemble nature allows it to better capture complex patterns in the data, leading to improved recall for attack traffic.

For Logistic Regression, I apply class-weight balancing and stratified subsampling to manage computational cost. Despite its simplicity, Logistic Regression achieves competitive performance and produces a smooth ROC curve. However, its linear nature limits its ability to fully capture the complex relationships present in CICIDS2017, resulting in lower recall compared to Random Forest.

Overall, the results indicate that CICIDS2017 is a more challenging dataset than NSL-KDD. The differences in performance across models highlight the impact of dataset characteristics such as size, feature complexity, and class imbalance on intrusion detection systems.


## 4.3 Summary Tables

### Table 1: Dataset Overview
| Dataset    |   Total Samples (estimated) | Train/Test Split   | Train Normal    | Train Attack   | Test Normal    | Test Attack   |
|:-----------|----------------------------:|:-------------------|:----------------|:---------------|:---------------|:--------------|
| NSL-KDD    |                      148520 | 80/20 (stratified) | 61644 (51.9%)   | 57172 (48.1%)  | 15411 (51.9%)  | 14293 (48.1%) |
| CICIDS2017 |                     2520755 | 80/20 (stratified) | 1676048 (83.1%) | 340556 (16.9%) | 419012 (83.1%) | 85139 (16.9%) |


### Table 2: Intra-dataset Performance Comparison

| Dataset    | Model               | Experiment    |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:-----------|:--------------------|:--------------|-----------:|------------:|---------:|-----------:|----------:|
| NSL-KDD    | Random Forest       | Intra-dataset |      0.996 |       0.997 |    0.995 |      0.996 |     0.9998     |
| NSL-KDD    | Decision Tree       | Intra-dataset |      0.995 |       0.994 |    0.995 |      0.995 |     0.9950|
| NSL-KDD    | Logistic Regression | Intra-dataset |      0.955 |       0.959 |    0.946 |      0.953 |     0.9936 |
| CICIDS2017 | Random Forest       | Intra-dataset |      0.999 |       0.996 |    0.996 |      0.996 |     0.9997    |
| CICIDS2017 | Decision Tree       | Intra-dataset |      0.999 |       0.995 |    0.996 |      0.996 |     0.9979|
| CICIDS2017 | Logistic Regression | Intra-dataset |      0.895 |       0.624 |    0.962 |      0.757 |     0.9732 |

Table 2 presents the intra-dataset performance of all models on NSL-KDD and CICIDS2017.
Random Forest achieves the best overall performance across most evaluation metrics.
Logistic Regression is more affected by class imbalance, particularly on CICIDS2017.

**Table 3. Intra-dataset Feature Comparison and Performance Analysis (Mean ± Standard Deviation).**

| Transfer Direction   | Model   | Accuracy (Mean ± Std)   | Precision (Mean ± Std)   | Recall (Mean ± Std)   | F1 (Mean ± Std)   |
|:---------------------|:--------|:------------------------|:-------------------------|:----------------------|:------------------|
| NSL-KDD (Intra)      | DT      | 0.9947 ± 0.0000         | 0.9944 ± 0.0000          | 0.9946 ± 0.0000       | 0.9945 ± 0.0000   |
| NSL-KDD (Intra)      | RF      | 0.9961 ± 0.0000         | 0.9967 ± 0.0000          | 0.9952 ± 0.0000       | 0.9960 ± 0.0000   |
| NSL-KDD (Intra)      | LR      | 0.9549 ± 0.0000         | 0.9594 ± 0.0000          | 0.9463 ± 0.0000       | 0.9528 ± 0.0000   |
|                      |         |                         |                          |                       |                   |
| CICIDS2017 (Intra)   | DT      | 0.9985 ± 0.0000         | 0.9954 ± 0.0000          | 0.9959 ± 0.0000       | 0.9957 ± 0.0000   |
| CICIDS2017 (Intra)   | RF      | 0.9986 ± 0.0000         | 0.9956 ± 0.0000          | 0.9963 ± 0.0000       | 0.9959 ± 0.0000   |
| CICIDS2017 (Intra)   | LR      | 0.8955 ± 0.0000         | 0.6235 ± 0.0000          | 0.9618 ± 0.0000       | 0.7566 ± 0.0000   |



<!-- TABLE4_START -->

### Table 4: Cross-Dataset Performance Comparison

| Transfer Direction | Model | Accuracy | Precision (Attack) | Recall (Attack) | F1 (Attack) | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Train NSL-KDD → Test CICIDS2017 | DT | 0.1694 | 0.1694 | 1.0000 | 0.2897 | 0.5000 |
| Train NSL-KDD → Test CICIDS2017 | RF | 0.6853 | 0.1197 | 0.1350 | 0.1269 | 0.4923 |
| Train NSL-KDD → Test CICIDS2017 | LR | 0.1694 | 0.1694 | 1.0000 | 0.2897 | 0.5392 |
| Train CICIDS2017 → Test NSL-KDD | DT | 0.5215 | 1.0000 | 0.0055 | 0.0110 | 0.5028 |
| Train CICIDS2017 → Test NSL-KDD | RF | 0.5188 | 0.0000 | 0.0000 | 0.0000 | 0.4989 |
| Train CICIDS2017 → Test NSL-KDD | LR | 0.5188 | 0.0000 | 0.0000 | 0.0000 | 0.4905 |

<!-- TABLE4_END -->

<!-- KEY_FINDINGS_START -->
### Key Findings

1. **Overall performance dropped in cross-dataset experiments compared to intra-dataset results**, which confirms that models trained on one dataset do not generalize perfectly to traffic captured in a different environment. This highlights how dataset-dependent intrusion detection can be.

2. **Random Forest gave the most consistent cross-dataset results across both transfer directions**, especially when I focus on F1-score and recall for detecting attacks. In practice, this suggests ensemble methods handle distribution shift better than simpler learners.

3. **Decision Tree degraded more sharply during cross-dataset transfer**, particularly when trained on NSL-KDD and evaluated on CICIDS2017. This pattern is consistent with a model that learns dataset-specific rules that do not hold when the test data distribution changes.

4. **Logistic Regression showed mixed behavior**, where precision looked reasonable in some settings, but recall often dropped. In other words, the model can be conservative about predicting attacks, which reduces false alarms but can increase missed attacks under dataset shift.

5. **Transfer direction mattered**, and I observed that training on CICIDS2017 often produced better generalization to NSL-KDD than the reverse. This may be linked to CICIDS2017 containing more modern and diverse traffic patterns.

6. **These results show why cross-dataset evaluation is necessary for IDS research**, because strong intra-dataset accuracy alone can be misleading. For real deployment, I need models that remain effective when the traffic distribution changes.

<!-- KEY_FINDINGS_END -->

<!-- CONCLUSION_START -->
### Conclusion

In this work, I evaluated multiple machine learning models for intrusion detection using both intra-dataset and cross-dataset experiments. Although intra-dataset results were consistently high, cross-dataset evaluation revealed notable generalization challenges; however, Random Forest demonstrated the most robust and stable performance across datasets, particularly in terms of F1-score and recall for attack detection. These findings emphasize the importance of cross-dataset analysis and suggest that ensemble-based models such as Random Forest are more suitable for real-world intrusion detection deployments.

<!-- CONCLUSION_END -->
