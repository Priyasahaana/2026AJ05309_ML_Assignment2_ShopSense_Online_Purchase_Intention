# ShopSense: Online Purchase Intention Analytics

## a. Problem statement
Predict whether an online retail browsing session results in a purchase. `Revenue=1` represents a purchase and `Revenue=0` represents no purchase.

## b. Dataset description
- Dataset: UCI Online Shoppers Purchasing Intention (dataset ID 468)
- Instances loaded: 12,330
- Input features: 17
- Target: Revenue (binary)
- Train-test split: stratified 80%/20%
- Random state: 42
- Numerical features: 10
- Categorical features: 7

## c. GitHub Repository Link
[ShopSense GitHub Repository](https://github.com/Priyasahaana/2026AJ05309_ML_Assignment2_ShopSense_Online_Purchase_Intention)

## d. Models used and evaluation

All five classifiers were trained and evaluated using the same stratified training and test partitions.

### Comparison of evaluation metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.8873 | 0.8887 | 0.6926 | 0.4895 | 0.5736 | 0.5211 |
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Support Vector Machine | 0.8873 | 0.8824 | 0.7047 | 0.4686 | 0.5629 | 0.5149 |
| KNN | 0.8723 | 0.8285 | 0.7055 | 0.3010 | 0.4220 | 0.4049 |
| Naive Bayes | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Decision Tree | Add your own observation based on its Accuracy, AUC, Precision, Recall, F1, MCC and confusion matrix. |
| Logistic Regression | Add your own observation based on its Accuracy, AUC, Precision, Recall, F1, MCC and confusion matrix. |
| Support Vector Machine | Add your own observation based on its Accuracy, AUC, Precision, Recall, F1, MCC and confusion matrix. |
| KNN | Add your own observation based on its Accuracy, AUC, Precision, Recall, F1, MCC and confusion matrix. |
| Naive Bayes | Add your own observation based on its Accuracy, AUC, Precision, Recall, F1, MCC and confusion matrix. |

### Overall winner for the dataset

**Provisional winner based on ROC-AUC:** Decision Tree

**Overall winner based on average ranking across the six evaluation metrics:** Decision Tree

The final selection should also consider the confusion matrices and the relative importance of false positives and false negatives.

## Streamlit application
The application supports labelled CSV upload, model selection, all six evaluation metrics, a confusion matrix, and a classification report. It loads saved model pipelines and does not retrain on uploaded test data.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository contents
- `app.py` — Streamlit application
- `requirements.txt` — application dependencies
- `README.md` — project documentation
- `test_data.csv` — held-out labelled test data
- `model/` — five saved preprocessing and classifier pipelines
- `model_comparison.csv` — model evaluation results
- `metadata.json` — experiment metadata
- `assets/` — EDA and evaluation plots
- `2026AJ05309_MACHINE_LEARNING_AIMLZG565_Assignment2.ipynb` — complete notebook source code

## Live Streamlit application
Add the deployed Streamlit Community Cloud URL here after deployment.

## BITS Virtual Lab evidence
The required BITS Virtual Lab execution screenshot will be included in the final assignment PDF. A Colab execution screenshot does not replace this evidence.
