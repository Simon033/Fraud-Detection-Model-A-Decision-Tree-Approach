# Fraud-Detection-Model-using-Logistic-Regression-for-Transaction-Data
A robust fraud detection model using Logistic Regression to analyze transaction data for a financial company with 99.94% accuracy and 98% recall.

## Data Dictionary & Source Acquisition
- Data Dictionary: The data dictionary of the dataset can be found [here](https://drive.google.com/uc?id=1VQ-HAm0oHbv0GmDKP2iqqFNc5aI91OLn&export=download).
- Data Source: The dataset can be found [here](https://drive.google.com/uc?export=download&confirm=6gh6&id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV).

### Tools Used
For this project everything was done using Python on Jupyter Notebook. Libraries such as pandas, scikit-learn, matplotlib, seaborn were used.

### Dataset
The dataset used to train the model contains historical transaction data from a financial institution. Each sample in the dataset represents a single transaction consisting of step, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, newbalanceDest, oldbalanceDest, isFraud, isFlaggedFraud

### Data Analysing and Processing
After carefully curating and preprocessing the dataset, comprehensive exploratory data analysis (EDA) was done to gain essential insights into the underlying patterns and relationships. During the EDA phase, thorough examination of data distributions, missing values and outliers was done. Assessed feature correlations to ensure the dataset was primed for modeling. Additionally, feature engineering techniques, such as label encoding was done to transform categorical variables into a numerical format suitable for Logistic Regression.

### Model Training
The widely-used Logistic Regression algorithm is used due to its ability to handle binary classification problems like fraud detection and the varying nature of the values in dataset. 
(In the end to verify the performance was measured with other model and Logistic Regression still had the best overall result)

### Model Evaluation
To evaluate the performance of the fraud detection model, the dataset is split into training and testing sets. Metrics such as accuracy, confusion matrix, precision, recall, F1-score(classification_report) are computed on the test set to assess the model's effectiveness in detecting fraudulent transactions.

The model achieved an impressive overall accuracy score of ***99.94%.*** It demonstrates high precision & recall in detecting non-fraudulent transactions, indicating its ability to correctly identify genuine transactions. On the other hand, the model shows a precision(0.71) but a significantly higher recall (0.98) for detecting fraudulent transactions. The high recall for fraud detection suggests that the model can correctly detect nearly ***98%*** of all actual fraud cases. This is a crucial aspect for a fraud detection system as missing fraudulent transactions can have severe consequences thus higher recall was given preferrence. Additionally, the low precision for fraud detection(0.71) means a higher false positive prediction. However, the actual false positives represent just ***0.05%*** of the total non-fraudulent cases, making the impact of false positives relatively insignificant.

In summary, the model's prioritization of recall over precision for fraud detection indicates a robust ability to identify a vast majority of fraudulent transactions while maintaining a very high accuracy in classifying non-fraudulent transactions. The trade-off between precision and recall aligns well with the fraud detection objective, making the model highly effective in catching potential fraud cases. The model's performance indicates a strong foundation for its practical application in real-world scenarios.

Here's the table and data showing the same.

**Accuracy score:** 0.9994766306961598

**Confussion Matrix**
          0 [ 1270271   633 ]
(Actual)  1 [   33     1587 ]
                0        1
                (Predicted)

**Classification Report:**
              **precision    recall  f1-score   support**

           *0       1.00      1.00      1.00   1270904*
           *1       0.71      0.98      0.83      1620*

    accuracy                           1.00   1272524
   macro avg       0.86      0.99      0.91   1272524
weighted avg       1.00      1.00      1.00   1272524

### Interpretting the Model
The models feature coefficients provide insights into the impact of each feature on the probability of fraud. 
The *'oldbalanceOrg'* column has a high positive coefficient of 8.85e-04 indicating that accounts with higher balance are often targeted probably because they present a more lucrative opportunity to maximize their gains. It also gives a chance to not be noticed, as small change in big number could be easily missed by the victim.
The *'amount'* column has the highest negative coefficient of -8.85e-04 indicating that most fraudsters tend to keep the fraudulent transaction amounts moderate rather than attempting extremely large amount fraudulent transactions. There could be several reasons for this behavior such as 'Avoiding Detection', 'Bypassing Authorization Limits',etc.
The *'newbalance'* column has the highest coeffecient of -9.643233e-04, this is a factor that is directly connected to and affected by oldbalance and amount. It is also possible to notice discrepancies in newbalance compared to what it should be.

### Limitations
While Logistic Regression is interpretable and performs really well in predicting binary output, it might have limitations in capturing complex relationships between features. In cases where interactions between features are non-linear eg('type','nameDest'), more sophisticated models like Random Forests, Gradient Boosting Machines, or Neural Networks may be more suitable.

### Future Improvements
To enhance the model's performance, further exploration of other machine learning algorithms and ensemble methods can be considered.(I couldn't because of GPU processing power limitations on my end)

### Conclusion
This Logistical Regression Model is capable of categorising transaction with an accuracy of ***99.94%*** and predicting ***98%*** of the fraud transactions. It mainly focuses on 'oldbalanceOrg', 'newbalanceOrig' and 'amount' to draw the prediction. This model can help financial institutions take premptive measures by predicting and stopping frauds before they happen.


A model is sucessfully created that predicts fraudulent transaction in a financial organisation with high accuracy.
— Simon Nadar

