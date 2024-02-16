# Fraud Detection Model: A Decision Tree Approach
A robust fraud detection model to analyze financial transaction data for fraud, with an overall accuracy of 92.2% and a recall of 98% in detecting fraud transaction. 
The model was trained on 6.3 Million records where 70% of the data was used for training and remaining 30% for testing.

### Introduction
The purpose of this fraud detection model is to distinguish between legitimate and fraulent transaction by looking for patterns indicative of fraud.

## Data Dictionary & Source Acquisition
- Data Dictionary: The data dictionary of the dataset can be found [here](https://drive.google.com/uc?id=1VQ-HAm0oHbv0GmDKP2iqqFNc5aI91OLn&export=download).
- Data Source: The dataset can be found [here](https://drive.google.com/uc?export=download&confirm=6gh6&id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV).

### Tools Used
For this project everything was done using Python on Jupyter Notebook. Libraries such as pandas, scikit-learn, statsmodel, matplotlib, seaborn, numpy were used.

### Dataset
The dataset used to train the model contains historical transaction data from a financial institution. Each sample in the dataset represents a single transaction consisting of step, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, newbalanceDest, oldbalanceDest, isFraud, isFlaggedFraud.
The data originally has 6.3 million records.

### Data Processing and EDA 
Data was cleaned for missing values and outliers. Multicolinearity was dealt with by dropping and converting certain columns. Categorical columns were encoded with One Hot Encoding. Newer more features wre created from old features.
Comprehensive exploratory data analysis (EDA) was done to alongside to gain essential insights into the underlying patterns, relationships and to determine relevant features. Assessed feature correlations to ensure the dataset was primed for modeling.
The severe Class Imbalance was addressed using SMOTE and the data was furthered scaled using Robust Scaler to prevent overfitting. This also added records resulting in total **12.7** Million records.

### Feature Selection
Based on EDA done with univariate, multivaiate analysis, correlation matrix, etc relevant features were selected for model building.

### Model Selection
Multiple models were trained and the best performing model (Decision Tree Classifier) was selected since it had an accuracy, f1 and recall score of **99.9%**. Further **HyperParmeter Tuning** was done to prevent overfitting and find best parameters. Furthermore Decision Tree also allows for good interpretablity of the model and the underlying problem.

### Model Training
Hyper Parameter Tuning was done with a 70:30 data split, where only 70%(8.8M records) were used for model training. K-Fold method was used to address data imbalance and avoid overfitting.

### Model Evaluation
The remaining 30% data(3.8M records) which was not used during the training process was used for Model Evaluation. The model was evaluated using **MCC accuracy score** as the scoring parameter. Metrics such as accuracy, confusion matrix, precision, recall, F1-score(classification_report) are computed on the test set to assess the model's effectiveness in detecting fraudulent transactions. The model yeilded an ***overall accuracy of 92.2% and 98% recall on detecting fraud***.

Here's the table and data showing the same.

Accuracy:  0.9226100569385514<br>
Classification Report:<br>

               precision    recall  f1-score   support

           0       0.98      0.94      0.96   1906585
           1       0.94      0.98      0.96   1906060

    accuracy                           0.96   3812645
    macro avg      0.96      0.96      0.96   3812645  
    weighted avg   0.96      0.96      0.96   3812645

**Confussion Matrix**

               0 [ 1791267   115318 ]
     (Actual)  1 [ 33821     1872239 ]
                      0        1
                    (Predicted)


### Interpretting the Model 
The models feature importance coefficients provide insights into the impact of each feature on the probability of fraud. 
1- Old Balance of the Origin account seems to be the most important factor, wherein account with very high balance are very highly susceptible to fraud.<br>
2- The type of transaction is also a key feature in determining if a transaction could be fraud, CASH_IN and TRANSFER are two types which are really useful in determinging fraud.<br>
3 Transaction amount is also a key feature, wherein transactions of really big amount are never fraud and the fraud tries to hide in the medium-high range.<br>
4- The Time to last transaction of the Destination Account and the type of destination account is also a key feature that helps determine fraud.<br>

           Feature  Coefficient
             step     0.032061
           amount     0.280566
    oldbalanceOrg     0.407410
         nameDest     0.064133
    oldbalanceDest    0.007755
     TimeDeltaDest    0.005672
          CASH_IN     0.173055
         CASH_OUT     0.012845
          PAYMENT     0.012921
        TRANSFER      0.003582


### Limitations & Future Improvements
- Computational Limits prevented me from exploring other more effective techniques like GNNs, Boosted Algorithms, etc.
- To enhance the model's performance, further exploration of other machine learning algorithms and ensemble methods can be considered.(I couldn't because of GPU processing power limitations on my end)


### Conclusion 
This Model is capable of categorising transaction with an accuracy of 92.2% and predicts 98% of the fraud transactions accurately. It mainly focuses on 'oldbalanceOrg', 'type' and 'amount' and 'step' to draw the prediction. This model can help financial institutions take premptive measures by predicting and stopping frauds before they happen.

A model is sucessfully created that predicts fraudulent transaction in a financial organisation with high accuracy.
— Simon Nadar

