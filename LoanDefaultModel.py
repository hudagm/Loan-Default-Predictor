import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

# Load train and test files
training = pd.read_csv('train.csv', infer_datetime_format = True)
testing = pd.read_csv('test_bqCt9Pv.csv', infer_datetime_format = True)

def preprocessing(df):
    # Filling missing values
    df['Employment.Type'] = df['Employment.Type'].fillna(df['Employment.Type'].mode()[0]) 

    # Converting to datetime
    df['Date.of.Birth'] = pd.to_datetime(df['Date.of.Birth'], infer_datetime_format = True, format = '%d-%m-%y')
    df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'], infer_datetime_format = True, format = '%d-%m-%y')
    df['Age'] = (df['DisbursalDate'] - df['Date.of.Birth']).astype('<m8[Y]').astype(int)

    # Converting 00 yrs 00 months format to months 
    df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].apply(lambda x: int(re.findall(r'\d+', x)[0])*12 + int(re.findall(r'\d+', x)[1]))
    df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].apply(lambda x: int(re.findall(r'\d+', x)[0])*12 + int(re.findall(r'\d+', x)[1]))
    return df

def feature_engineering(df):
    df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace({'No Bureau History Available' : 'No History'})

    df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace = [
        'Not Scored: Not Enough Info available on the customer',
        'Not Scored: No Activity seen on the customer (Inactive)',
        'Not Scored: Sufficient History Not Available',
        'Not Scored: No Updates available in last 36 months',
        'Not Scored: Only a Guarantor',
        'Not Scored: More than 50 active Accounts found'],
        value = 'Not Scored')

    df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace = [
        'A-Very Low Risk',
        'B-Very Low Risk',
        'C-Very Low Risk',
        'D-Very Low Risk'],
        value = 'Very Low Risk')

    df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace = [
        'E-Low Risk',
        'F-Low Risk',
        'G-Low Risk'],
        value = 'Low Risk')

    df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace = [
        'H-Medium Risk',
        'I-Medium Risk'],
        value = 'Medium Risk')

    df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace = [
        'J-High Risk',
        'K-High Risk'],
        value = 'High Risk')

    df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace = [
        'L-Very High Risk',
        'M-Very High Risk'],
        value = 'Very High Risk')


    df['TOTAL_NO_OF_ACCTS'] = df['PRI.NO.OF.ACCTS'] + df['SEC.NO.OF.ACCTS']
    df['TOTAL_ACTIVE_ACCTS'] = df['PRI.ACTIVE.ACCTS'] + df['SEC.ACTIVE.ACCTS']
    df['TOTAL_OVERDUE_ACCTS'] = df['PRI.OVERDUE.ACCTS'] + df['SEC.OVERDUE.ACCTS']
    df['TOTAL_CURRENT_BALANCE'] = df['PRI.CURRENT.BALANCE'] + df['SEC.CURRENT.BALANCE']
    df['TOTAL_SANCTIONED_AMOUNT'] = df['PRI.SANCTIONED.AMOUNT'] + df['SEC.SANCTIONED.AMOUNT']
    df['TOTAL_DISBURSED_AMOUNT'] = df['PRI.DISBURSED.AMOUNT'] + df['SEC.DISBURSED.AMOUNT']
    df['TOTAL_INSTALL_AMOUNT'] = df['PRIMARY.INSTAL.AMT'] + df['SEC.INSTAL.AMT']    

    return df

def data_labelling(df):
    cns_risk_label = {'No History' : 0, 'Not Scored' : 0, 'Very Low Risk' : 1, 'Low Risk' : 2, 
              'Medium Risk': 3, 'High Risk': 4, 'Very High Risk': 5}
    employment_label = {'Self employed' : 0, 'Salaried' : 1}

    df.loc[:,'employment_label'] = df.loc[:,'Employment.Type'].apply(lambda x: employment_label[x])
    df.loc[:,'cns_risk_label'] = (df.loc[:,'PERFORM_CNS.SCORE.DESCRIPTION'].apply(lambda x: cns_risk_label[x]))

    return df

def clean_data(df):
    df = preprocessing(df)
    df = feature_engineering(df)
    df = data_labelling(df)
    return df

train_df =  clean_data(training)
test_df =  clean_data(testing)

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,StratifiedKFold,cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,recall_score,roc_auc_score,roc_curve,auc

features = ['disbursed_amount', 'asset_cost', 'MobileNo_Avl_Flag','Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag',
'PERFORM_CNS.SCORE', 'PRI.NO.OF.ACCTS','PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT','PRI.DISBURSED.AMOUNT',
'SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT','SEC.DISBURSED.AMOUNT',
'PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT','NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','AVERAGE.ACCT.AGE',
'CREDIT.HISTORY.LENGTH','NO.OF_INQUIRIES','Age','TOTAL_NO_OF_ACCTS','TOTAL_ACTIVE_ACCTS','TOTAL_OVERDUE_ACCTS','TOTAL_CURRENT_BALANCE',
'TOTAL_SANCTIONED_AMOUNT','TOTAL_DISBURSED_AMOUNT', 'TOTAL_INSTALL_AMOUNT','employment_label','cns_risk_label']

# Dropping unwanted columns
columns_to_drop = ['UniqueID', 'ltv', 'branch_id', 'supplier_id', 'manufacturer_id', 'Current_pincode_ID',
                   'Date.of.Birth', 'Employment.Type','DisbursalDate', 'State_ID', 'Employee_code_ID', 'PERFORM_CNS.SCORE.DESCRIPTION']

train_df = training.drop(columns = columns_to_drop)
test_df = testing.drop(columns = columns_to_drop)

scaler = RobustScaler()

scaled_train = train_df.copy()
scaled_test = test_df.copy()

scaled_train[features] = scaler.fit_transform(train_df[features])
scaled_test[features] = scaler.fit_transform(test_df[features])

X = scaled_train[features]
y = scaled_train['loan_default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

def train_model(model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print('------', model, '------')
    print('Accuracy Score',accuracy_score(y_test, pred))
    print('Recall_score:',round(recall_score(y_test, pred),2))
    print('F1_score:',round(f1_score(y_test, pred),2))
    print('roc_auc_score:',round(roc_auc_score(y_test, pred), 2))
    print('Confusion_matrix')
    print(pd.DataFrame(confusion_matrix(y_test, pred)))
    print('-'*50)
    return model

#### BEFORE Sampling
lr = train_model(LogisticRegression(max_iter = 150))
rfc = train_model(RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=1))
dtc = train_model(DecisionTreeClassifier(criterion="entropy", max_depth=3))
etc = train_model(ExtraTreesClassifier())

# Oversampling using SMOTE
from imblearn.over_sampling import SMOTE

oversample = SMOTE(random_state = 2)
X, y = oversample.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#### AFTER Sampling
lr = train_model(LogisticRegression(max_iter = 100))
rfc = train_model(RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=1))
dtc = train_model(DecisionTreeClassifier(criterion="entropy", max_depth=3))
etc = train_model(ExtraTreesClassifier())

# Predicting on test data
y_pred = etc.predict(scaled_test[features])
prediction = pd.DataFrame({'predicted_loan_default':y_pred})

prediction_df = pd.DataFrame(data = testing['UniqueID'], columns = ['UniqueID'])
prediction_df['predicted_loan_default'] = prediction['predicted_loan_default']
prediction_df.to_csv('test_prediction.csv', index = False)




