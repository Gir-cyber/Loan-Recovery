import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    
    try:
        logging.info("Loading data from csv")
        df = pd.read_csv(r"C:\Users\giris\Downloads\cs-training.csv\cs-training.csv")
        logging.info("Read the csv")
        return df
    except Exception as e:
        raise CustomException(e, sys)

def preprocess_data(df:pd.DataFrame):

    try:
        logging.info("Preprocessing data")
        df = df.drop('Unnamed: 0', axis=1)
        df = df.rename(columns={
                "SeriousDlqin2yrs": "Default_In_2yrs",
                "RevolvingUtilizationOfUnsecuredLines": "Credit_Utilization",
                "age": "Age",
                "NumberOfTime30-59DaysPastDueNotWorse": "Past_Due30_59",
                "DebtRatio": "Debt_IncomeRatio",
                "MonthlyIncome": "Monthly_Income",
                "NumberOfOpenCreditLinesAndLoans": "Open_Credit_Lines",
                "NumberOfTimes90DaysLate": "Past_Due90",
                "NumberRealEstateLoansOrLines": "Real_Estate_Loans",
                "NumberOfTime60-89DaysPastDueNotWorse": "Past_Due60_89",
                "NumberOfDependents": "Dependents"
            })
        return df
    except Exception as e:
        raise CustomException(e. sys)
    

def save_data(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, data_path:str):

    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        X_train.to_csv(os.path.join(raw_data_path, "X_train"), index=False)
        X_test.to_csv(os.path.join(raw_data_path, "X_test"), index=False)
        y_train.to_csv(os.path.join(raw_data_path, "y_train"), index=False)
        y_test.to_csv(os.path.join(raw_data_path, "y_test"), index=False)

        logging.info("train and test data saved")
    except Exception as e:
        raise CustomException(e,sys)

def main():

    try:
        df = load_data()
        df = preprocess_data(df)

        X = df.drop(columns=['Default_In_2yrs'])
        y = df['Default_In_2yrs']

        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        save_data(X_train, y_train, X_test, y_test, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data"))

    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    main()
