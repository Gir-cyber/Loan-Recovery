import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report
model = RandomForestClassifier(n_estimators=400,max_depth=6,class_weight="balanced",random_state=42,)


from src.exception import CustomException
from src.logger import logging

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Age buckets
        X["Age_cat"] = pd.cut(
            X["Age"],
            bins=[-1, 0, 25, 45, 65, 90, np.inf],
            labels=["Invalid", "Young", "Adult", "Middle", "Senior", "Very_Old"]
        )

        # Dependents buckets
        X["Dependents_cat"] = pd.cut(
            X["Dependents"],
            bins=[-1, 0, 2, 4, np.inf],
            labels=["None", "Small", "Medium", "Large"]
        )

        return X



def main():

    X_train = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data/raw/X_train.csv"))
    y_train = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data/raw/y_train.csv"))
    X_test = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data/raw/X_test.csv"))
    y_test = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data/raw/y_test.csv"))

    numerical_features = [
    "Age",
    "Dependents",
    "Monthly_Income",
    "Real_Estate_Loans",
    "Debt_Income_Ratio",
    "Credit_Utilization",
    "Open_Credit_Lines",
    "Past_Due30_59",
    "Past_Due60_89",
    "Past_Due90"
]

    categorical_features = [
        "Age_cat",
        "Dependents_cat"
    ]


    num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
    ])


    cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])


    preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features)
    ])


    
    pipeline = Pipeline(steps=[
    ("feature_binning", FeatureBinner()),
    ("preprocessing", preprocessor),
    ("model", model)
    ])

    # Fix y shape
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Debug columns
    print(X_train.columns.tolist())
    print(numerical_features)
    print(categorical_features)


    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    with open("metrics.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))


    with open("Pipeline.pkl", "wb") as file_obj:
        joblib.dump(pipeline, file_obj)


if __name__ == "__main__":
    main()





