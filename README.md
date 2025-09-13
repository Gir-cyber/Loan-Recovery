# 📌 Smart Loan Recovery System

## 🔹 Problem Statement  
Banks and financial institutions face major losses when borrowers default on loans.  
This project builds a **machine learning system to predict loan defaults** so that recovery teams can **prioritize high-risk customers** and improve repayment rates.

---

## 🔹 Dataset  
- Source: [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit)  
- Size: 10,000+ loan records  
- Features:  
  - `RevolvingUtilizationOfUnsecuredLines` → Credit card utilization ratio  
  - `age` → Age of borrower  
  - `NumberOfTime30-59DaysPastDueNotWorse` → Past due counts  
  - `DebtRatio`, `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`  
  - `NumberOfTimes90DaysLate`, `NumberRealEstateLoansOrLines`  
  - `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfDependents`  
- Target:  
  - `DefaultIn2yrs = 1` → Borrower defaulted  
  - `DefaultIn2yrs = 0` → Loan repaid  

---

## 🔹 Approach  
1. **Data Preprocessing**  
   - Handled missing values (e.g., MonthlyIncome, Dependents).  
   - Balanced dataset (equal defaults & non-defaults).  
   - Feature scaling where needed.  

2. **Modeling**  
   - Baseline: Logistic Regression (Accuracy ~0.73).  
   - Tree Ensembles: Random Forest, Gradient Boosting.  
   - Advanced: **XGBoost** (best performance).  

3. **Hyperparameter Tuning**  
   - GridSearchCV for `n_estimators`, `max_depth`, `learning_rate`, `subsample`.  

4. **Threshold Tuning**  
   - Default threshold (0.5) → Recall for defaults = ~75%.  
   - Adjusted threshold (0.20) → Recall for defaults = **94%**.  

