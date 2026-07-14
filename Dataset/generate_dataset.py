"""
Generates a synthetic loan_prediction.csv dataset matching the classic
Analytics Vidhya / Kaggle "Loan Prediction" schema used in the Smart Lender project:

Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 614  # same size as the original public dataset

gender = rng.choice(["Male", "Female"], size=N, p=[0.81, 0.19])
married = rng.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = rng.choice(["0", "1", "2", "3+"], size=N, p=[0.58, 0.17, 0.17, 0.08])
education = rng.choice(["Graduate", "Not Graduate"], size=N, p=[0.78, 0.22])
self_employed = rng.choice(["Yes", "No"], size=N, p=[0.14, 0.86])
property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.38, 0.38, 0.24])
credit_history = rng.choice([1.0, 0.0], size=N, p=[0.84, 0.16])

applicant_income = rng.lognormal(mean=8.4, sigma=0.55, size=N).round().astype(int)
applicant_income = np.clip(applicant_income, 150, 81000)

has_coapplicant = rng.random(N) > 0.4
coapplicant_income = np.where(
    has_coapplicant,
    rng.lognormal(mean=7.6, sigma=0.7, size=N).round().astype(int),
    0,
)
coapplicant_income = np.clip(coapplicant_income, 0, 42000)

base_loan = (applicant_income + coapplicant_income * 0.5) / 25
loan_amount = (base_loan * rng.normal(1.0, 0.25, size=N)).round().astype(int)
loan_amount = np.clip(loan_amount, 9, 700)

loan_term = rng.choice([360, 180, 120, 300, 84, 60, 36, 12],
                       size=N, p=[0.83, 0.06, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01])

# ---- Target variable: Loan_Status, driven mainly by Credit_History + income/loan ratio ----
score = (
    2.2 * credit_history
    + 0.9 * (education == "Graduate")
    + 0.00004 * (applicant_income + coapplicant_income)
    - 0.006 * loan_amount
    + 0.5 * (property_area == "Semiurban")
    + rng.normal(0, 0.8, size=N)
)
prob_approved = 1 / (1 + np.exp(-(score - 1.2)))
loan_status = np.where(rng.random(N) < prob_approved, "Y", "N")

df = pd.DataFrame({
    "Loan_ID": [f"LP{str(1000 + i).zfill(4)[-4:]}" for i in range(2, N + 2)],
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})
df["Loan_ID"] = "LP00" + (df.index + 1002).astype(str)

# Inject realistic missing values (mirrors the original dataset's null pattern)
def inject_nulls(col, frac):
    idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

inject_nulls("Gender", 0.02)
inject_nulls("Married", 0.005)
inject_nulls("Dependents", 0.025)
inject_nulls("Self_Employed", 0.05)
inject_nulls("LoanAmount", 0.035)
inject_nulls("Loan_Amount_Term", 0.023)
inject_nulls("Credit_History", 0.08)

df.to_csv("loan_prediction.csv", index=False)
print(df.shape)
print(df.head())
print(df.isnull().sum())
