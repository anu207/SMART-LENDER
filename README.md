# Smart Lender — Loan Approval Prediction

A complete, working version of the Smart Lender project: dataset, EDA/model-training
notebook, and a Flask web app that predicts loan approval.

## Folder structure
```
SmartLender/
├── Dataset/
│   ├── generate_dataset.py     # generates loan_prediction.csv
│   └── loan_prediction.csv     # 614-row loan applicant dataset
├── Training/
│   └── Loan Prediction using ML.ipynb   # EDA, preprocessing, SMOTE, model training
├── Flask/
│   ├── app.py                  # Flask backend
│   ├── rdf.pkl                 # trained model (already generated & tested)
│   ├── scale1.pkl              # fitted StandardScaler (already generated & tested)
│   ├── static/css/input.css
│   └── templates/
│       ├── home.html
│       ├── input.html
│       └── output.html
└── requirements.txt
```

## About the dataset
`loan_prediction.csv` is a synthetic dataset built to match the schema of the
well-known Kaggle/Analytics Vidhya "Loan Prediction" dataset (Loan_ID, Gender,
Married, Dependents, Education, Self_Employed, ApplicantIncome,
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
Property_Area, Loan_Status), including realistic missing values. If you'd
rather use the original dataset, just replace this CSV with it — the column
names are identical, so the notebook needs no changes.

## Setup
```bash
cd SmartLender
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1 — Train the model (optional — rdf.pkl / scale1.pkl are already included)
Open `Training/Loan Prediction using ML.ipynb` in Jupyter/Anaconda and run all
cells. It will regenerate `Flask/rdf.pkl` and `Flask/scale1.pkl`.
```bash
jupyter notebook "Training/Loan Prediction using ML.ipynb"
```

## Step 2 — Run the Flask app
```bash
cd Flask
python app.py
```
Open the URL shown in the terminal (usually `http://127.0.0.1:5000`), click
**Predict**, fill in the applicant details, and submit to see the result.

## Notes
- The model, scaler, and full Flask app were tested end-to-end in this
  environment (home page → form → prediction) before delivery.
- `app.py` feeds inputs to the model in the exact same column order used
  during training: Gender, Married, Dependents, Education, Self_Employed,
  ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
  Credit_History, Property_Area.
- The notebook uses `imbalanced-learn`'s `SMOTE` for class balancing — make
  sure it's installed (`pip install imbalanced-learn`) before running it.
