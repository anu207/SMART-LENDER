# SMART-LENDER
AI-powered loan approval prediction system built with Flask and scikit-learn — predicts loan approval status from applicant financial and demographic data using an XGBoost/Random Forest pipeline.  Topics: machine-learning flask loan-prediction python scikit-learn xgboost smote data-science classification fintech


# 🏦 Smart Lender — AI-Powered Loan Approval Prediction

Smart Lender is a machine learning web application that predicts whether a loan
application is likely to be **Approved** or **Rejected**, based on applicant
income, credit history, employment status, education, marital status, and
other financial indicators. It automates a process that's traditionally
manual and time-consuming for banks and financial institutions, helping
credit officers make faster, more consistent, and less biased lending
decisions.

---

## 🚀 Overview

Traditional loan approval relies on manual document verification and
subjective judgment. Smart Lender replaces that with a trained classification
model served through a lightweight Flask web app — applicants (or credit
officers) submit details through a simple form and get an instant prediction.

## ✨ Features

- End-to-end ML pipeline: data collection → EDA → preprocessing → model
  training → deployment
- Handles missing data, categorical encoding, and class imbalance (SMOTE)
- Compares four classification algorithms and automatically selects the best
- Simple, responsive Flask front end for entering applicant details and
  viewing results
- Deployable to any cloud platform (IBM Cloud, Render, Railway, etc.)

## 🏗️ Architecture

```
User (Applicant / Credit Officer)
        │
        ▼
 Web Browser Interface (HTML templates)
        │
        ▼
 Flask Application Layer (routing, form handling, input validation)
        │
        ▼
 Prediction Engine (loads rdf.pkl → feature vector → predict)
        │
        ▼
 Result rendered back to the browser
```

The model itself is produced by a separate ML pipeline: Dataset Collection →
Data Preprocessing (missing values, encoding, SMOTE, scaling) → Train/Test
Split → Model Training (Decision Tree, Random Forest, KNN, Gradient
Boosting) → Evaluation → Best Model Selection → Pickle Serialization.

## 🗂️ Project Structure

```
SmartLender/
├── Dataset/
│   └── loan_prediction.csv
├── Training/
│   └── Loan Prediction using ML.ipynb
├── Flask/
│   ├── app.py
│   ├── rdf.pkl
│   ├── scale1.pkl
│   ├── static/css/
│   └── templates/
│       ├── home.html
│       ├── input.html
│       └── output.html
├── requirements.txt
└── README.md
```

## 🧠 Machine Learning Pipeline

| Stage | Details |
|---|---|
| **Data Collection** | Loan applicant dataset (Kaggle / Analytics Vidhya schema) |
| **EDA** | Univariate, bivariate, and multivariate analysis (distplot, countplot, swarmplot) |
| **Preprocessing** | Missing value imputation (mean/mode), categorical encoding |
| **Class Balancing** | SMOTE (Synthetic Minority Over-sampling Technique) |
| **Feature Scaling** | StandardScaler |
| **Models Trained** | Decision Tree, Random Forest, KNN, Gradient Boosting (XGBoost) |
| **Evaluation** | Accuracy, Confusion Matrix, Classification Report, 5-fold Cross-Validation |
| **Model Storage** | Best model serialized with `pickle` |

## 🛠️ Tech Stack

- **Language:** Python
- **ML/Data:** NumPy, Pandas, Scikit-learn, imbalanced-learn, Matplotlib, Seaborn
- **Web Framework:** Flask
- **Model Serialization:** Pickle
- **IDE/Tools:** Jupyter Notebook, Anaconda, PyCharm

## ⚙️ Installation & Setup

```bash
git clone https://github.com/<your-username>/SmartLender.git
cd SmartLender
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Running the App

```bash
cd Flask
python app.py
```
Open the URL printed in the terminal (typically `http://127.0.0.1:5000`),
click **Predict**, fill in the applicant details, and submit to view the
prediction.

## 📓 Retraining the Model

Open and run all cells in `Training/Loan Prediction using ML.ipynb` to
regenerate `rdf.pkl` and `scale1.pkl` from the dataset.

## 📊 Entity Relationship Overview

- **User** (Credit Officer / Applicant) → submits multiple **Loan Applications**
- **Applicant Profile** → linked to a Loan Application
- **Credit History** → one record per applicant, used in prediction
- **Loan Application** → produces one **Prediction Result**
- **Model** → generates multiple Prediction Results

## 🔮 Future Enhancements

- Integration of additional financial indicators
- Real-time credit score validation
- Advanced ensemble techniques for improved accuracy
- Cloud deployment with CI/CD

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
