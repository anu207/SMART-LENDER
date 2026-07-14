## SMART LENDER :
Smart Lender is a simple web app that predicts whether a loan should be approved or rejected. It looks at details like income, credit history, education, and marital status, and gives an instant answer using a machine learning model.

Banks usually check loan applications by hand, which takes time and can be inconsistent. This project makes that process faster and fairer by letting a trained model do the first check.

## How it works

A user fills in a short form with their income, loan amount, credit history, and a few other details. Flask (a Python web framework) takes this information, passes it to a saved machine learning model, and shows whether the loan is likely to be approved or rejected.

Behind the scenes, the model was built by cleaning up real loan data, filling in missing values, balancing the dataset so it isn't biased toward one outcome, and then training a few different models to see which one works best.

## What's inside

The project has three main parts. The **Dataset** folder holds the loan data used for training. The **Training** folder has a notebook that explores the data, cleans it, and trains the model. The **Flask** folder has the actual web app people use, along with the saved model.

## Tools used

The project is built in Python using Pandas and NumPy for handling data, Scikit-learn for building the models, Seaborn and Matplotlib for charts, and Flask for the website itself.

## Running it yourself

Clone the project, install the requirements with pip, then go into the Flask folder and run `python app.py`. Open the link it shows in your terminal, click Predict, fill in the form, and you'll see the result right away.

## What's next

Down the line, this could be improved by adding more financial details, checking credit scores in real time, and trying stronger models to make the predictions even more accurate.

