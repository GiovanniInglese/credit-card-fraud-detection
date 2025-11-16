📘 Credit Card Fraud Detection

A machine learning project by Giovanni Inglese

This project is part of my personal journey into learning machine learning, data preprocessing, and model training using real-world style datasets. I wanted to explore how fraud detection models work and how machine learning can identify unusual or suspicious financial activity.
I used a credit card transaction dataset (not included here for privacy reasons) and built a simple model using Python and scikit-learn. This project helped me understand the full ML workflow — from loading data, cleaning it, training a model, and evaluating the results.

🚀 What I Learned

This project was a great opportunity to get hands-on experience with:
logistic Regression
I learned how Logistic Regression works for binary classification problems — in this case, predicting whether a transaction is fraudulent or legitimate.
Data preprocessing

I practiced tasks like:
Handling missing values
Scaling numerical features
Splitting data into training/testing sets
Understanding why imbalanced datasets are tricky for ML models

 Model evaluation
I now understand the importance of metrics like:
Accuracy
Precision
Recall
F1-score

Especially for fraud detection, recall and precision are super important because false negatives can be very costly.

 Project Workflow
1. Load & explore the data
Using pandas, I loaded the dataset and checked basic info, correlations, and class distribution.

2. Preprocess the data
I cleaned the data and prepared the features for modeling.

3. Train the model
I trained a Logistic Regression model using scikit-learn.
I also learned how parameters like max_iter and solver affect convergence.

4. Evaluate performance
I generated predictions and evaluated how well the model performs at detecting fraud.

Technologies Used

Python 

Pandas

NumPy

scikit-learn

Matplotlib (optional for plotting)

Install dependencies:
pip install -r requirements.txt

Running the Project
Add your dataset (example: creditcard.csv) to the project folder
(This repository does NOT include the original data for privacy reasons.)

Run the training script:
python train_data.py
The script will train the model and print out metrics like accuracy and recall.

 Future Improvements
As I continue learning more about machine learning, I want to:
Experiment with Random Forest and XGBoost
Try out SMOTE to handle the class imbalance better
Build a small Flask or FastAPI app where users can upload a transaction and get a prediction
Add visualizations for fraud vs. non-fraud distributions

 Goal of This Project
This isn’t meant to be a perfect production-ready fraud model.
It’s meant to be a learning experience — helping me understand the core ideas behind:
Machine learning
Data preprocessing
Training models

Evaluating results

Handling real-world problems like imbalanced data

If you're also learning ML, feel free to fork the project and play around with it.
