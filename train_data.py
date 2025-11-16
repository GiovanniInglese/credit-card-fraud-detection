import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data = pd.read_csv("C:\\Users\\giova\\Downloads\\Credit card fraud detection\\creditcard.csv")

data=credit_card_data.head()


info = credit_card_data.tail()

null_values = credit_card_data.isnull().sum()

#dataset is very unabalnced 0 = normal trasaction, 1 = fraud transaction
#seperate data for better fraud analysis
num_choices= credit_card_data['Class'].value_counts()
#print(num_choices)

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
#print(legit.shape)
#print(fraud.shape)

#Decribes the statistical values of the amounts used in the legit trasaction such as highest and lowest amounts as well as means and percentiles
L=legit.Amount.describe()
#print(L)
# Same with fraud amounts 
r=fraud.Amount.describe()
#print(r)

#Compare the values for both transaction types

credit_card_data.groupby('Class').mean()


#we will use under-sampling, building sample from original dataset
#Containing similar distribution of normal trasactions and fraudelent transactions

legit_sample = legit.sample(n=492)

#concatenating the two 492 good transaction and the 492 bad transactions

new_dataset = pd.concat([legit_sample,fraud], axis = 0)

# Now its even 492,492
new_dataset['Class'].value_counts()

#means for the datasets from earlier and now are similar so not to much variation
new_dataset.groupby('Class').mean()
#
X = new_dataset.drop(columns='Class', axis = 1)
Y =new_dataset['Class']
#print(X)
#print(Y)

#Split data into training data and testing data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,stratify = Y, random_state=2)
#print(X.shape, X_train.shape ,X_test.shape)

#Model training
#logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,     # or 2000 if needed
    solver="lbfgs"     # this is already the default, but explicit is fine
)

#train the model

model.fit(X_train,Y_train)
#Accuracy score
#checks how accurate the model was in predicting with training data
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print("Accuracy on Training data: ",training_data_accuracy)