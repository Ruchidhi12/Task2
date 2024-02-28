# Task2
Build a Predictive Model in Python to determine the likelihood of survival of Passengers on the Titanic using Data Science Techniques.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
titanic_data = pd.read_csv('titanic.csv')
# Clean the data
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())
titani# Convert categorical variables to numerical variables
c_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'])
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)
new_passenger = pd.DataFrame({
    'Pclass': 3,
    'Sex': 0,
    'Age': 20   
    SibSp': 1,
        'Parch': 0,
    'Fare': 7.2500,
    'Embarked_C': 0,
    'Embarked    'Embarked_S': 1
_Q': 0,
})
prediction = model.predict(new_passenger)
if prediction == 1:
    print('The passenger is likely to survive.')
else:
    print('The passenger is not likely to survive.')



