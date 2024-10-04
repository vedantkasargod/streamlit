import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# Ensure that the directory exists for saving models
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
 
# Load Titanic Dataset
def load_titanic_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    titanic_data = pd.read_csv(url)
    titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})  # Convert categorical to numeric
    titanic_data = titanic_data.dropna(subset=['Age', 'Fare'])  # Drop rows with missing values
    return titanic_data
 
# 1. Train Linear Regression Model to predict Fare
def train_linear_regression(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Sex']]
    y = titanic_data['Fare']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = LinearRegression()
    model.fit(X_train, y_train)
 
    # Save the model
    with open(os.path.join(MODEL_DIR, 'linear_regression_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
 
    print("Linear Regression model saved.")
 
# 2. Train Logistic Regression Model to predict Survival
def train_logistic_regression(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Fare', 'Sex']]
    y = titanic_data['Survived']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
 
    # Save the model
    with open(os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
 
    print("Logistic Regression model saved.")
 
# 3. Train Naive Bayes Model to predict Survival
def train_naive_bayes(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Fare', 'Sex']]
    y = titanic_data['Survived']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = GaussianNB()
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes Accuracy: {accuracy:.4f}")
 
    # Save the model
    with open(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
 
    print("Naive Bayes model saved.")
 
# 4. Train Decision Tree Model to predict Survival
def train_decision_tree(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Fare', 'Sex']]
    y = titanic_data['Survived']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
 
    # Save the model
    with open(os.path.join(MODEL_DIR, 'decision_tree_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
 
    print("Decision Tree model saved.")
 
# 5. Apriori Algorithm for Recommendation System (based on Survival, Pclass, and Sex)
def train_apriori(titanic_data):
    # Use relevant binary features (1: present, 0: not present) for Apriori
    titanic_data['Survived'] = titanic_data['Survived'].apply(lambda x: 1 if x == 1 else 0)
    titanic_data['Pclass_1'] = titanic_data['Pclass'].apply(lambda x: 1 if x == 1 else 0)
    titanic_data['Pclass_2'] = titanic_data['Pclass'].apply(lambda x: 1 if x == 2 else 0)
    titanic_data['Pclass_3'] = titanic_data['Pclass'].apply(lambda x: 1 if x == 3 else 0)
   
    data_for_apriori = titanic_data[['Survived', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex']]
   
    # Apply the Apriori algorithm
    frequent_itemsets = apriori(data_for_apriori, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
 
    print(f"Apriori Rules: \n{rules.head()}")
 
    # Save the results
    with open(os.path.join(MODEL_DIR, 'apriori_model.pkl'), 'wb') as f:
        pickle.dump(frequent_itemsets, f)
 
    print("Apriori model saved.")
 
if __name__ == "__main__":
    # Load data
    titanic_data = load_titanic_data()
 
    # Train all models
    train_linear_regression(titanic_data)
    train_logistic_regression(titanic_data)
    train_naive_bayes(titanic_data)
    train_decision_tree(titanic_data)
    train_apriori(titanic_data)