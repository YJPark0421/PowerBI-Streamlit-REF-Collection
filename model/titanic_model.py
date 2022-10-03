import pandas as pd
import numpy as np
import pickle
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/YJPark0421/Streamlit-POWERBI/main/titanic.csv'
df = pd.read_csv(url, index_col=0).reset_index()

df = df.dropna()

y = df['Survived']
X = df.iloc[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pickle.dump(log_reg, open('titanic_clf.pkl', 'wb'))
