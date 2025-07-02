import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv('StudentsPerformance.csv')
df['average'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
df['result'] = df['average'].apply(lambda x: 'Pass' if x >= 40 else 'Fail')


X = df[['math score', 'reading score', 'writing score']]
y = df['result']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = RandomForestClassifier()
model.fit(X_train, y_train)


joblib.dump(model, 'model.pkl')  # This creates model.pkl
