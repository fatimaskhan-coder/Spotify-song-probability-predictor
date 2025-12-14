# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns



# Loading dataset
df = pd.read_csv(r"C:\Users\Hp\OneDrive\Documents\spotify_2023.csv", encoding='latin1')
df.head()
df.columns
df.describe()
df.isnull().sum()
df.info()
#separating numerical and categorical columns 
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

#Handling Missing values 
#Mode imputation for categorical values
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

#Mean imputation for numerical values 
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
# Making sure 'streams' is numeric
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')



df['Popular'] = np.where(df['streams'] >= 100_000_000, 1, 0)


print(df[['streams','Popular']].head())


print(df.isnull().sum())




#selecting features
numeric_features = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'artist_count']
categorical_features = ['key']

#one hot encoding
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({'key': ['C', 'D', 'E', 'C', 'D']})

# One-hot encoding
encoded = pd.get_dummies(df['key'], drop_first=True) 
# Combine with original data if needed
df_encoded = pd.concat([df, encoded], axis=1)

print(df_encoded)



#scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)




#logistic reg
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)




#Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)




X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)




#applying random forest classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


#making pred
y_pred = model.predict(X_test)



#Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)






print("Model Comparison ")
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))




import joblib
joblib.dump(model, "spotify_rf_model.pkl")
joblib.dump(scaler, "spotify_scaler.pkl")
joblib.dump(encoded_cat.columns, "spotify_encoded_columns.pkl") 