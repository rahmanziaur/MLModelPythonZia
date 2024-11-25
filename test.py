import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

from sklearn.preprocessing import StandardScaler

# Load the iris dataset
dataset = pd.read_csv('iris1.csv')

# Define features and target variable
X = dataset.iloc[:, :-1]  # Features (all columns except the last)
y = dataset.iloc[:, -1]  # Target variable (last column)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test,y_pred))

scaler = StandardScaler()

pickle.dump(clf,open("yourdrivename/" + "model" + ".sav", "wb"))
pickle.dump(scaler, open("yourdrivename/" + "scalermodel" +".sav","wb"))