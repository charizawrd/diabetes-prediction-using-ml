# Import necessary libraries
import pandas as pd
import numpy as np
# Importing machine learing library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Split data into features and target
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# List of models to test
models = {
    "Random Forest": RandomForestClassifier(n_estimators=726, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
    "SVM": SVC(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Neural Network": MLPClassifier(max_iter=300, random_state=42)
}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"{model_name} Accuracy: {round(accuracy * 100, 2)} %")

# Find and display the best performing model
best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} with accuracy: {round(results[best_model] * 100, 2)} %\n\n")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        form_data = request.form
        pregnancies = float(form_data.get('pregnancies', 0))
        glucose = float(form_data.get('glucose', 0))
        blood_pressure = float(form_data.get('blood_pressure', 0))
        skin_thickness = float(form_data.get('skin_thickness', 0))
        insulin = float(form_data.get('insulin', 0))
        bmi = float(form_data.get('bmi', 0))
        pedigree_function = float(form_data.get('pedigree_function', 0))
        age = int(form_data.get('age', 0))

        # Prepare input for the model
        new_sample = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]])
        new_sample_scaled = scaler.transform(new_sample)

        # Predict using the model
        prediction = model.predict(new_sample_scaled)[0]
        result = "Diabetes Detected" if prediction == 1 else "No Diabetes"

        # Post the result
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)