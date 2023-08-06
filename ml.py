from flask import Flask, request, render_template
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)
data = pd.read_csv('data.csv')
X = data[['ph', 'TDS', 'Turbidity']]
y = data['Potabailty']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
joblib.dump(rfc, 'model.pkl')
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get pH, TDS, and turbidity values from the form
    ph = float(request.form['pH'])
    tds = float(request.form['TDS'])
    turbidity = float(request.form['turbidity'])

    # Make a prediction using the model
    
    quality = model.predict([[ph, tds, turbidity]])

    # Convert the prediction to a human-readable string
    if quality[0] == 0:
        quality_str = 'Poor'
    else:
        quality_str = 'Good'

    # Render the prediction on the results page
    return render_template('result.html', quality=quality_str)

if __name__ == '__main__':
    app.run()
