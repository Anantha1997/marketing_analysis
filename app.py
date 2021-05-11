from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'random_forest_classifier_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        job = int(request.form['job'])
        marital = int(request.form['marital'])
        education = int(request.form['education'])
        balance = int(request.form['balance'])
        housing = float(request.form['housing'])
        loan = float(request.form['loan'])
        contact = int(request.form['contact'])
        day = int(request.form['day'])
        month = int(request.form['month'])
        duration = int(request.form['duration'])
        campaign = int(request.form['campaign'])
        previous = int(request.form['previous'])
        poutcome = int(request.form['poutome'])
        

        data = np.array([[age, job, marital, education, balance, housing, loan, contact, day, month, duration, campaign, previous, poutcome]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
    
    