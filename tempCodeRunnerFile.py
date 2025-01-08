from flask import Flask, request, jsonify
from LinearRegression import main as linear_regression_main
from LogisticRegression import main as logistic_regression_main
from KNN import main as knn_main
from SVM import main as svm_main

svm_main(False)
app = Flask(__name__)

@app.route('/getResults.py', methods=['GET'])
def get_model_results():
    model = request.args.get('model')
    
    if model == 'LinearRegression':
        results = linear_regression_main(False)
    elif model == 'LogisticRegression':
        results = logistic_regression_main(False)
    elif model == 'KNN':
        results = knn_main(False)
    elif model == 'SVM':
        results = svm_main(False)
    else:
        results = {'error': 'Invalid model name'}
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)