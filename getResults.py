from flask import Flask, request, jsonify, send_from_directory
from LinearRegression import main as linear_regression_main
from LogisticRegression import main as logistic_regression_main
from KNN import main as knn_main
from SVM import main as svm_main
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  
@app.route('/getResults', methods=['GET'])  # Updated route name
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

@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
