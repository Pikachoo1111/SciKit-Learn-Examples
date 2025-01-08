# Linear Regression Sample File

# import scikit
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

import pandas as pd

def main(printData):
    # import wineset data
    wine = datasets.load_wine()
    winedf = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    if printData:
        print(winedf)
    
    x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2)
    model = LinearRegression(max_iter=10000)
    model.fit(x_train, y_train)
    
    # Predict on the test data
    predictions = model.predict(x_test)
    print(predictions)
    
    if printData:
        print(f'Accuracy: {accuracy_score(y_test, predictions)}')

if __name__ == "__main__":
    main(True)
