# demo logistic regression with scikit

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd

# import wineset data

def main(printData):
    wine = datasets.load_wine()
    winedf = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    if printData:
        print(winedf)
    x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2)
    model = LogisticRegression(max_iter=10000)  # Increase max_iter to 1000
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(predictions)
    if(printData):
        print(accuracy_score(y_test, predictions))
    return predictions.tolist()

if __name__ == "__main__":
    main(True)