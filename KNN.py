#demo a knn with scikit

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pandas as pd

def main(printData):
    # import wineset data

    wine = datasets.load_wine()
    winedf = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    if printData:
        print(winedf)
    x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=3)  # Increase max_iter to 1000
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(predictions)
    return predictions.tolist()

if __name__ == "__main__":
    main(True)