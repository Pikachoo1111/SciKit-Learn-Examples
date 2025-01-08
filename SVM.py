# svm with wine set

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#What is SVM?
#Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for classification or regression problems. It uses a technique called the kernel trick to transform your data and then based on these transformations it finds an optimal boundary between the possible outputs.
#What is the kernel trick?
#The kernel trick is a method of using a linear classifier to solve a non-linear problem. It transforms the input data into a higher dimensional space and then finds a linear boundary in that space. This allows you to solve non-linear problems with a linear classifier.

import pandas as pd

def main(printData):
    # import wineset data
    wine = datasets.load_wine()
    winedf = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    if printData:
        print(winedf)
    x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2)
    model = SVC(max_iter=10000)  # Increase max_iter to 1000
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(predictions)
    if(printData):
        print(accuracy_score(y_test, predictions))

if __name__ == "__main__":
    main(True)
