import numpy as np
from sklearn import  linear_model
def load_data():
    data = np.loadtxt("./data/aircraft-data-training.txt", delimiter="|", usecols=[1,2,3,4])
    X = data[:, :3]
    Y = data[:, 3]
    return X, Y


if __name__ == '__main__':
    x_train, y_train = load_data()

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    predict = regr.predict(np.array([1957,2,2641]).reshape(-1,3))

    print(predict)
    print(regr.coef_)