import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')


def load_data():
    data = np.loadtxt("./data/data.txt", delimiter=",", skiprows=1)
    X = data[:, :3]
    Y = data[:, 4]
    return X, Y


def plot_features(ax, X, Y, x_labels, y_labels, figure_name):
    for i in range(len(ax)):
        ax[i].scatter(X[:, i], Y)
        ax[i].set_xlabel(x_labels[i])
    ax[0].set_ylabel(y_labels)
    plt.savefig(figure_name)


def run_gradient_descent(X_train, Y_train, iterations, alpha):
    return None


def run_linear_regression():
    X_train, Y_train = load_data()
    X_features = ['size(sqft)', 'bedrooms', 'floors']

    fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    plot_features(ax, X_train, Y_train, X_features, "Price (1000's)", "./figures/features-plots.png")

    _, _, hist = run_gradient_descent(X_train, Y_train, 10, alpha=9.9e-7)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_linear_regression()
