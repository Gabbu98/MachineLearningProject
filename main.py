import copy
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('dark_background')
dlblue = '#0096ff';
dlorange = '#FF9300';
dldarkred = '#C00000';
dlmagenta = '#FF40FF';
dlpurple = '#7030A0';
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
dlc = dict(dlblue='#0096ff', dlorange='#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')


def load_data():
    # Load the data from the CSV file using pandas
    df = pd.read_csv("./data/dataset.csv")

    # Exclude the first row (index 0)
    X = df.iloc[1:, [6, 7]]

    # Convert the elements in 'X' to floats and convert to NumPy array
    X = X.astype(float).to_numpy()

    # Extract the target variable 'y' from the DataFrame and convert to NumPy array
    y = df.iloc[1:, 11].astype(float).to_numpy()

    m = np.shape(X)
    new_X = []
    new_y = []
    for i in range(m[0]):
        if not np.isnan(X[i][0]):
            new_X.append(X[i])
            new_y.append(y[i])

    new_X = np.asarray(new_X)
    new_y = np.asarray(new_y)

    # Convert 'X' to its scientific notation representation (as float64)
    X_scientific = np.vectorize(lambda x: "{:.6e}".format(x) if not np.isnan(x) else "1.0")(new_X)

    # Convert 'X_scientific' back to float64
    X_scientific = X_scientific.astype(np.float64)

    return X_scientific, new_y


# This version saves more values and is more verbose than the assigment versons
def gradient_descent_houses(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = len(X)

    hist = {}
    hist['cost'] = []
    hist['params'] = []
    hist['grads'] = []
    hist['iter'] = []

    w = copy.deepcopy(w_in)
    b = b_in
    save_interval = np.ceil(num_iters / 1000)

    print(
        f"Iteration Cost          w0       w1       w2        b       djdw0    djdw1    djdw2    djdb  ")
    print(
        f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|")

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        # Save cost J,w,b at each save interval for graphing
        if i == 0 or i % save_interval == 0:
            hist["cost"].append(cost_function(X, y, w, b))
            hist["params"].append([w, b])
            hist["grads"].append([dj_dw, dj_db])
            hist["iter"].append(i)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            # print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")
            cst = cost_function(X, y, w, b)
            print(
                f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_db: 0.1e}")

    return w, b, hist  # return w,b and history for graphing


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        predicted = np.dot(X[i], w) + b
        cost = cost + (predicted - y[i]) ** 2
    cost = cost / (2 * m)
    return (np.squeeze(cost))


def compute_gradient_matrix(X, y, w, b):
    m, n = X.shape
    f_wb = X @ w + b
    e = f_wb - y
    dj_dw = (1 / m) * (X.T @ e)
    dj_db = (1 / m) * np.sum(e)

    return dj_db, dj_dw


def run_gradient_descent(X, y, iterations=1000, alpha=1e-6):
    m, n = X.shape

    # init the params
    init_w = np.zeros(n)
    init_b = 0

    # run gradient descent
    output_w, output_b, output_histogram = gradient_descent_houses(X, y, init_w, init_b, compute_cost,
                                                                   compute_gradient_matrix, alpha, iterations)

    print(f"w,b found by gradient descent: w: {output_w}, b: {output_b:0.2f}")

    return (output_w, output_b, output_histogram)


def plot_cost_i_w(X, y, hist, img):
    ws = np.array([p[0] for p in hist["params"]])
    rng = max(abs(ws[:, 0].min()), abs(ws[:, 0].max()))
    wr = np.linspace(-rng + 1.27, rng + 1.27, 20)
    cst = [compute_cost(X, y, np.array([-32, -67]), 221) for i in range(len(wr))]

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(hist["iter"], (hist["cost"]));
    ax[0].set_title("Cost vs Iteration")
    ax[0].set_xlabel("iteration");
    ax[0].set_ylabel("Cost")
    ax[1].plot(wr, cst);
    ax[1].set_title("Cost vs w[0]")
    ax[1].set_xlabel("w[0]");
    ax[1].set_ylabel("Cost")
    ax[1].plot(ws[:, 0], hist["cost"])
    plt.savefig(img)


def zscore_normalize_features(X):
    # mu of the jth column
    mean = np.mean(X, axis=0)

    # standard deviation of the jth column
    stnd_dev = np.std(X, axis=0)

    m = np.shape(X)

    X_norm = (X - mean) / stnd_dev

    return (X_norm, mean, stnd_dev)


def run_linear_regression():
    # load the dataset
    X_train, y_train = load_data()

    X_features = ['Bedrooms', 'Bathrooms']

    w, b, hist = run_gradient_descent(X_train, y_train, 10, alpha=1e-7)

    plot_cost_i_w(X_train, y_train, hist, "not_normalized.png")

    # normalize the original features
    X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
    # print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
    # print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train, axis=0)}")
    # print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm, axis=0)}")

    w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

    # plot_cost_i_w(X_train, y_train, hist,"normalized_features_cost.png")

    # predict target using normalized features
    m = X_norm.shape[0]
    yp = np.zeros(m)
    for i in range(m):
        yp[i] = np.dot(X_norm[i], w_norm) + b_norm

        # plot predictions and targets versus original features
    fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:, i], y_train, label='target')
        ax[i].set_xlabel(X_features[i])
        ax[i].scatter(X_train[:, i], yp, color=dlc["dlorange"], label='predict')
    ax[0].set_ylabel("Price");
    ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.savefig('targets.png')

    # First, normalize out example.
    x_house = np.array([3.0, 2.0])
    #x_house_norm = (x_house - X_mu) / X_sigma
    # print(x_house_norm)
    x_house_predict = np.dot(x_house, w) + b
    print(f" predicted price of a house with 150 sqft, 3 bedrooms, 2 bathrooms${x_house_predict}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_linear_regression()
