import copy
import math

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')


def load_data():
    data = np.loadtxt("./data/aircraft-data-training.txt", delimiter="|", usecols=[1,2,3,4])
    X = data[:, :3]
    Y = data[:, 3]
    return X, Y


def plot_features(ax, X, Y, x_labels, y_labels, figure_name):
    for i in range(len(ax)):
        ax[i].scatter(X[:, i], Y)
        ax[i].set_xlabel(x_labels[i])
    ax[0].set_ylabel(y_labels)
    plt.savefig(figure_name)


def gradient_descent(X, Y, w_in, b_in, cost_function, gradient_function, alpha, iterations):
    m = len(X)

    # An array to store values at each iteration primarily for graphing later
    hist = {}
    hist["cost"] = []
    hist["params"] = []
    hist["grads"] = []
    hist["iter"] = []

    w = copy.deepcopy(w_in)
    b = b_in

    save_interval = np.ceil(iterations / 10000)  # prevent resource exhaustion for long runs

    print(
        f"Iteration Cost          w0       w1       w2        b       djdw0    djdw1    djdw2    djdb  ")
    print(
        f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|")

    for i in range(iterations):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, Y, w, b)

        # update parameters simultanously
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        # Save cost J,w,b at each save interval for graphing
        if i == 0 or i % save_interval == 0:
            hist["cost"].append(cost_function(X, Y, w, b))
            hist["params"].append([w, b])
            hist["grads"].append([dj_dw, dj_db])
            hist["iter"].append(i)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iterations / 10) == 0:
            # print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")
            cst = cost_function(X, Y, w, b)
            print(
                f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_db: 0.1e}")

    return w, b, hist  # return w,b and history for graphing


def compute_cost(X, Y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        predicted = np.dot(X[i], w) + b
        cost = cost + (predicted - Y[i]) ** 2
    cost = cost / (2 * m)
    return (np.squeeze(cost))


def compute_gradient_matrix(X, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
      X : (array_like Shape (m,n)) variable such as house size
      y : (array_like Shape (m,1)) actual value
      w : (array_like Shape (n,1)) Values of parameters of the model
      b : (scalar )                Values of parameter of the model
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.

    """
    m, n = X.shape
    f_wb = X @ w + b
    e = f_wb - y
    dj_dw = (1 / m) * (X.T @ e)
    dj_db = (1 / m) * np.sum(e)

    return dj_db, dj_dw


def run_gradient_descent(X, Y, iterations, alpha=1e-6):
    m, n = np.shape(X)

    init_w = np.zeros(n)
    init_b = 0

    # run gradient descent
    w_out, b_out, hist_out = gradient_descent(X, Y, init_w, init_b,
                                              compute_cost, compute_gradient_matrix, alpha, iterations)
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.2f}")

    return (w_out, b_out, hist_out)


def plot_cost_i_w(X, y, hist,filename):
    ws = np.array([p[0] for p in hist["params"]])
    rng = max(abs(ws[:, 0].min()), abs(ws[:, 0].max()))
    wr = np.linspace(-rng + 0.27, rng + 0.27, 20)
    cst = [compute_cost(X, y, np.array([wr[i], -32, -67]), 221) for i in range(len(wr))]

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
    plt.savefig(filename)


def zscore_normalize_features(X):
    u = np.mean(X,axis=0)
    o = np.std(X,axis=0)
    X_norm = (X-u)/o

    return (X_norm,u,o)


def run_linear_regression():
    X_train, Y_train = load_data()
    X_features = ['year', 'seats', 'hours']

    fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    plot_features(ax, X_train, Y_train, X_features, "Price (1000's)", "./figures/features-plots.png")

    _, _, hist = run_gradient_descent(X_train, Y_train, 10, alpha=1e-7)
    plot_cost_i_w(X_train, Y_train, hist,"./figures/unnormalized-features-cost-graph.png")

    X_norm, u, o = zscore_normalize_features(X_train)
    print(f"X_mu = {u}, \nX_sigma = {o}")
    print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
    print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

    w_norm, b_norm, hist = run_gradient_descent(X_norm, Y_train, 1000, 1.0e-1, )

    # predict target using normalized features
    m = X_norm.shape[0]
    yp = np.zeros(m)
    for i in range(m):
        yp[i] = np.dot(X_norm[i], w_norm) + b_norm

        # plot predictions and targets versus original features
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    plot_features(ax, X_norm, Y_train,X_features, "Price (1000's)", "./figures/features-unnormalized-plots.png")

    # First, normalize out example.
    x_house = np.array([2007, 4, 510])
    x_house_norm = (x_house - u) / o
    print(x_house_norm)
    x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
    print(
        f" predicted price of aircraft with 2007, 4 seats, 510 hours, 40 years old = ${x_house_predict * 1000:0.0f}")

if __name__ == '__main__':
    run_linear_regression()
