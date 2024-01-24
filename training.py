import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from predi import load_csv



def model(X, theta):
    """
    Simple fonction to calculate the price of a car with a mileage in km
    args: X, theta
    X: mileage in km
    theta: thetas value
    return: Price of the car
    """
    return X.dot(theta)

def grad(X, Y, theta):
    """
    This function calculate the gradient 
    args: X, Y, theta
    X: mileage in km
    Y: price of the car
    theta: thetas value
    return: Gradient to update the thetas
    """
    return 1 / X.shape[0] * X.T.dot(model(X, theta) - Y)

def gradient_descent(X, Y, theta, learning_rate, learning_loop):
    """
    This function calculate the gradient descent
    args: X, Y, theta, learning_rate, learning_loop
    X: mileage in km
    Y: price of the car
    theta: thetas value
    learning_rate: learning rate of the gradient descent
    learning_loop: number of iterations
    return: thetas before training and cost history to be able to plot the cost function
    """
    cost_history = np.zeros(learning_loop)
    for loop in range(learning_loop):
        theta = theta - learning_rate * grad(X, Y, theta)
        cost_history[loop] = cost_function(X, Y, theta)
    return theta, cost_history

def cost_function(X, Y, theta):
    """
    This function calculate the cost function
    args: X, Y, theta
    X: mileage in km
    Y: price of the car
    theta: thetas value
    return: Cost determined by the thetas to fill a cost history
    """
    return 1 / (2 * X.shape[0]) * np.sum((model(X, theta) - Y) ** 2)

def algo_perf(prediction, Y):
    """
    This function calculate the coefficient of determination R² to evaluate the performance of the algorithm
    args: prediction, Y
    prediction: prediction of the price of the car
    Y: price of the car
    return: Coefficient of determination R²
    """
    mean_Y = np.mean(Y)
    SST = np.sum((Y - mean_Y) ** 2)
    SSR = np.sum((Y - prediction) ** 2)
    R_sq = 1 - ((SSR / SST))
    print("Coefficient de détermination R² :", R_sq)


def print_stats(data, cost_history, predict, learning_loop):
    """
    This function plot the cost function and the scatter distribution of price vs km
    args: data, cost_history, predict, learning_loop
    data: data from the csv file
    cost_history: cost history to plot the cost function
    predict: prediction of the price of the car
    learning_loop: number of iterations
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(range(learning_loop), cost_history, color='blue')
    ax1.set_title('Cost function', color='blue')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Cost')

    ax2.scatter(data['km'], data['price'], color='blue', s=10)
    ax2.plot(data['km'], predict, color='red')
    ax2.set_xlabel('Mileage (Km)', color='blue')
    ax2.set_ylabel('Price (€)', color='blue')
    ax2.set_title('Scatter distribution of price vs km', color='blue')
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    ax2.xaxis.set_major_locator(plt.MultipleLocator(40000))
    plt.tight_layout()
    plt.show()

def main():
    """
    Program to train the model with a linear regression
    """
    data = load_csv('./data.csv')
    learning_rate = 0.01
    learning_loop = 1000
    X =  data['km'].values
    X = np.c_[X, np.ones(X.shape[0])]
    Y = data['price'].values
    Y = Y.reshape(-1,1)
    theta = np.zeros((2,1))
    x_mean = X[:,0].mean()
    std_x = X[:,0].std()
    X[:,0] = (X[:,0] - x_mean) / std_x
    theta, cost_history = gradient_descent(X, Y, theta, learning_rate, learning_loop)
    datacsv = {'theta0': [theta[0,0]], 'theta1': [theta[1,0]], 'km_std': [std_x], 'km_mean': [x_mean]}
    theta_data = pd.DataFrame(datacsv, columns=['theta0', 'theta1', 'km_std', 'km_mean'])
    theta_data.to_csv('./theta.csv', index=False)
    predict = model(X, theta)
    algo_perf(predict, Y)
    print_stats(data, cost_history, predict, learning_loop)


if __name__ == "__main__":
    main()