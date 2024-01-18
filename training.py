import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


try:
    data = pd.read_csv('./data.csv')
    if data.empty:
        print("File data.csv is empty")
        exit()
    data.dropna(inplace=True) # Remove incomplete values
except FileNotFoundError:
    print("File data.csv not found")
    exit()

learning_rate = 0.01
learning_loop = 1000

X =  data['km'].values
X = np.c_[X, np.ones(X.shape[0])]
Y = data['price'].values
Y = Y.reshape(-1,1)
theta = np.zeros((2,1))
m = data.shape[0]
x_mean = X[:,0].mean()
std_x = X[:,0].std()
X[:,0] = (X[:,0] - x_mean) / std_x

def model(X, theta):
    return X.dot(theta)

def grad(X, Y, theta):
    return 1 / m * X.T.dot(model(X, theta) - Y)

def gradient_descent(X, Y, theta, learning_rate, learning_loop):
    for loop in range(learning_loop):
        theta = theta - learning_rate * grad(X, Y, theta)
    return theta

def algo_perf(prediction):
    
    mean_Y = np.mean(Y)
    SST = np.sum((Y - mean_Y) ** 2)
    SSR = np.sum((Y - prediction) ** 2)
    R_sq = 1 - ((SSR / SST))
    print("Coefficient de détermination R² :", R_sq)


theta = gradient_descent(X, Y, theta, learning_rate, learning_loop)
datacsv = {'theta0': [theta[0,0]], 'theta1': [theta[1,0]], 'km_std': [std_x], 'km_mean': [x_mean]}
theta_data = pd.DataFrame(datacsv, columns=['theta0', 'theta1', 'km_std', 'km_mean'])
theta_data.to_csv('./theta.csv', index=False)
predict = model(X, theta)
algo_perf(predict)
plt.scatter(data['km'], data['price'], color='blue',s=10)
plt.plot(data['km'], predict, color='red')
plt.xlabel('Mileage(Km)', color='blue')
plt.ylabel('Price(€)',color='blue')
plt.title('Scatter distribution of price vs km', color='blue')
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(40000))

if __name__ == "__main__":
    plt.show()
