import pandas as pd
import numpy as np

try:
    data = pd.read_csv('./data.csv')
    if data.empty:
        print("File data.csv is empty")
        exit()
    data.dropna(inplace=True) # Remove incomplete values
except FileNotFoundError:
    print("File data.csv not found")
    exit()

try:
    theta_data = pd.read_csv('./theta.csv')
    if theta_data.empty:
        km_mean = 1
        km_std = 1
        theta0 = 0
        theta1 = 0
    else:
        theta0 = theta_data['theta0'][0]
        theta1 = theta_data['theta1'][0]
        km_std = theta_data['km_std'][0]
        km_mean = theta_data['km_mean'][0]
except FileNotFoundError:
    km_mean = 1
    km_std = 1
    theta0 = 0
    theta1 = 0
# print(theta)
theta = np.zeros((2,1))
theta[0] = theta0
theta[1] = theta1

def predict(x, theta):
    return x.dot(theta)

print("Want to know a car price ?")
print("Enter a mileage in km (only numbers) or write EXIT to leave")
while True:
    try:
        km_input = input("Kilometers : ")
        if km_input == "EXIT":
            break
        if km_input.isdigit():
            km_input = float(km_input)
            km_input = np.array([[km_input, 1]])  # Créez un tableau NumPy avec la structure correcte
            km_input[:, 0] = (km_input[:, 0] - km_mean) / km_std
            
            result = predict(km_input, theta)
            print("The car may cost {:.0f} €".format(result[0, 0]))

        else:
            print("Please enter a mileage in km (only numbers) or write EXIT to leave ")
    except EOFError:
        print("\nBye")
        break
    except KeyboardInterrupt:
        print("\nBye")
        break