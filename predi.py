import pandas as pd
import numpy as np

def load_csv(csv_file):
    """
    This function load a .csv file and return a pandas dataframe
    args: csv_file
    csv_file: path of the .csv file
    return: Datas of the .csv file
    """
    try:
        data = pd.read_csv(csv_file)
        if data.empty:
            print("File .csv is empty")
            exit()
        data.dropna(inplace=True) # Remove incomplete values
        return data
    except FileNotFoundError:
        print("File .csv not found")
        exit()

def predict(x, theta):
    """
    Simple fonction to calculate the price of a car with a mileage in km
    args: x, theta
    x: mileage in km
    theta: thetas value
    return: Price of the car
    """
    return x.dot(theta)

def predict_result():
    """
    This function predict the price of a car with a mileage in km
    The result depends on the thetas value.
    If you train the model before using it you will have a better result.
    """
    data = load_csv('./data.csv')
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

    theta = np.zeros((2,1))
    theta[0] = theta0
    theta[1] = theta1

    print("Want to know a car price ?")
    print("Enter a mileage in km (only numbers) or write EXIT to leave")
    while True:
        try:
            km_input = input("Kilometers : ")
            if km_input == "EXIT":
                break
            if km_input.isdigit():
                km_input = float(km_input)
                km_input = np.array([[km_input, 1]])
                km_input[:, 0] = (km_input[:, 0] - km_mean) / km_std
                
                result = predict(km_input, theta)
                if result[0, 0] < 0:
                    result[0, 0] = 0
                print("The car may cost {:.0f} €".format(result[0, 0]))

            else:
                print("Please enter a mileage in km (only numbers) or write EXIT to leave ")
        except EOFError:
            print("\nBye")
            break
        except KeyboardInterrupt:
            print("\nBye")
            break

def main():
    """
    This program predict the price of a car with a mileage in km
    """
    predict_result()

if __name__ == "__main__":
    main()