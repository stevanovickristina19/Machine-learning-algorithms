import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import PolynomialFeatures


#priprema podataka
file_path = 'C:\\Users\\PC\\Desktop\\data.csv'
data = pd.read_csv(file_path)
y = data.iloc[:, 5]
features = data.iloc[:, :5]

for column in features.columns:
    mean = features[column].mean()
    std = features[column].std()
    features[column] = (features[column] - mean) / std

x1 = features.iloc[:, 0]
x2 = features.iloc[:, 1]
x3 = features.iloc[:, 2]
x4 = features.iloc[:, 3]
x5 = features.iloc[:, 4]
X = pd.DataFrame({

    'x1': x1,
    'x2': x2,
    'x3': x3,
    'x4': x4,
    'x5': x5,
    'x6': x1 * x1,
    'x7': x1 * x2,
    'x8': x1 * x3,
    'x9': x1 * x4,
    'x10': x1 * x5,
    'x11': x2 * x2,
    'x12': x2 * x3,
    'x13': x2 * x4,
    'x14': x2 * x5,
    'x15': x3 * x3,
    'x16': x3 * x4,
    'x17': x3 * x5,
    'x18': x4 * x4,
    'x19': x4 * x5,
    'x20': x5 * x5
})

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
Y_train = Y_train-np.mean(Y_train)
Y_test = Y_test-np.mean(Y_test)


#kros-validacija
k = 5
fold_size = len(X_train) // k
I = np.identity(X_train.shape[1])
alpha = list(range(1, 100))
mean = np.empty(len(alpha))
std = np.empty(len(alpha))


for i in range(len(alpha)):
    validation_loss = []
    for m in range(k):
        # Define the indices for the current fold
        start_idx = m * fold_size
        end_idx = (m + 1) * fold_size
        X_validation_test = X_train[start_idx:end_idx]
        Y_validation_test = Y_train[start_idx:end_idx]
        X_validation_train = np.concatenate([X_train[:start_idx], X_train[end_idx:]])
        Y_validation_train = np.concatenate([Y_train[:start_idx], Y_train[end_idx:]])

        teta0 = Y_validation_train.mean()
        matrix = np.matrix(X_validation_train.T@X_validation_train+alpha[i]*I)
        Inv = np.linalg.inv(matrix)
        teta = (Inv)@X_validation_train.T@Y_validation_train
        teta = np.array(teta)


        y_predicted = X_validation_test@teta.T+teta0
        loss = 0
        for s in range (len(y_predicted)):
            loss += ((y_predicted.iloc[s]-Y_validation_test.iloc[s])**2)

        loss = loss/y_predicted.shape[0]
        loss = np.sqrt(loss)
        validation_loss = np.append(validation_loss , loss[0])


    mean[i] = -np.mean(validation_loss)
    std[i] = np.std(validation_loss)


plt.figure()
plt.plot(alpha, mean)
s = np.sqrt(std/(len(std)))
plt.fill_between(alpha, (mean-s), (mean+s), color='blue', alpha=0.1)
plt.xlabel("Alpha")
plt.ylabel("Score")
plt.show()

max_index = np.argmax(mean)
alpha_final=alpha[max_index]
print("Optimalno alfa je: ", alpha_final)




#treniranje sa izabranim alfa
teta0_final = Y_train.mean()
matrix = np.matrix(X_train.T @ X_train + alpha_final * I)
Inv = np.linalg.inv(matrix)
teta_final = (Inv) @ X_train.T @ Y_train
teta_final = np.array(teta_final)

loss = 0

y_predicted = X_test@ teta_final.T + teta0_final

for s in range(len(y_predicted)):
    loss += ((y_predicted.iloc[s] - Y_test.iloc[s]) ** 2)

loss = loss / y_predicted.shape[0]
loss = np.sqrt(loss)
print("RMSE na test skupu je: ", loss)



















































































