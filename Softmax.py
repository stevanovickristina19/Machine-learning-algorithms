import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from matplotlib import pyplot as plt


file_path = 'C:\\Users\\PC\\Desktop\\Mašinsko projekti\\domaci2\\multiclass_data.csv'
data = pd.read_csv(file_path)

x = data.iloc[:, :5]
categories = data.iloc[:, 5]
scaler = StandardScaler()
x = scaler.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, categories, test_size=0.2, random_state=42)

X_train_df = pd.DataFrame(X_train)
y_train_df = pd.DataFrame(y_train)
X_train_df.reset_index(drop=True, inplace=True)
y_train_df.reset_index(drop=True, inplace=True)
X_train = X_train_df.to_numpy()
y_train = y_train_df.to_numpy()
n = X_train.shape[1]
m = X_train.shape[0]
np.random.seed(1)


def choose_random_batch(X_train,y_train, l):
    m = len(X_train)
    random_indices = random.sample(range(m), l)
    x_mb = X_train[random_indices, :]
    y_mb = y_train[random_indices]
    return x_mb, y_mb

num_categories = 3
num_predictors = X_train.shape[1]
num_examples = X_train.shape[0]

batch_sizes = [4, 8, 16, 32, 64, 128]
learning_rates = [0.001, 0.01, 0.1, 1, 10]

#ispitivanje konvergencije za sve kombinacije alfa i batch_size
for a in learning_rates:
    for batch_size in batch_sizes:
        teta = np.zeros((num_predictors, num_categories))
        max_iteracija = 4000 // batch_size
        J_iter = np.zeros(max_iteracija)

        for iteracija in range(max_iteracija):
            x_mb, y_mb = choose_random_batch(X_train, y_train, batch_size)
            grad_teta_0 = np.zeros_like(teta[:, 0])
            grad_teta_1 = np.zeros_like(teta[:, 1])

            for i in range(batch_size):
                exp1 = np.exp(teta[:, 0].T @ x_mb[i])
                exp2 = np.exp(teta[:, 1].T @ x_mb[i])
                exp3 = np.exp(teta[:, 2].T @ x_mb[i])
                s = exp1 + exp2 + exp3

                expression1 = (y_mb[i] == 0).astype(int) - (exp1 / s)
                expression2 = (y_mb[i] == 1).astype(int) - (exp2 / s)

                grad_teta_0 += x_mb[i, :] * expression1
                grad_teta_1 += x_mb[i, :] * expression2

                J_iter[iteracija] += teta[:, y_mb[i]].T @ x_mb[i] - np.log(s)


            teta[:, 0] = teta[:, 0] + a * grad_teta_0
            teta[:, 1] = teta[:, 1] + a * grad_teta_1
            teta[:, 2] = 0

        plt.figure(figsize=(8, 5))
        plt.plot(range(batch_size, (max_iteracija + 1) * batch_size, batch_size), J_iter, color='red')

        plt.xlabel('Number of Training Examples Processed')
        plt.ylabel('J Values')
        plt.title(f'Learning Rate {a}, Batch Size {batch_size}')
        plt.grid(True)
        plt.show()


#optimal batch_size = 64
opt_batch_size = 64
learning_rates = [0.001, 0.01, 0.1, 1]
for a in learning_rates:

        teta = np.zeros((num_predictors, num_categories))
        max_iteracija = 4000 // opt_batch_size
        J_iter = np.zeros(max_iteracija)

        for iteracija in range(max_iteracija):
            x_mb, y_mb = choose_random_batch(X_train, y_train, opt_batch_size)
            grad_teta_0 = np.zeros_like(teta[:, 0])
            grad_teta_1 = np.zeros_like(teta[:, 1])

            for i in range(opt_batch_size):
                exp1 = np.exp(teta[:, 0].T @ x_mb[i])
                exp2 = np.exp(teta[:, 1].T @ x_mb[i])
                exp3 = np.exp(teta[:, 2].T @ x_mb[i])
                s = exp1 + exp2 + exp3

                expression1 = (y_mb[i] == 0).astype(int) - (exp1 / s)
                expression2 = (y_mb[i] == 1).astype(int) - (exp2 / s)

                grad_teta_0 += x_mb[i, :] * expression1
                grad_teta_1 += x_mb[i, :] * expression2

                J_iter[iteracija] += teta[:, y_mb[i]].T @ x_mb[i] - np.log(s)


            teta[:, 0] = teta[:, 0] + a * grad_teta_0
            teta[:, 1] = teta[:, 1] + a * grad_teta_1
            teta[:, 2] = 0

        plt.figure(figsize=(8, 5))
        plt.plot(range(opt_batch_size, (max_iteracija + 1) * opt_batch_size, opt_batch_size), J_iter, color='green')

        plt.xlabel('Number of Training Examples Processed')
        plt.ylabel('J Values')
        plt.title(f'Learning Rate {a}, Optimal Batch Size {opt_batch_size}')
        plt.grid(True)
        plt.show()

#optimal alpha = 0.1
a_opt = 0.1
batch_sizes = [4, 8, 16, 32, 64, 128]
for batch_size in batch_sizes:
    teta = np.zeros((num_predictors, num_categories))
    max_iteracija = 4000 // batch_size
    J_iter = np.zeros(max_iteracija)

    for iteracija in range(max_iteracija):
        x_mb, y_mb = choose_random_batch(X_train, y_train, batch_size)
        grad_teta_0 = np.zeros_like(teta[:, 0])
        grad_teta_1 = np.zeros_like(teta[:, 1])

        for i in range(batch_size):
            exp1 = np.exp(teta[:, 0].T @ x_mb[i])
            exp2 = np.exp(teta[:, 1].T @ x_mb[i])
            exp3 = np.exp(teta[:, 2].T @ x_mb[i])
            s = exp1 + exp2 + exp3

            expression1 = (y_mb[i] == 0).astype(int) - (exp1 / s)
            expression2 = (y_mb[i] == 1).astype(int) - (exp2 / s)

            grad_teta_0 += x_mb[i, :] * expression1
            grad_teta_1 += x_mb[i, :] * expression2

            J_iter[iteracija] += teta[:, y_mb[i]].T @ x_mb[i] - np.log(s)


        teta[:, 0] = teta[:, 0] + a_opt * grad_teta_0
        teta[:, 1] = teta[:, 1] + a_opt * grad_teta_1
        teta[:, 2] = 0

    plt.figure(figsize=(8, 5))
    plt.plot(range(batch_size, (max_iteracija + 1) * batch_size, batch_size), J_iter, color='blue')

    plt.xlabel('Number of Training Examples Processed')
    plt.ylabel('J Values')
    plt.title(f'Optimal Learning Rate {a_opt}, Batch Size {batch_size}')
    plt.grid(True)
    plt.show()



#optimalno alfa i optimalan batch_size i testiranje
a = 0.1
batch_size = 64
teta_final = np.zeros((num_predictors, num_categories))
max_iteracija = 4000 // batch_size


for iteracija in range(max_iteracija):
        x_mb, y_mb = choose_random_batch(X_train, y_train, batch_size)
        grad_teta_0 = np.zeros_like(teta_final[:, 0])
        grad_teta_1 = np.zeros_like(teta_final[:, 1])

        for i in range(batch_size):
            exp1 = np.exp(teta_final[:, 0].T @ x_mb[i])
            exp2 = np.exp(teta_final[:, 1].T @ x_mb[i])
            exp3 = np.exp(teta_final[:, 2].T @ x_mb[i])
            s = exp1 + exp2 + exp3

            expression1 = (y_mb[i] == 0).astype(int) - (exp1 / s)
            expression2 = (y_mb[i] == 1).astype(int) - (exp2 / s)

            grad_teta_0 += x_mb[i, :] * expression1
            grad_teta_1 += x_mb[i, :] * expression2



        teta_final[:, 0] = teta_final[:, 0] + a * grad_teta_0
        teta_final[:, 1] = teta_final[:, 1] + a * grad_teta_1
        teta_final[:, 2] = 0

predictions = np.zeros(len(y_test))
for i in range(len(y_test)):

    exp1 = np.exp(teta_final[:, 0].T @ X_test[i])
    exp2 = np.exp(teta_final[:, 1].T @ X_test[i])
    exp3 = np.exp(teta_final[:, 2].T @ X_test[i])
    s = exp1+exp2+exp3
    fi = [exp1/s, exp2/s, exp3/s]
    predictions[i] = np.argmax(fi)

true = 0
for i in range(len(predictions)):
    if predictions[i] == y_test.iloc[i]:
        true += 1

print('Tacnost na test skupu: ', true / len(predictions) * 100, '%')

#tacnost na trening skupu
predictions_train = np.zeros(len(y_train))
for i in range(len(y_train)):

    exp1 = np.exp(teta_final[:, 0].T @ X_train[i])
    exp2 = np.exp(teta_final[:, 1].T @ X_train[i])
    exp3 = np.exp(teta_final[:, 2].T @ X_train[i])
    s = exp1+exp2+exp3
    fi = [exp1/s, exp2/s, exp3/s]
    predictions_train[i] = np.argmax(fi)

true = 0
for i in range(len(predictions_train)):
    if predictions_train[i] == y_train[i]:
        true += 1

print('Tacnost na trening skupu: ', true / len(predictions_train) * 100, '%')