import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

file_path = 'C:\\Users\\PC\\Desktop\\Mašinsko projekti\\domaci2\\multiclass_data.csv'
data = pd.read_csv(file_path)
new_column_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
data.columns = new_column_names
categories = data.iloc[:, 5]
x = data.iloc[:, :5]
x_train, x_test, y_train, y_test = train_test_split(x, categories, test_size=0.2, random_state=42)
combined_df = pd.concat([x_train, y_train], axis=1)


num_0 = len(y_train[y_train == 0])
num_1 = len(y_train[y_train == 1])
num_2 = len(y_train[y_train == 2])


prior_0 = num_0 / len(y_train)
prior_1 = num_1 / len(y_train)
prior_2 = num_2 / len(y_train)

#learning parameters

stats_by_category = combined_df.groupby('y').agg({'x1': ['mean', 'std'],
                                                  'x2': ['mean', 'std'],
                                                  'x3': ['mean', 'std'],
                                                  'x4': ['mean', 'std'],
                                                  'x5': ['mean', 'std']})

mi_0 = stats_by_category.loc[0, [('x1', 'mean'), ('x2', 'mean'), ('x3', 'mean'), ('x4', 'mean'), ('x5', 'mean')]].values
mi_1 = stats_by_category.loc[1, [('x1', 'mean'), ('x2', 'mean'), ('x3', 'mean'), ('x4', 'mean'), ('x5', 'mean')]].values
mi_2 = stats_by_category.loc[2, [('x1', 'mean'), ('x2', 'mean'), ('x3', 'mean'), ('x4', 'mean'), ('x5', 'mean')]].values

sigma_0 = stats_by_category.loc[0, [('x1', 'std'), ('x2', 'std'), ('x3', 'std'), ('x4', 'std'), ('x5', 'std')]].values
sigma_1 = stats_by_category.loc[1, [('x1', 'std'), ('x2', 'std'), ('x3', 'std'), ('x4', 'std'), ('x5', 'std')]].values
sigma_2 = stats_by_category.loc[2, [('x1', 'std'), ('x2', 'std'), ('x3', 'std'), ('x4', 'std'), ('x5', 'std')]].values

#model and test
def prob(x, mean, std):
    exp = np.exp(-((x-mean)**2)/(2*(std**2)))
    return (1/(std*np.sqrt(2*math.pi)))*exp

predictions = np.zeros(x_test.shape[0])
for j in range(x_test.shape[0]):

    expression0 = 1
    for i in range(x_test.shape[1]):
        expression0 = expression0*prob(x_test.iloc[j, i], mi_0[i], sigma_0[i])
    expression0 = expression0*prior_0

    expression1 = 1
    for i in range(x_test.shape[1]):
        expression1 = expression1*prob(x_test.iloc[j, i], mi_1[i], sigma_1[i])
    expression1 = expression1*prior_1

    expression2 = 1
    for i in range(x_test.shape[1]):
        expression2 = expression2*prob(x_test.iloc[j, i], mi_2[i], sigma_2[i])
    expression2 = expression2*prior_2

    sum = expression0+expression1+expression2
    p_0_x = expression0/sum
    p_1_x = expression1/sum
    p_2_x = expression2/sum
    p = [p_0_x, p_1_x, p_2_x]

    predictions[j] = np.argmax(p)

true=0
for i in range(len(predictions)):
    if predictions[i] == y_test.iloc[i]:
            true += 1

print('Tacnost : ', true / len(predictions) * 100)

predictions_train = np.zeros(x_train.shape[0])
for j in range(x_train.shape[0]):

    expression0 = 1
    for i in range(x_test.shape[1]):
        expression0 = expression0*prob(x_train.iloc[j, i], mi_0[i], sigma_0[i])
    expression0 = expression0*prior_0

    expression1 = 1
    for i in range(x_test.shape[1]):
        expression1 = expression1*prob(x_train.iloc[j, i], mi_1[i], sigma_1[i])
    expression1 = expression1*prior_1

    expression2 = 1
    for i in range(x_test.shape[1]):
        expression2 = expression2*prob(x_train.iloc[j, i], mi_2[i], sigma_2[i])
    expression2 = expression2*prior_2

    sum = expression0+expression1+expression2
    p_0_x = expression0/sum
    p_1_x = expression1/sum
    p_2_x = expression2/sum
    p = [p_0_x, p_1_x, p_2_x]

    predictions_train[j] = np.argmax(p)

true=0
for i in range(len(predictions_train)):
    if predictions_train[i] == y_train.iloc[i]:
            true += 1

print('Tacnost na trening skupu: ', true / len(predictions_train) * 100)









