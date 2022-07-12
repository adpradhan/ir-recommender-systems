import numpy as np
import pandas as pd
def transformData(data, min_num_user, max_num_user, num_movies):
    my_data = []
    for id_users in range(min_num_user, max_num_user + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_my_ratings = data[:, 2][data[:, 0] == id_users]
        my_ratings = np.zeros(num_movies)
        my_ratings[id_movies - 1] = id_my_ratings
        my_data.append(list(my_ratings))
    return my_data

def makeMatrix():
    training_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    training_set = np.array(training_set, dtype='int')
    test_set = np.array(test_set, dtype='int')
    num_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
    num_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
    min_num_user = 1
    max_num_user = num_users
    training_set = transformData(training_set, min_num_user, max_num_user, num_movies)
    test_set = transformData(test_set, min_num_user, max_num_user, num_movies)
    training_data = np.array([np.array(x) for x in training_set])
    test_data = np.array([np.array(x) for x in test_set])
    return (training_data, test_data)