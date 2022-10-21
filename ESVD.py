# ESVD Implementation from IEEE Journal Article 2017

# Pseudocode lifted from 10.1109/ACCESS.2017.2772226

import numpy as np
import scipy as sp

def rmse(truth, pred):
    T = truth.shape[0] * truth.shape[1]
    errors = np.sum((truth - pred)**2)
    return np.sqrt(errors / T)

def generate_utility_matrix(df):
    utility_R = np.zeros((df["user_id"].max(), df["movie_id"].max()))
    for i, row in df.iterrows():
        utility_R[int(row["user_id"]) - 1, int(row["movie_id"]) - 1] = row["rating"]
    return utility_R

def get_popular_movies_list(df):
    popular_movies = df.groupby("movie_id")["rating"].count().sort_values(ascending=False)
    popular_movies_list = list(map(lambda idx: idx - 1, popular_movies.index.tolist())) # convert from 1-indexing to 0-indexing
    return popular_movies_list

def get_active_users_list(df):
    active_users = df.groupby("user_id")["rating"].count().sort_values(ascending=False)
    active_users_list = list(map(lambda idx: idx - 1, active_users.index.tolist())) # convert from 1-indexing to 0-indexing
    return active_users_list

def get_intersection_matrix(utility_R, n, popular_movies_list, active_users_list):
    intersection_matrix = np.zeros(utility_R.shape)
    for i in range(n):
        for j in range(n):
            intersection_matrix[i,j] = utility_R[active_users_list[i], popular_movies_list[j]]
    intersection_matrix = intersection_matrix[:, ~np.all(intersection_matrix == 0, axis=0)]
    intersection_matrix = intersection_matrix[~np.all(intersection_matrix == 0, axis=1), :]
    return intersection_matrix

def rsvd(a):
    U, S, Vh = np.linalg.svd(a, full_matrices=True)
    Sd = sp.linalg.diagsvd(S, *a.shape)
    a_r = U @ Sd @ Vh
    return a_r, U, S, Vh

def predict_empty(a, U, S, Vh):
    pred = np.zeros((U.shape[0], Vh.shape[0]))
    Sd = sp.linalg.diagsvd(S, *a.shape)
    US = U @ Sd
    for i in range(US.shape[0]):
        for j in range(Vh.shape[0]):
            if a[i,j] == 0:
                pred[i,j] = np.abs(US[i,:] @ Vh.T[:,j])
            else:
                pred[i,j] = a[i,j]
    return pred

def predict_all(a, U, S, Vh):
    pred = np.zeros((U.shape[0], Vh.shape[0]))
    Sd = sp.linalg.diagsvd(S, *a.shape)
    US = U @ Sd
    for i in range(US.shape[0]):
        for j in range(Vh.shape[0]):
            pred[i,j] = np.abs(US[i,:] @ Vh.T[:,j])
    return pred

def remap_intersection_to_utility(utility_R, predicted_intersection, active_users_list, popular_movies_list):
    u = k = 0
    for i in active_users_list:
        k = 0
        for j in popular_movies_list:
            utility_R[i,j] = predicted_intersection[u,k]
            # print(i,j,u,k)
            k += 1
        u += 1
    return utility_R

def fit(df, n_top):
    utility = generate_utility_matrix(df)
    pop_movies = get_popular_movies_list(df)
    act_users = get_active_users_list(df)
    n = round(len(pop_movies) * n_top)
    pop_movies = pop_movies[:n]
    act_users = act_users[:n]
    intersection = get_intersection_matrix(utility, n, pop_movies, act_users)
    _, i_U, i_S, i_Vh = rsvd(intersection)
    pred_intersection = predict_empty(intersection, i_U, i_S, i_Vh)
    remap_utility = remap_intersection_to_utility(utility, pred_intersection, act_users, pop_movies)
    return remap_utility

def transform_train_test(utility_R, utility_R_test):
    padded_test = None
    if utility_R_test.shape[1] < utility_R.shape[1]:
        # Pad the test
        padded_test = np.pad(utility_R_test, ((0, utility_R.shape[0] - utility_R_test.shape[0]), (0, utility_R.shape[1] - utility_R_test.shape[1])), 'constant')
        padded_test.shape
    else:
        # Pad the train set
        padded_test = utility_R_test.copy()
        utility_R = np.pad(utility_R, ((0, utility_R_test.shape[0] - utility_R.shape[0]), (0, utility_R_test.shape[1] - utility_R.shape[1])), 'constant')
    return utility_R, padded_test
    
def test(util, test_df):
    test_util = generate_utility_matrix(test_df)
    util, test_util = transform_train_test(util, test_util)
    _, U, S, Vh = rsvd(util)
    test_pred = predict_all(test_util, U, S, Vh)
    return rmse(test_util, test_pred), test_pred
    
def train_and_test(train_df, test_df, n):
    util = fit(train_df, n)
    cal_rmse = test(util, test_df)
    return cal_rmse

def main():
    pass

if __name__ == "__main__":
    main()