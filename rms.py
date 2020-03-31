
#  import  packages
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('./TrainingData.csv', encoding='utf-8', sep=',')
ratings = ratings.fillna(0)


def standardize(row):
    new_row = (row - row.mean())/(row.max() - row.min())
    return new_row


df_std = ratings.apply(standardize)
corrMatrix = pd.DataFrame(cosine_similarity(sparse.csr_matrix(df_std.values)))
corrMatrix = ratings.corr(method='pearson')


def get_similar(user_id):
    similar_score = corrMatrix[user_id]
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score


def get_n_nearest_neighbors(user_id, n):
    similar_scores = pd.DataFrame()
    similar_scores = similar_scores.append(
        get_similar(user_id), ignore_index=True)
    neighborhood = similar_scores.sum().sort_values(ascending=False)

    arr = []
    for x in range(1, n+1):
        arr.insert(x, neighborhood.index[x])
    return arr


print(get_n_nearest_neighbors("user1", 2))
