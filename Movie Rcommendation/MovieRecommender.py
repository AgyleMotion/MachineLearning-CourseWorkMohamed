# %% [markdown]
# # Movie Recommendations HW

# %% [markdown]
# **Name:**  

# %% [markdown]
# **Collaboration Policy:** Homeworks will be done individually: each student must hand in their own answers. Use of partial or entire solutions obtained from others or online is strictly prohibited.

# %% [markdown]
# **Late Policy:** Late submission have a penalty of 2\% for each passing hour. 

# %% [markdown]
# **Submission format:** Successfully complete the Movie Lens recommender as described in this jupyter notebook. Submit a `.py` and an `.ipynb` file for this notebook. You can go to `File -> Download as ->` to download a .py version of the notebook. 
# 
# **Only submit one `.ipynb` file and one `.py` file.** The `.ipynb` file should have answers to all the questions. Do *not* zip any files for submission. 

# %% [markdown]
# **Download the dataset from here:** https://grouplens.org/datasets/movielens/1m/

# %%
# Import all the required libraries
import numpy as np
import pandas as pd

# %% [markdown]
# ## Reading the Data
# Now that we have downloaded the files from the link above and placed them in the same directory as this Jupyter Notebook, we can load each of the tables of data as a CSV into Pandas. Execute the following, provided code.

# %%
# Read the dataset from the two files into ratings_data and movies_data
column_list_ratings = ["UserID", "MovieID", "Ratings","Timestamp"]
ratings_data  = pd.read_csv('ratings.dat',sep='::',names = column_list_ratings, engine='python')
column_list_movies = ["MovieID","Title","Genres"]
movies_data = pd.read_csv('movies.dat',sep = '::',names = column_list_movies, engine='python',encoding='ISO-8859-1')
column_list_users = ["UserID","Gender","Age","Occupation","Zixp-code"]
user_data = pd.read_csv("users.dat",sep = "::",names = column_list_users, engine='python')

# %%
### use numpy to create a ratings data matrix
nr_users = np.max(ratings_data.UserID.values)
nr_movies = np.max(ratings_data.MovieID.values)
ratings_matrix = np.ndarray(shape=(nr_users, nr_movies),dtype=np.uint8)

# %%
ratings_matrix[ratings_data.UserID.values - 1, ratings_data.MovieID.values - 1] = ratings_data.Ratings.values

# %%
# Print the shape
print("sample of 10x10 of rating matrix:\n", ratings_matrix[:10,:10])
print("shape of ratings_matrix:",ratings_matrix.shape)

# %% [markdown]
# ## Question 2

# %% [markdown]
# Normalize the ratings matrix (created in **Question 1**) using Z-score normalization. While we can't use `sklearn`'s `StandardScaler` for this step, we can do the statistical calculations ourselves to normalize the data.
# 
# Before you start:
# - All of the `NaN` values in the dataset should be replaced with the average rating for the given movie. This is a complex topic, but for our case replacing empty values with the mean will make it so that the absence of a rating doesn't affect the overall average, and it provides an "expected value" which is useful for computing correlations and recommendations in later steps. 
# - Your first step should be to get the average of every *column* of the ratings matrix (we want an average by title, not just by user!).
# - Second, we want to subtract the average from the original ratings thus allowing us to get a mean of 0 in every column. It may be very close but not exactly zero because of the limited precision `float`s allow.

# %%
ratings_matrix = (ratings_matrix - ratings_matrix.mean(axis = 0))/ratings_matrix.std(axis = 0) #normalize the data

# %%
ratings_matrix[np.isnan(ratings_matrix)] = 0 #replace nan values with 0.

# %%
print("sample of 10x10 of Normalized rating matrix:\n", ratings_matrix[:10,:10])
print("shape of Normalized ratings_matrix:",ratings_matrix.shape)

# %% [markdown]
# SVD COMPUTATION OF NORMALIZED MATRIX

# %%
U, S, vh = np.linalg.svd(ratings_matrix, full_matrices=False) # vh is VT

# %%

print("shape of U:",U.shape)
print("shape of S:",S.shape)
print("shape of vT:",vh.shape)


# %% [markdown]
# Verification of SVD computation

# %%
S_diagonal=np.diag(S) #DIAGONALIZATION
print("shape of S diagonal: \n",S_diagonal.shape)
print ()
c = np.matmul(U,S_diagonal )
ver=np.matmul(c,vh)
print("ver:",ver [:5,:5])
print("ratings_matrix:\n",ratings_matrix[:5,:5])

# %% [markdown]
# Slicing with different K

# %%
k1=3
U1=U[:,:k1]
S1=S_diagonal[:k1,:k1]
vh1=vh.T[:,:k1]
print(" U1:\n",U1)
print(" s1:\n",S1)
print(" vh1:\n",vh1)

print("shape of U1:",U1.shape)
print("shape of S1:",S1.shape)
print("shape of vh1:",vh1.shape)

step1 = np.matmul(U1,S1 )
R1=np.matmul(step1,vh1.T)
print("shape of R1:",R1.shape)
print("user rating ", R1[0,1376])

k2=1000
U2=U[:,:k2]
S2=S_diagonal[:k2,:k2]
vh2=vh[:k2,:]
step2 = np.matmul(U2,S2 )
R2=np.matmul(step2,vh2)
print("Shape of R2:",R2.shape)
print("User rating ", R2[50,1377])

k3=2000
U3=U[:,:k3]
S3=S_diagonal[:k3,:k3]
vh3=vh[:k3,:]
step3 = np.matmul(U3,S3 )
R3=np.matmul(step3,vh3)
print("Shape of R3:",R3.shape)
print("User rating ", R3[0,1376])

k4=3000
U4=U[:,:k4]
S4=S_diagonal[:k4,:k4]
vh4=vh[:k4,:]
step4 = np.matmul(U4,S4 )
R4=np.matmul(step4,vh4)
print("User rating ", R4[0,1376])










# %%
def top_cosine_similarity(ratings_data , MovieID, top_n=10):
    index = MovieID - 1 # Movie id starts from 1
    movie_row = ratings_data [index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', ratings_data , ratings_data ))
    similarity = np.dot(movie_row, ratings_data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Helper function to print top N similar movies
def print_similar_movies(movies_data, MovieID, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movies_data[movies_data.MovieID == MovieID].Title.values[0]))
    for id in top_indexes + 1:
        print(movies_data[movies_data.MovieID == id].Title.values[0])

# %%
k = 1000
movie_id = 1377 
top_n = 5
sliced = vh.T[:, :k]  
indexes = top_cosine_similarity(sliced, movie_id, top_n)
print_similar_movies(movies_data, movie_id, indexes)


