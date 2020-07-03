#!/usr/bin/env python
# coding: utf-8

# # Research on Algorithms and Notes
# 
# Reference: 
# The algorithm implemented in this project follows the direction of CodeHeroku Intro ML video series by Mihir Thakkar, Kaggle Movie Recommender article by Rounak Banik, and several other youtube videos. Dataset comes from Kaggle.com 
# (Note: I orginally used a really big dataset from MovieLens, but I realized that it was too big and my code runs super slowly, therefore, I switched to a relatively smaller dataset.)
# 
# ## Recommendation System General Notes 
# There are three types of movie recommendation systems: Rating-based, Content-based, and Collaborative-Based systems. 
# ### 1. Rating-based 
# Rating-based filter focuses on the similarity of ratings and popularities.The system usually keeps track of the click count (and/or ratings of movie) so that the recommendation system always recommends the most trending movies, possibly from the recent most popular to the least popular movies. 
# ### 2. Content-based
# Content-based filter focuses on the similarity of contents, take the movies that user currently likes as inputs and analyze several features such as genre, cast, directors of the movie to find out what other movies might have similar contents and make a recommendation to users. 
# ### 3. Collaborative-Based
# This type of system take similar users as inputs based on people's similar preferences.	If several users are classified as being similar to each other — watch movies directed by the same director or movie of the same genre — then the system cross check if one user has not watched movies that several other "similar" users have watch, and recommends that movie to this particular user.
# 
# #### Final Work done:
# 1. First, I implemented a content-based system: 
#     Features accounted for are: 
#     * `director`
#     * `genre`
#     * `cast`
#     * `keywords`
# 2. To improve the content-based recommendation system, I added the `popularity rating` as a new layer in my sorting process, allowing the model to give the final recommendation based on popularity
# 3. Implemented a new feature to recommend the top rated movies by `genre`
# 4. Made a UI mock-up hosted on figma: https://www.figma.com/proto/LuHyG0h47FUhRasBUEhpN4/HAI?node-id=1%3A210&scaling=scale-down

# In[80]:


# Package Import 
# general packages
import pandas as pd  
import numpy as np  


# In[81]:


# read dataset
df = pd.read_csv("movie_dataset.csv");
df.head()


# In[82]:


for col in df.columns: 
    print(col) 


# ## Feature 1: Content-based Filtering
# ### 1. Define Similarity score    
# Similarity score is defined by the cosine similarity, that is, if a certain word occurred in movie 1 once and in movie 2 twice, we can map the occurrance of that word on a xy coordinate system and calculate their distance by cosine theta (through u.v/len(u).len(v)). The advantage 
# of such a method is that the final score will be mapped in range of 0 to 1 range.
# ### 2. Calculate Cosine Similarity
# The sklearn package will be able to calculate the cosine similarity score for us via `sklearn.metrics.pairwise` package. We would also need a package to parse the strings from the texts and the `sklearn.feature_extraction.text` package would help us do that.

# In[83]:


# string parser and cosine score calculator
from sklearn.feature_extraction.text import CountVectorizer  #parse text
from sklearn.metrics.pairwise import cosine_similarity  # cosine_similarity package, for content-based similarity modeling


# ## Feature Selection
# First, I need to select what type of features I want to use in the prediction. From the df.head() we could see the content of each column and `genres`, `director`, `keywords`, and `cast` might be good fits for the purpose of parsing similar texts. The overview might be useful as well, but we might end-up getting a bunch of propositions, which could become a distraction to the algorithm. After some research, I decide not to use this feature for now. 

# In[84]:


# select features I want to use
features = ['keywords','cast','genres','director']


# In[85]:


# combine all selected features
for feature in features:
    df[feature] = df[feature].fillna('') # fill all NA cells

# produce a column which has all features combined for each movie
def combine_features(movie):
    try:
        return movie[features[0]]+" "+movie[features[1]]+" "+movie[features[2]]+" "+movie[features[3]]
    except:
        print("Error:", movie)

df["combine_features"] = df.apply(combine_features,axis=1)

print("Features:\n", df["combine_features"].head())


# In[86]:


# Create count matrix from this new column
cv = CountVectorizer() # from sklearn.feature_extraction.text 
count_matrix = cv.fit_transform(df["combine_features"])


# In[87]:


# Compute the Cosine Similarity based on the count_matrix
cosine = cosine_similarity(count_matrix) # from sklearn.metrics.pairwise


# In[88]:


# helper
# given an index, get the title of selected movie
def get_title(index):
    return df[df.index == index]["title"].values[0]

# given a title, get the index of selected movie
def get_index(title):
    return df[df.title == title]["index"].values[0]


# In[89]:


# test with the dark knight... 
# hypothetically, it will recommend things like batman, dark knight rises, etc. 
user_fav = "The Dark Knight"


# In[90]:


# Get index from title
idx = get_index(user_fav)
similar = list(enumerate(cosine[idx]))


# In[91]:


# Get a list of similar movies in descending order of similarity score
sorted_similar = sorted(similar,key=lambda x:x[1],reverse=True)[1:20]


# In[93]:


# Print titles of first 10 movies
i=0
print("top 10 movies we think you would like!")
for element in sorted_similar:
    print(get_title(element[0]))
    i=i+1
    if i>9:
        break


# #### Feature 1.5
# I'm going to add a new layer — voting— to the sorting.

# In[94]:


# take voting into account
df["vote_average"].head()


# In[95]:


sort_by_average_voting = sorted(sorted_similar,key=lambda x:df["vote_average"][x[0]],reverse=True)
print(sort_by_average_voting)


# In[96]:


i=0
print("We think you would like to watch the following movies:\n")
for element in sort_by_average_voting:
    print(get_title(element[0]))
    i=i+1
    if i>9:
        break


# In[97]:


# function to print top recommended movies
def recommend_by_title(title):
    idx = get_index(title)
    similar = list(enumerate(cosine[idx]))
    sorted_similar = sorted(similar,key=lambda x:x[1],reverse=True)[1:20]
    sort_by_average_voting = sorted(sorted_similar,key=lambda x:df["vote_average"][x[0]],reverse=True)
    i=0
    print("We think you would like to watch the following movies:\n")
    for element in sort_by_average_voting:
        print(get_title(element[0]))
        i=i+1
        if i>9:
            break    


# In[98]:


recommend_by_title('The Dark Knight')


# ### Performance
# I cross checked my algorithm's output with google: the results are somewhat similar. I saw the three batman series in both my list and the google recommendation; and I saw the dark knight rises. 

# ![darkknight1.png](attachment:darkknight1.png)

# ## Feature 2: Top-Rated Movies by genre

# I added this feature which could recommend top rated movies by genre. The high-level idea is adopted from the IMDB's weighted rating formulae: 
# 
# weighted rating = (#vote / (#votes+min_vote)* ave_rating) + (min_vote / (#votes+min_vote)* ave_report)
# 
# where #vote is the number of vote; min_vote is the minimum vote on the board, ave_rating is the average rating of the movie, and ave_report stands for the mean vote across the board. 
# 
# To get enough movies in the pool, I set the top 75% percentile (i.e. a movie to be put on board should have at least higher votes than 75% of the movies across the entire dataset).

# In[99]:


from ast import literal_eval


# In[100]:


df['vote_average'].head()


# In[101]:


df['vote_count'].head()


# In[102]:


# clean vote_count column
md = df[~df['vote_count'].astype(str).str.contains('id',na=False)]
md = md[~md['vote_count'].astype(str).str.contains('name',na=False)]


# In[103]:


# clean vote_average column
md = md[~md['vote_average'].astype(str).str.contains('id',na=False)]
md = md[~md['vote_average'].astype(str).str.contains('name',na=False)]


# In[104]:


# convert to int
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype(str).astype(int)
    


# In[105]:


# convert score to float
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype(str).astype(float)


# In[106]:


# calculate mean across the board 
C = vote_averages.mean()
C


# In[107]:


# calculate 0.75 quantile
m = vote_counts.quantile(0.75)
m


# In[108]:


# calculate weighted ratings
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[109]:


qualified = md[(md['vote_count'] >= m) & 
                   (md['vote_count'].notnull()) & 
                   (md['vote_average'].notnull())][['title', 
                                                    'vote_count', 
                                                    'vote_average', 
                                                    'popularity']]


# In[110]:


qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)


# In[111]:


qualified = qualified.sort_values('weighted_rating', ascending=False).head(250)


# In[112]:


qualified.head(15)


# In[113]:


# filter by genre
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = df.drop('genres', axis=1).join(s)


# In[114]:


# test
gen_md[gen_md['genre'].astype(str).str.contains('Action')].head()


# In[115]:


def build_rec(genre, percentile=0.75):
    md = gen_md[gen_md['genre'].astype(str).str.contains(genre)]
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('float')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = md[(md['vote_count'] >= m) & 
                   (md['vote_count'].notnull()) & 
                   (md['vote_average'].notnull())][['title', 
                                                    'vote_count', 
                                                    'vote_average', 
                                                    'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('float')
    
    qualified['weighted_rating'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('weighted_rating', ascending=False).head(250)
    
    return qualified


# In[116]:


# for example, I want to search for top crime movies
print('We think you would love these Crime movies: \n')
for index, row in build_rec('Crime').head(10).iterrows():
    print(row['title'])


# In[ ]:




