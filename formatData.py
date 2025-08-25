import pandas as pd
import random as r

#User info
df = pd.read_csv("./rawData/u.user", sep="|", header=None, names=['id', 'age', 'gender', 'occupation', 'zip_code'])
df = df.drop(['zip_code'], axis=1)
df.set_index("id").to_csv("./data/userInfo.csv")

#Movie info
df = pd.read_csv("./rawData/u.item", sep="|", header=None, names=[
    "id", "title", "release date", "video release date",
    "IMDb URL", "unknown", "Action", "Adventure", "Animation",
    "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
])
df = df.drop(["IMDb URL", "video release date"], axis=1)
df.set_index("id").to_csv("./data/movieInfo.csv")

#Review info
def watchedFunc(x):
    if x >= 4:
        if r.random() <= 0.85:
            return 1
        else:
            return round(r.uniform(.7, 1), 2)
    elif x == 3:
        if r.random() <= .75:
            return 1
        else:
            return round(r.uniform(.4, .6), 2)
    elif x <= 2:
        if r.random() <= .6:
            return 1
        else:
            return round(r.uniform(0, .3), 2)

df = pd.read_csv("./rawData/u.data", sep="\t", header=None, names=["user id", "movie id", "rating", "timestamp"])
    # watched percent of movie (randomly generated)
df['watched'] = df['rating'].apply(lambda x: watchedFunc(x))
df.set_index(["user id", "movie id"]).to_csv("./data/reviewInfo.csv")
