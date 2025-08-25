# MovieLens 100k Recommendation model - overview
*(formatData.py)*

At the beginning, it was needed to convert `.csv` files into dataframes. Before that I checked if there are any not available data what I did with pycharm csv filters.
The result was only 1 damaged record, then it wasn't necessary to write script deleting such things and I manually remove it.
During conversion, I implemented into reviews fields `watched` and `weighted rating`. First one determines percent of movie, that reviewer watched, randomly generated with some constraints. The second is result of `watched * rating` which purpose is favourize reviews with higher `watched` factor.

*(model.py)*

This file is to merge dataframes, from the previous file, and fitting with it neural network along with building recommendation model. A first try was with MLP architecture, it wasn't working as expected because it mixes users and movies embeddings what's not good solution for recommendations. A response for it is Two Tower architecture due to its processing initially in seperated dense layers what improves reasoning of semantic relations between elements. Also I deployed into it records simulations (LightFm)

*(test_model.py)*

Unit tests for `model.py`. Example output:
```
Testing data loading...
Data loaded correctly!

Testing encoders...
Encoders working correctly!

Testing model building...
/home/krystian/miniconda3/envs/recom-proj/lib/python3.10/site-packages/keras/src/engine/functional.py:642: UserWarning: Input dict contained keys ['weighted_rating'] which did not match any model input. They will be ignored by the model.
  inputs = self._flatten_to_reference_inputs(inputs)
Model built successfully!

Testing recommendations...
/home/krystian/miniconda3/envs/recom-proj/lib/python3.10/site-packages/sklearn/utils/validation.py:2749: UserWarning: X does not have valid feature names, but MinMaxScaler was fitted with feature names
  warnings.warn(
Recommendations working correctly!

Testing simulated records...
Simulated records included correctly!

Testing unknown user handling...
User with ID 999999 was not found.
Unknown user handling works!



Ran 6 tests in 110.627s

OK
```

*(hyperparmeters_tests.ipynb)*

Jupyter notebook with hyperparmeters tests for recommendation model with final result.

***Example result of recommendation***
```
Recommendation for random user with id: 349

/home/krystian/miniconda3/envs/recom-proj/lib/python3.10/site-packages/sklearn/utils/validation.py:2749: UserWarning: X does not have valid feature names, but MinMaxScaler was fitted with feature names
  warnings.warn(
      movie_id                       title  predicted_rating
482        484  Maltese Falcon, The (1941)         -3.688107
609        611                Laura (1944)         -3.720114
523        525       Big Sleep, The (1946)         -3.987827
1202      1204   To Be or Not to Be (1942)         -4.603813
1473      1476             Raw Deal (1948)         -4.658393
481        483           Casablanca (1942)         -4.985613
392        394    Radioland Murders (1994)         -5.067250
491        493        Thin Man, The (1934)         -5.108849
639        641       Paths of Glory (1957)         -5.176889
601        603          Rear Window (1954)         -5.216152
```

# How to run
Make sure you use linux distro and have python 3.10.

1. Clone repository
```
git clone https://github.com/DarkLightOfFuture/MovieLens-100k---Recommendation-model.git
```
2. Install required packages
```
pip install -r ./requirements.txt
```

