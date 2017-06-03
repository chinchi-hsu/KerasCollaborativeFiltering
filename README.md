# Keras Collaborative Filtering

My Keras implementation of vanilla Matrix Factorization (MF) and Bayesian Personalized Ranking via Matrix Factorization (BPR-MF).

## Introduction

Collaborative filtering based recommender systems assume that a user's preferences over items can be predicted from other users' preferences.
Two users that have similar preferences could like or dislike the same set of items.
If one user rated (or seen) an item but the other did not, then the recommender system can push the unseen item to the other user.

For real-valued ratings, Matrix Factorization (MF) may be the most popular collaborative filtering implementation.
However, if we can access only implicit feedbacks (also named one-class collaborative filtering; that is, we can observe a user saw an item, but cannot understand the rating opinion about the item), the traditional MF fails to recommend items.
BPR framework provides an extension of MF to tackle the one-class collaborative filtering.

Keras is a well-known Python library to build deep neural networks (or multi-layer perceptron), but it could implement other machine learning models optimized by stochastic gradient descent (SGD).
I try to build MF and BPR-MF using Keras to call its well-implemented buiit-in functions like Adam and early stopping.

## Running

```
python3 matrix_factorization.py [rating_file]
python3 bayesian_personalized_ranking.py [rating_file]
```

Most of the hyperparameters can be seen in the arguments of class constructor method.

## Input File

Three values for each line in [rating_file]:

```
user_id item_id rating
```

## Output Files

* **[rating_file].user_embedding**: factorized matrix about user latent factors (i.e. embeddings).
* **[rating_file].item_embedding**: factorized matrix about item latent factors (i.e. embeddings).

## References

```
Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 42.8 (2009).
```

```
Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009.
```
