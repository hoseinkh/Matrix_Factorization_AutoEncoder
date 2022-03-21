# Matrix Factorization using Auto-Encoder

Here we implement Matrix Factorization using embedding layers.

<br />

## Task:

The goal is to derive latent representation of the user and item feature vectors. The (predicted) ratings that a user gives to an item is the inner product of user's latent vector and the item's latent vector.

The idea here is to use **autoencoder** to get the latent representations.

We use the 20 million MovieLens data set available on [Kaggle](https://www.kaggle.com/grouplens/movielens-20m-dataset). Though, for practical implementation on a pc we shrink this dataset.

---

The following figure shows how auto-encoder wroks:

<p float="left">
  <img src="/figs/MF_autoencoder_form.png" width="450" />
</p>




---

### Codes & Results

The code consist of two parts. One is for the data preprocessing, and one implements and matrix factorization using auto-encoder, and gets the results.

<p float="left">
  <img src="/figs/MF_autoencoder_mse_error.png" width="450" />
</p>

and if we add regularization to the cost function, we get the following results

<p float="left">
  <img src="/figs/MF_autoencoder_mse_and_regul_loss.png" width="450" />
</p>





------

### References

1. [Recommender Systems Handbook; Ricci, Rokach, Shapira](https://www.cse.iitk.ac.in/users/nsrivast/HCC/Recommender_systems_handbook.pdf)
2. [Statistical Methods for Recommender Systems; Agarwal, Chen](https://www.cambridge.org/core/books/statistical-methods-for-recommender-systems/0051A5BA0721C2C6385B2891D219ECD4)

