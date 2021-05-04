"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from numpy import newaxis


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    N=np.divide(np.exp(np.divide(-np.sum(np.square(np.abs(X[newaxis,:,:] - mixture.mu[:,newaxis,:])),axis=2),(2*mixture.var[:,newaxis]))),np.power(2*np.pi*mixture.var[:,newaxis],X.shape[1]/2))
    post = np.divide(np.multiply(mixture.p[:,newaxis],N),np.sum(np.multiply(mixture.p[:,newaxis],N),axis=0)).transpose()
    LL = np.sum(np.log(np.sum(np.multiply(mixture.p[:,newaxis],N),axis=0),dtype=np.float64))
    return post, LL
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    mu = np.divide(np.matmul(post.transpose(),X),np.sum(post,axis=0)[:,newaxis])
    var = np.diagonal(np.divide(np.matmul(np.sum(np.square(np.abs(X[newaxis,:,:] - mu[:,newaxis,:])),axis=2),post),X.shape[1]*np.sum(post,axis=0)[:,newaxis]))
    p = np.divide(np.sum(post,axis=0),X.shape[0])
    return GaussianMixture(mu,var,p)
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_LL = None
    LL = None
    while(prev_LL is None or LL - prev_LL >= 1e-6*np.abs(LL)):
      prev_LL = LL
      post, LL = estep(X, mixture)
      mixture = mstep(X, post)

    return mixture, post, LL
    raise NotImplementedError
