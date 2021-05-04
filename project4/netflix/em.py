"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from numpy import newaxis


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    delta = np.zeros(X.shape)
    delta[X!=0] = 1

    N=np.divide(np.exp(np.divide(-np.sum(np.square(np.abs(np.multiply(X[newaxis,:,:] - mixture.mu[:,newaxis,:],delta))),axis=2),
                      (2*mixture.var[:,newaxis])))[:,:,newaxis],
                np.power(2*np.pi*mixture.var[:,newaxis,newaxis],np.count_nonzero(X,axis=1)[:,newaxis]/2))
    f_ui = np.log(mixture.p[:,newaxis,newaxis] + 1e-16) + np.log(N)
    logsumexp_f_ui = np.amax(f_ui,axis=0) + logsumexp(f_ui - np.amax(f_ui,axis=0),axis=0)
    logpost = f_ui - logsumexp_f_ui
    post = np.squeeze(np.exp(logpost),axis=2).transpose()
    LL = np.sum(logsumexp_f_ui)
    return post, LL
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    delta = np.zeros(X.shape)
    delta[X!=0] = 1
    
    mu = np.divide(np.matmul(post.transpose(),np.multiply(delta,X)),np.matmul(post.transpose(),delta),out=mixture.mu,where = np.greater_equal(np.matmul(post.transpose(),delta),np.full(mixture.mu.shape,1)))
    var = np.clip(np.diagonal(np.divide(np.matmul(np.sum(np.square(np.abs(np.multiply(X[newaxis,:,:] - mu[:,newaxis,:],delta))),axis=2),post),
                                        np.sum(np.multiply(post,np.count_nonzero(X,axis=1)[:,newaxis]),axis=0)[:,newaxis])),
                  min_variance,min_variance+np.max(mixture.var))
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
      mixture = mstep(X, post, mixture)

    return mixture, post, LL
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
