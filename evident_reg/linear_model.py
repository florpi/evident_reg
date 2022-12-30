from typing import Optional, Tuple
import jax.numpy as np
from jax import vmap  # , jit
from jax.scipy.stats import multivariate_normal

# from functools import partial
import scipy


class GaussianLinear:
    def __init__(
        self,
        x: np.array,
        prior_linear_params: scipy.stats.multivariate_normal,
        noise_covariance: np.array,
    ):
        """Class to define a linear model with Gaussian priors on the parameters,
        and Gaussian noise.

        ..math:: y_i = \sum_j A_{ij} \theta_j + n_i

        where ..math:: A_{ij} = f_j(x_i) is a matrix of functions of the data coordinates,
        and ..math: f_j(x_i) = (j+1) x_i

        The prior on the parameters is defined by a multivariate Gaussian distribution.

        Args:
            x (np.array): input array of data coordinates
            prior_linear_params (multivariate_normal): prior distribution on the linear parameters
            noise_covariance (np.array): covariance matrix of the noise (n_i)
        """
        self.x = x
        self.prior_linear_params = prior_linear_params
        self.prior_mean = self.prior_linear_params.mean
        self.prior_cov = self.prior_linear_params.cov
        self.noise_covariance = noise_covariance
        n_params = self.prior_linear_params.dim
        self.A_matrix = vmap(self.coordinate_function, in_axes=(0, None))(
            self.x, np.arange(n_params)
        )

    def coordinate_function(self, x: np.array, j: int) -> np.array:
        """Defines function for linear model matrix: ..math:: A_{ij}=f_j(x_i)

        Args:
            x (np.array): data coordinates x
            j (int): parameter index corresponding to \theta_j

        Returns:
            np.array: f(x,j) = x*(j+1)
        """
        return (j + 1) * x

    def sample(
        self,
        weights: np.array = None,
    ) -> Tuple[float, np.array]:
        """Sample from the prior distribution of the linear parameters and
        generate an observation

        Args:
            weights (np.array, optional): Array of weights that define a linear
            transformation of the data. Defaults to None.

        Returns:
            Tuple[float, np.array]: tuple of linear parameters and observation
        """
        linear_params = np.atleast_1d(self.prior_linear_params.rvs())
        if weights is not None:
            return linear_params, weights.dot(self.A_matrix.dot(linear_params))
        return linear_params, self.A_matrix.dot(linear_params)

    # @partial(jit, static_argnums=(0,))
    def get_evidence(
        self,
        y: np.array,
        weights: Optional[np.array] = None,
    ) -> float:
        """Calculate the evidence for a given observation y, and linear transformation
        defined by weights

        Args:
            y (np.array): observation
            weights (np.array, optional): Array of weights that define a linear
            transformation of the data. Defaults to None.

        Returns:
            float: evidence value
        """
        mu_evidence = np.dot(self.A_matrix, self.prior_mean)
        cov_evidence = (
            np.dot(np.dot(self.A_matrix, self.prior_cov), self.A_matrix.T)
            + self.noise_covariance
        )
        if weights is not None:
            y = weights.dot(y)
            mu_evidence = weights.dot(mu_evidence)
            cov_evidence = weights.dot(cov_evidence).dot(weights.T)
        return multivariate_normal.pdf(y, mean=mu_evidence, cov=cov_evidence)

    # @partial(jit, static_argnums=(0,))
    def get_posterior(
        self,
        linear_params: np.array,
        y: np.array,
        weights: Optional[np.array] = None,
    ):
        """Get the posterior distribution for the linear parameters given an observation,
        evaluated at linear_params

        Args:
            linear_params (np.array): where to evaluate the posterior
            y (np.array): observation
            weights (np.array, optional): Array of weights that define a linear
            transformation of the data. Defaults to None.

        Returns:
            posterior: evaluated at linear_params
        """
        inverse_cov_prior = np.linalg.inv(self.prior_cov)
        if weights is not None:
            noise_covariance = weights.dot(self.noise_covariance).dot(weights.T)
            A_matrix = weights.dot(self.A_matrix)
            inverse_noise_covariance = np.linalg.inv(noise_covariance)
            y = weights.dot(y)
        else:
            A_matrix = self.A_matrix
            inverse_noise_covariance = np.linalg.inv(self.noise_covariance)
        inverse_cov_posterior = (
            A_matrix.T.dot(inverse_noise_covariance.dot(A_matrix)) + inverse_cov_prior
        )
        cov_posterior = np.linalg.inv(inverse_cov_posterior)
        mean_posterior = cov_posterior.dot(
            A_matrix.T.dot(inverse_noise_covariance).dot(y)
        ) + cov_posterior.dot(inverse_cov_prior.dot(self.prior_mean))
        return multivariate_normal.pdf(
            linear_params,
            mean=mean_posterior,
            cov=cov_posterior,
        )

    # @partial(jit, static_argnums=(0,))
    def get_fisher_info(
        self,
        weights: Optional[np.array] = None,
    ) -> float:
        """Get the Fisher information for the linear parameters

        Args:
            weights (np.array, optional): Array of weights that define a linear
            transformation of the data. Defaults to None.

        Returns:
            float: fisher information
        """
        if weights is None:
            inverse_noise_covariance = np.linalg.inv(self.noise_covariance)
            A_matrix = self.A_matrix
        else:
            inverse_noise_covariance = np.linalg.inv(
                weights.dot(self.noise_covariance).dot(weights.T)
            )
            A_matrix = weights.dot(self.A_matrix)
        inverse_prior_cov = np.linalg.inv(self.prior_cov)
        return np.linalg.det(
            np.dot(A_matrix.T, np.dot(inverse_noise_covariance, A_matrix))
            + inverse_prior_cov
        )

    # @partial(jit, static_argnums=(0,))
    def get_evidence_ratio_from_samples(
        self,
        y_fiducial: np.array,
        y_samples: np.array,
        weights: Optional[np.array] = None,
    ) -> float:
        """Get ratio of evidences for an observation respect to the average of a set of samples

        Args:
            y_fiducial (np.array): observation we want to compare to the average.
            y_samples (np.array): samples from the prior used to calculate the average.
            weights (np.array, optional): Array of weights that define a linear
            transoformation of the data. Defaults to None.

        Returns:
            float: evidence ratio
        """
        evidence_fiducial = self.get_evidence(
            y=y_fiducial,
            weights=weights,
        )
        average_evidence = 0
        for sample in y_samples:
            average_evidence += self.get_evidence(
                y=sample,
                weights=weights,
            )
        average_evidence /= len(y_samples)
        return np.log(evidence_fiducial / average_evidence)

    # @partial(jit, static_argnums=(0,))
    def get_evidence_ratio(
        self,
        y: np.array,
        weights: np.array = None,
    ) -> float:
        """Get ratio of evidences for an observation respect to the average evidence.
        Computed analtyically.

        Args:
            y (np.array): observation we want to compare to the average.
            weights (np.array, optional): Array of weights that define a linear
            transformation of the data. Defaults to None.

        Returns:
            float: evidence ratio
        """
        if weights is not None:
            chi_sq = weights.dot(y - self.A_matrix.dot(self.prior_mean))
            cov_term = weights.dot(
                self.noise_covariance
                + self.A_matrix.dot(self.prior_cov).dot(self.A_matrix.T)
            ).dot(weights.T)
        else:
            chi_sq = y - self.A_matrix.dot(self.prior_mean)
            cov_term = self.noise_covariance + self.A_matrix.dot(self.prior_cov).dot(
                self.A_matrix.T
            )
        inv_cov_term = np.linalg.inv(cov_term)
        return 0.5 * (
            chi_sq.T.dot(inv_cov_term.dot(chi_sq)) - np.linalg.matrix_rank(weights)
        )

    def get_loss(
        self,
        y_fiducial: np.array,
        y_samples: np.array,
        lambda_ev: float = 1.0,
        weights: Optional[np.array] = None,
    ) -> float:
        """Loss function that combines the evidence ratio and the Fisher information
        where the evidence ratio is weighted by lambda_ev

        Args:
            y_fiducial (np.array): biased observation
            y_samples (np.array): samples of unbiased observations, used to estimate the
            average evidence
            lambda_ev (float, optional): weight given to the evidence. Defaults to 1..
            weights (np.array, optional): Array of weights that define a linear
            transformation of the data. Defaults to None.

        Returns:
            float: loss
        """
        return lambda_ev * self.get_evidence_ratio_from_samples(
            y_fiducial=y_fiducial, y_samples=y_samples, weights=weights
        ) + self.get_fisher_info(weights=weights)
