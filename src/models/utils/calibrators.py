# innovative_models/utils/calibrators.py
import numpy as np
from scipy.special import digamma, gammaln
from scipy.optimize import minimize
from scipy.stats import beta
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class BayesianCalibrator:
    """Bayesian probability calibration with Beta distribution"""

    def __init__(self, alpha_prior=1, beta_prior=1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.fitted = False

    def fit(self, y_true, y_pred):
        """Trains Beta distribution on calibrated probabilities"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        mask = (y_pred > 0) & (y_pred < 1) & (~np.isnan(y_pred)) & (~np.isnan(y_true))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 10:
            self.alpha = self.alpha_prior
            self.beta = self.beta_prior
            self.fitted = True
            return self

        if np.any(y_true == 1):
            mean = np.mean(y_pred[y_true == 1])
            var = np.var(y_pred[y_true == 1])
        else:
            mean = np.mean(y_pred)
            var = np.var(y_pred)

        if var > mean * (1 - mean):
            var = mean * (1 - mean) * 0.9

        if mean * (1 - mean) / var - 1 <= 0:
            self.alpha = self.alpha_prior
            self.beta = self.beta_prior
            self.fitted = True
            return self

        alpha0 = mean * (mean * (1 - mean) / var - 1)
        beta0 = (1 - mean) * (mean * (1 - mean) / var - 1)

        alpha0 = max(alpha0, 0.1)
        beta0 = max(beta0, 0.1)

        def neg_log_likelihood(params):
            alpha, beta = params
            alpha = max(alpha, 0.1)
            beta = max(beta, 0.1)

            ll = np.sum((alpha - 1) * np.log(y_pred + 1e-10) +
                        (beta - 1) * np.log(1 - y_pred + 1e-10) -
                        gammaln(alpha + beta) +
                        gammaln(alpha) +
                        gammaln(beta))

            prior = (self.alpha_prior - 1) * np.log(alpha) + (self.beta_prior - 1) * np.log(beta)
            return -(ll + prior)

        try:
            result = minimize(neg_log_likelihood, [alpha0, beta0],
                              bounds=[(0.1, 100), (0.1, 100)],
                              method='L-BFGS-B',
                              options={'maxiter': 100})

            self.alpha = float(result.x[0])
            self.beta = float(result.x[1])
            self.fitted = True

        except Exception as e:
            print(f"Warning: Bayesian calibration failed, using prior: {e}")
            self.alpha = self.alpha_prior
            self.beta = self.beta_prior
            self.fitted = True

        return self

    def calibrate(self, y_pred):
        """Calibrates predicted probabilities"""
        if not self.fitted:
            return y_pred

        y_pred = np.array(y_pred).flatten()

        calibrated = (y_pred * self.alpha + (1 - y_pred) * self.alpha_prior) / \
                     (y_pred * self.alpha + (1 - y_pred) * self.alpha_prior +
                      y_pred * self.beta + (1 - y_pred) * self.beta_prior)

        return np.clip(calibrated, 1e-10, 1 - 1e-10)

    def confidence_interval(self, y_pred, confidence=0.9) -> Tuple[np.ndarray, np.ndarray]:
        """Returns Bayesian credible intervals"""
        if not self.fitted:
            return y_pred - 0.1, y_pred + 0.1

        y_pred = np.array(y_pred).flatten()
        lower = []
        upper = []

        for p in y_pred:
            a = p * self.alpha + (1 - p) * self.alpha_prior
            b = p * self.beta + (1 - p) * self.beta_prior

            a = max(a, 0.1)
            b = max(b, 0.1)

            ci = beta.interval(confidence, a, b)
            lower.append(ci[0])
            upper.append(ci[1])

        return np.array(lower), np.array(upper)