# -*- coding: utf-8 -*-
"""
Conformal Prediction for Regression
Provides 80% coverage prediction intervals with adaptive calibration
"""

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator


class ConformalRegressor(BaseEstimator):
    """
    Conformal prediction wrapper for regression models.
    Provides prediction intervals with guaranteed coverage.
    
    Uses non-conformity scores (absolute residuals) to calibrate intervals.
    """
    
    def __init__(self, base_model, confidence_level=0.8, alpha_buffer=0.05):
        """
        Parameters
        ----------
        base_model : sklearn regressor
            The underlying regression model to calibrate
        confidence_level : float
            Target coverage level (0.0-1.0). Default 0.8 for 80% coverage.
        alpha_buffer : float
            Small buffer for adaptive alpha adjustment. Default 0.05.
        """
        self.base_model = base_model
        self.confidence_level = confidence_level
        self.alpha_buffer = alpha_buffer
        
        # Calibration data
        self.calibration_residuals = None
        self.quantile_threshold = None
        self.is_fitted = False
    
    def __setstate__(self, state):
        """
        Handle deserialization from joblib/pickle.
        Ensures backward compatibility with models saved before is_fitted was added.
        """
        self.__dict__.update(state)
        # Add missing attributes for old saved models
        if not hasattr(self, 'is_fitted'):
            self.is_fitted = False
        if not hasattr(self, 'calibration_residuals'):
            self.calibration_residuals = None
        if not hasattr(self, 'quantile_threshold'):
            self.quantile_threshold = None
        if not hasattr(self, 'alpha_buffer'):
            self.alpha_buffer = 0.05
        
    def fit(self, X_train, y_train, X_calibrate=None, y_calibrate=None, calibration_split=0.2, min_calibration_size=50):
        """
        Fit the conformal model.
        
        Parameters
        ----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_calibrate : array-like, optional
            Calibration features (if None, taken from X_train with calibration_split)
        y_calibrate : array-like, optional
            Calibration targets (if None, taken from y_train with calibration_split)
        calibration_split : float
            Fraction of training data to use for calibration if X_calibrate not provided
        min_calibration_size : int
            Minimum number of calibration samples. If calibration set is smaller,
            it will be expanded automatically.
            
        Returns
        -------
        self
        """
        # If calibration data not provided, split from training
        if X_calibrate is None or y_calibrate is None:
            n_train = int(len(X_train) * (1 - calibration_split))
            X_train_subset = X_train.iloc[:n_train] if hasattr(X_train, 'iloc') else X_train[:n_train]
            y_train_subset = y_train.iloc[:n_train] if hasattr(y_train, 'iloc') else y_train[:n_train]
            X_calibrate = X_train.iloc[n_train:] if hasattr(X_train, 'iloc') else X_train[n_train:]
            y_calibrate = y_train.iloc[n_train:] if hasattr(y_train, 'iloc') else y_train[n_train:]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        # Warn if calibration size is too small
        n_cal_actual = len(X_calibrate) if hasattr(X_calibrate, '__len__') else len(y_calibrate)
        if n_cal_actual < min_calibration_size:
            import warnings
            warnings.warn(
                f"Calibration size ({n_cal_actual}) is less than recommended minimum ({min_calibration_size}). "
                f"This may lead to unreliable coverage guarantees."
            )
        
        # Fit the base model on training data
        self.base_model.fit(X_train_subset, y_train_subset)
        
        # Compute non-conformity scores on calibration set (absolute residuals)
        y_pred_calibrate = self.base_model.predict(X_calibrate)
        self.calibration_residuals = np.abs(y_calibrate - y_pred_calibrate)
        
        # Compute quantile threshold for the confidence level
        # Using Barber et al. (2021) distribution-free quantile approach
        n_cal = len(self.calibration_residuals)
        alpha = 1 - self.confidence_level
        
        # The formula ceil((n+1)*(1-alpha))/n gives a lower confidence bound
        # that holds regardless of the underlying distribution
        quantile_index = np.ceil((n_cal + 1) * (1 - alpha))
        quantile_level = quantile_index / n_cal
        
        # Ensure we're within valid quantile range
        quantile_level = np.clip(quantile_level, 0.0, 1.0)
        
        # Use the quantile to set threshold on calibration residuals
        # This guarantees coverage >= confidence_level on average
        self.quantile_threshold = np.quantile(
            self.calibration_residuals, 
            quantile_level,
            method='linear'
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make point predictions.
        
        Parameters
        ----------
        X : array-like
            Features to predict
            
        Returns
        -------
        predictions : array
            Point predictions
        """
        if not getattr(self, 'is_fitted', False):
            raise ValueError("Model must be fitted before prediction")
        return self.base_model.predict(X)
    
    def predict_with_intervals(self, X, confidence=None, min_bound=0.0):
        """
        Make predictions with confidence intervals.
        
        Parameters
        ----------
        X : array-like
            Features to predict
        confidence : float, optional
            Override confidence level. If None, uses fitted confidence_level.
        min_bound : float, optional
            Minimum allowed value for lower bound (default: 0.0 for non-negative targets)
            Set to None to disable floor constraint
            
        Returns
        -------
        predictions : tuple
            (point_predictions, lower_bounds, upper_bounds)
        """
        if not getattr(self, 'is_fitted', False):
            raise ValueError("Model must be fitted before prediction")
        
        # Get point predictions
        y_pred = self.base_model.predict(X)
        
        # Use threshold from calibration
        threshold = self.quantile_threshold
        
        # Create intervals
        lower_bounds = y_pred - threshold
        upper_bounds = y_pred + threshold
        
        # Apply floor constraint if specified (e.g., revenue cannot be negative)
        if min_bound is not None:
            lower_bounds = np.maximum(lower_bounds, min_bound)
            # Ensure upper bound is at least equal to lower bound
            upper_bounds = np.maximum(upper_bounds, lower_bounds)
        
        return y_pred, lower_bounds, upper_bounds
    
    def get_interval_width(self):
        """Get the current interval width (2 * threshold)."""
        if not getattr(self, 'is_fitted', False):
            return None
        return 2 * self.quantile_threshold
    
    def evaluate_coverage(self, X, y):
        """
        Evaluate coverage on test data.
        
        Parameters
        ----------
        X : array-like
            Test features
        y : array-like
            Test targets
            
        Returns
        -------
        metrics : dict
            Contains 'coverage', 'mae', 'interval_width', 'efficiency'
        """
        y_pred, lower, upper = self.predict_with_intervals(X)
        
        coverage = np.mean((y >= lower) & (y <= upper))
        mae = mean_absolute_error(y, y_pred)
        interval_width = np.mean(upper - lower)
        
        # Efficiency: how tight are intervals relative to target coverage?
        # Ideally ratio close to 1.0
        efficiency = self.confidence_level / max(coverage, 0.001)
        
        return {
            'coverage': coverage,
            'mae': mae,
            'interval_width': interval_width,
            'efficiency': efficiency,
            'target_coverage': self.confidence_level
        }
