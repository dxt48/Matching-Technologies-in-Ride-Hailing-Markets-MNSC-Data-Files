"""
Theory.py
===============================================================================
Analytical Wait-Time Approximations for Street-Hailing and Central Dispatch
-------------------------------------------------------------------------------
Manuscript Reference:
    "Matching Technology, Fleet Size, and Passenger Wait in Ride-Hailing Markets"
    - (Main_Document): paper_minor_2.tex, Section 4 ("Matching Technology Approximations")
        * Approximation 1 (Section 4.1, Equations (1a)-(1b)): Street-Hailing CDF & Mean Wait
        * Approximation 2 (Section 4.2, Equations (2a)-(2b)): Central Dispatch CDF & Mean Wait

Purpose:
    Provides closed-form probability density functions (PDFs), cumulative
    distribution functions (CDFs), and expected waiting time formulas for both
    Street-Hailing (subscript 's') and Central Dispatch (subscript 'd').
    This module is imported by `Lem_4_Prop_3.py` to calibrate fleet sizes and
    generate Figure 6 (`Figures/wait_time_comparison.png`).
===============================================================================
"""

import logging
import math
from math import pi
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import probplot, rayleigh
import statsmodels.api as sm


# Compatibility shim for display() when running outside an IPython/Jupyter kernel
try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs):
        for arg in args:
            print(arg)

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# =============================================================================
# MODEL PRIMITIVES AND CALIBRATED CONSTANTS
# (See Section 5.1 and Table 1 in (Main_Document))
# =============================================================================
CITY_SIDE_MILES = 4.0                           # Side length of the 16-sq-mile Manhattan grid
NUM_BLOCKS = 80                                 # Grid dimensions: 80 x 80 = 6,400 corners (J)
SPEED_MPH = 10                                  # Baseline travel speed (mph)
TRIP_LENGTH_MILES = 2                           # Average passenger trip distance (miles)
BLOCK_LENGTH = CITY_SIDE_MILES / NUM_BLOCKS     # Length of one city block (0.05 miles)
BASE_ARRIVALS_PER_MIN = 23592.0 / 60.0          # City-wide arrival rate: 23,592 pax/hr (~393.2/min)

# Default parameter tuples: (n, J, nu, tau, lambda_per_corner)
#   n      : Fleet size (number of drivers)
#   J      : Number of corners in the city grid (80^2 = 6,400)
#   nu     : Vehicle travel speed in blocks per minute
#   tau    : Inverse of average trip distance in blocks (1 / 40 blocks)
#   lambda_: Per-corner passenger arrival rate per minute (lambda * J / J)
DEFAULT_PARAMS_S = (6783, NUM_BLOCKS**2, 3, 1/40, BASE_ARRIVALS_PER_MIN / (NUM_BLOCKS**2))
DEFAULT_PARAMS_D = (5350, NUM_BLOCKS**2, 3, 1/40, BASE_ARRIVALS_PER_MIN / (NUM_BLOCKS**2))
DEFAULT_PARAMS_SIM = (NUM_BLOCKS, SPEED_MPH, TRIP_LENGTH_MILES, 0)


# =============================================================================
# STREET-HAILING AND CENTRAL DISPATCH ANALYTICAL APPROXIMATIONS
# =============================================================================

def f_s(w, params_s=DEFAULT_PARAMS_S):
    """
    Probability density function (PDF) for passenger wait time under Street-Hailing.
    Derived from Approximation 1 in Section 4.1 of (Main_Document):
        f_s(w) = theta * exp(-theta * w),
    where theta = (nu * n_s - lambda * J * (1 + tau) / tau) / J is the effective
    corner service rate in the matching-dominated regime.

    Args:
        w (float or np.ndarray): Passenger wait time(s) in minutes (w >= 0).
        params_s (tuple): (n_s, J, nu, tau, lambda_) model parameters.

    Returns:
        float or np.ndarray: Street-hailing PDF evaluated at w.
    """
    n_s, J, nu, tau, lambda_ = params_s
    theta = (nu * n_s - lambda_ * J * (1 + tau) / tau) / J
    return theta * np.exp(-theta * w)


def f_d(w, params_d=DEFAULT_PARAMS_D):
    """
    Probability density function (PDF) for passenger wait time under Central Dispatch.
    Derived from Approximation 2 in Section 4.2 of (Main_Document) (Rayleigh distribution):
        f_d(w) = 2 * K * w * exp(-K * w^2),
    where K = 2 * nu^2 * ((n_d - lambda * J / (tau * nu)) / J).

    Args:
        w (float or np.ndarray): Passenger wait time(s) in minutes (w >= 0).
        params_d (tuple): (n_d, J, nu, tau, lambda_) model parameters.

    Returns:
        float or np.ndarray: Central dispatch PDF evaluated at w.
    """
    n_d, J, nu, tau, lambda_ = params_d
    K = 2 * nu**2 * ((n_d - lambda_ * J / (tau * nu)) / J)
    pdf_val = 2 * K * w * np.exp(-K * w**2)
    return pdf_val


def F_s(w, params_s=DEFAULT_PARAMS_S):
    """
    Cumulative distribution function (CDF) for passenger wait time under Street-Hailing.
    Implements Equation (1a) in Approximation 1 of (Main_Document):
        F_s(w, lambda, n_s) = 1 - exp(-w * (nu * n_s - lambda * J * (1 + tau) / tau) / J).

    Args:
        w (float or np.ndarray): Passenger wait time(s) in minutes (w >= 0).
        params_s (tuple): (n_s, J, nu, tau, lambda_) model parameters.

    Returns:
        float or np.ndarray: Probability P(W_s <= w).
    """
    n_s, J, nu, tau, lambda_ = params_s
    return 1 - np.exp(-w * (nu * n_s - lambda_ * J * (1 + tau) / tau) / J)


def F_d(w, params_d=DEFAULT_PARAMS_D):
    """
    Cumulative distribution function (CDF) for passenger wait time under Central Dispatch.
    Implements Equation (2a) in Approximation 2 of (Main_Document):
        F_d(w, lambda, n_d) = 1 - exp(-2 * (w * nu)^2 * ((n_d - lambda * J / (tau * nu)) / J)).

    Args:
        w (float or np.ndarray): Passenger wait time(s) in minutes (w >= 0).
        params_d (tuple): (n_d, J, nu, tau, lambda_) model parameters.

    Returns:
        float or np.ndarray: Probability P(W_d <= w).
    """
    n_d, J, nu, tau, lambda_ = params_d
    exponent = -2 * (w * nu)**2 * ((n_d - lambda_ * J / (tau * nu)) / J)
    return 1 - np.exp(exponent)


def taxi_wait(n, params_s=(NUM_BLOCKS**2, 3, 1/40, BASE_ARRIVALS_PER_MIN / (NUM_BLOCKS**2))):
    """
    Approximate expected passenger wait time under Street-Hailing, E[W_s].
    Implements Equation (1b) in Approximation 1 of (Main_Document):
        W_bar_s(lambda, n_s) = J / (n_s * nu - lambda * J * (1 + tau) / tau).

    Args:
        n (float): Street-hailing fleet size (n_s).
        params_s (tuple): (J, nu, tau, lambda_) city and demand parameters.

    Returns:
        float: Expected passenger wait time in minutes.
    """
    J, nu, tau, lambda_ = params_s
    return (J * tau) / (n * nu * tau - J * lambda_ * (1 + tau))


def uber_wait(n, params_d=(NUM_BLOCKS**2, 3, 1/40, BASE_ARRIVALS_PER_MIN / (NUM_BLOCKS**2))):
    """
    Approximate expected passenger wait time under Central Dispatch, E[W_d].
    Implements Equation (2b) in Approximation 2 of (Main_Document):
        W_bar_d(lambda, n_d) = (sqrt(2 * pi) / (4 * nu)) * sqrt(J / (n_d - lambda * J / (tau * nu))).

    Args:
        n (float): Central dispatch fleet size (n_d).
        params_d (tuple): (J, nu, tau, lambda_) city and demand parameters.

    Returns:
        float: Expected passenger wait time in minutes.
    """
    J, nu, tau, lambda_ = params_d
    return (math.sqrt(2 * pi) / (4 * nu)) * math.sqrt(J / (n - (J * lambda_) / (nu * tau)))


# =============================================================================
# DIAGNOSTIC SUMMARY (Executed when module is imported or run directly)
# =============================================================================
num_of_blocks = 80
speed = 10
trip_length_in_miles = 2
block_length = 4 / num_of_blocks
print(f"block length: {block_length} miles")
print(f"trip length: {trip_length_in_miles/block_length} blocks")
print(f"time per block: {60*(block_length/speed)} minutes")
print(f"speed: {int((speed/60)/block_length)} blocks per minute")
print(f"taxi wait in minutes: {taxi_wait(6785)} minutes")
print(f"uber wait in minutes: {uber_wait(5350)} minutes")