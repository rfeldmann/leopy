"""Auxiliary functions for the analysis of the xGASS data set.

This module is part of LEO-Py --
Likelihood Estimation of Observational data with Python

Copyright 2019 University of Zurich, Robert Feldmann

LEO-Py is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LEO-Py is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with LEO-Py. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy
import scipy.special
import scipy.stats


def put_SFR_floor(x, y, sy, f_err):
    """Impose minimum absolute and relative SFR uncertainty."""
    # x is lgMstar
    # y is SFR
    # sy is uncertainty of SFR
    # f_err is a minimum relative uncertainty

    # impose mininum absolute uncertainty
    res = np.maximum(
       sy, 10**((0.127*(x-10) - 0.733)
                / (0.045*(x-10) + (1 - 0.283))))

    # impose minimum relative uncertainty
    res = np.maximum(res, f_err*y)

    return res


def get_SFR_equals_eSFR(x, f=1.):
    """Return SFR equaling e_SFR according to the SFR_error fit."""
    # x is lgMstar
    # return SFR s.t. SFR = f * e_SFR(SFR) according to SFR_error() function
    return 10**((0.127*(x-10) - 0.733 + np.log10(f))
                / (0.045*(x-10) + (1 - 0.283)))


def SFR_error(x, y):
    """Fit to typical SFR uncertainty in GASS, see xGASS_SFR_uncertainty.py ."""
    # x is lgMstar
    # y is SFR
    return 10**(0.127 * (x-10) + 0.283 * np.log10(y) +
                -0.045 * (x-10) * np.log10(y) - 0.733)


def calc_scatter(a, r):
    """Return upward and downward scatter based on probability density."""
    # see paper for explanation
    z = - np.exp(-r**2/(2.*a)) / np.exp(1.)
    W_plus = scipy.special.lambertw(z, k=0).real
    W_minus = scipy.special.lambertw(z, k=-1).real
    scatter_up = np.log10(-W_minus)/r
    scatter_down = -np.log10(-W_plus)/r
    return scatter_up, scatter_down


def calc_scatter2(a):
    """Return upward and downward scatter based on percentiles."""
    mean = a
    mean_perc = scipy.stats.gamma.cdf(mean, a)
    # now only consider galaxies above, want 34-percentile above sequence
    # i.e., 84 percentile if sequence is at median
    upper_perc = 1 - 0.1587/0.5 * (1 - mean_perc)
    lower_perc = 0.1587/0.5 * mean_perc
    # get position
    scatter_up = np.log10(scipy.stats.gamma.ppf(upper_perc, a) / mean)
    scatter_down = np.log10(mean / scipy.stats.gamma.ppf(lower_perc, a))
    return scatter_up, scatter_down


def get_slope_intercept(theta, perp):
    """Map angle & perp. distance to slope & intercept parametrization."""
    # input: angle of line to x-axis and perp. distance of line to origin
    # output: slope and intercept
    m = np.tan(theta)
    n = perp / np.cos(theta)
    return m, n
