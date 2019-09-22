"""Example of data with missing values (NOT missing completely at random) and
    with a normal distribution.

This file is part of LEO-Py --
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
import pandas as pd
import scipy.stats
import scipy.optimize

import pytest

import leopy

np.random.seed(16)

dist = scipy.stats.norm
Ndata = 300
rho = 0.5
R = np.array([[1., rho], [rho, 1.]])
loc_true = np.array([0., 2.])
scale_true = np.array([1., 3.])

print('population parameters: [{} {} {} {} {}]'.format(
    *loc_true, *scale_true, rho))

## -- create observational data
x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)

# print('rho(x) = {}'.format(np.corrcoef(x.T)))
rho_x_sample = np.corrcoef(x.T)[0, 1]

y = dist.ppf(scipy.stats.norm.cdf(x),
             loc=loc_true, scale=scale_true)
y_true = np.copy(y)
ey = np.zeros_like(y)
ey[:, 0] = 0.2  # 1e-6  # 0.2 # 0.1
ey[:, 1] = 0.1 # 1e-6   # 0.1
y[:, 0] += ey[:, 0] * np.random.randn(Ndata)
y[:, 1] += ey[:, 1] * np.random.randn(Ndata)

print('sample parameters: [{:.3g} {:.3g} {:.3g} {:.3g} {:.3g}]'.format(
    np.nanmean(y[:, 0]), np.nanmean(y[:, 1]),
    np.nanstd(y[:, 0]), np.nanstd(y[:, 1]), rho_x_sample))

# print('rho(y) = {}'.format(np.corrcoef(y.T)))

def logistic(x):
    return np.exp(x) / (np.exp(x) + 1.)

# data missing data at random (MAR) based on values of other column
m1 = scipy.stats.bernoulli.rvs(logistic(y[:, 0]-1.)).astype(bool)  # for col 1
m0 = scipy.stats.bernoulli.rvs(logistic(y[:, 1]-2.)).astype(bool)  # for col 0
y[m1, 1] = np.float('NaN')
y[m0, 0] = np.float('NaN')

print('sample parameters (w/ missing as NaN): [{:.3g} {:.3g} {:.3g} {:.3g} N/A]'.format(
    np.nanmean(y[:, 0]), np.nanmean(y[:, 1]),
    np.nanstd(y[:, 0]), np.nanstd(y[:, 1])))

# complete cases:
ycc = y[np.all(~np.isnan(y), axis=1)]
eycc = ey[np.all(~np.isnan(y), axis=1)]

ncc = np.sum(np.all(~np.isnan(y), axis=1))
ncm = np.sum(np.all(np.isnan(y), axis=1))
print('{} total cases, {} complete, {} incomplete, {} completely '
      'missing'.format(Ndata, ncc, Ndata - ncc, ncm))

for irun in range(2):

    if irun == 0:
        print('--- Using all data (incl. missing) ---')
        ly = y
        ley = ey
    else:
        print('--- Using only complete cases ---')
        ly = ycc
        ley = eycc

    df = pd.DataFrame(np.array([ly[:, 0], ly[:, 1], ley[:, 0], ley[:, 1]]).T,
                      columns=['v0', 'v1', 'e_v0', 'e_v1'])
    obs = leopy.Observation(df, 'test', verbosity=0)

    ## -- set up Likelihood and find maximum likelihood parameters
    like = leopy.Likelihood(obs, p_true='norm', p_cond='norm')

    # comment out the following two lines to force numerical convolution
    # like.p_obs[0].name = 'composite'
    # like.p_obs[1].name = 'composite'

    def f_mlnlike(x):
        # print(x)
        loc_true = x[0:2]
        scale_true = x[2:4]
        rho = x[4]
        R = np.array([[1., rho], [rho, 1.]])
        pp = like.p(loc_true, scale_true, R_true=R)
        if np.sum(pp==0) > 0:
            return np.inf
        else:
            return -np.sum(np.log(pp))

    bounds = scipy.optimize.Bounds(
        [-np.inf, -np.inf, 1e-3, 1e-3,  1e-3],
        [np.inf, np.inf, 10., 10., 1-1e-3])
    optres = scipy.optimize.minimize(f_mlnlike, [0., 0., 1., 1., 0.3],
                                     bounds=bounds, method='SLSQP',
                                     options={'disp': False, 'ftol': 1e-12})
    print('Maximum Likelihood optimization: '
          '[{:.3g} {:.3g} {:.3g} {:.3g} {:.3g}]'.format(*optres.x))
