"""Maximum likelihood analysis of the xGASS data set.

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
import time

import statsmodels.formula.api as smf

# turn off accuracy warnings in the convolution & integrate modules
# comment out if you prefer to see all warnings
import warnings
from leopy.misc import AccuracyWarning
warnings.filterwarnings('ignore', category=AccuracyWarning)

import leopy

import auxiliary
import xGASS_auxiliary

sel_lgMstar = [9, 11.5]

f_tol = 1e-8
T = 0.01
method = 'SLSQP'
n_basinhopping = 20

start = [0.5, 0., 0., 0., 1., -1., 1., -0.5, -2]

print('Analysis of the SFR -- Mstar relation in the xGASS data set')

lx = 'v0'
lsx = 'e_v0'
ly = 'v1'
lsy = 'e_{}'.format(ly)
lcy = 'c_{}'.format(ly)
lly = 'l_{}'.format(ly)

# -- Step 1. Read the data and print statistics
catalog_filename = 'xGASS.csv'

df = pd.read_csv(catalog_filename, comment='#', na_values='.')
df = df.rename(columns={
    'lgMstar': 'v0', 'SFR': 'v1',
    'e_SFR': 'e_v1', 'SFR_censored': 'c_v1', 'SFR_limit': 'l_v1'})

df = df[(df['v0'] >= sel_lgMstar[0]) & (df['v0'] <= sel_lgMstar[1])]

# statistics:
if lcy in df:
    Ndet = np.sum((df[lcy] == False) & np.logical_not(np.isnan(df[ly])))
    Ncen = np.sum((df[lcy]))
    Nmis = np.sum((df[lcy] != True) & np.isnan(df[ly]))
else:
    Ndet = np.sum(np.logical_not(np.isnan(df[ly])))
    Ncen = 0
    Nmis = np.sum(np.isnan(df[ly]))
print('Data set: Ntotal = {}, Ndet = {}, Ncen = {}, Nmiss = {}'.format(
    df.shape[0], Ndet, Ncen, Nmis))

# downsampling for test purposes
if 0:
    np.random.seed(2)
    df = df.sample(frac=0.1)
    print('Downsampling: Ntotal = {}, Ndet = {}, Ncen = {}, Nmiss = {}'.format(
        df.shape[0], Ndet, Ncen, Nmis))

# -- Step 2. Prepare LEO-Py
obs = leopy.Observation(df, 'xGASS', variables=[lx, ly])
df = obs.df

like = leopy.Likelihood(
    obs, p_true=['norm', leopy.stats.zi_gamma_lognorm],
    p_cond=[None, 'norm'])

# -- Step 3. Prepare Maximum Likelihood analysis
def f_mlnlike(x, pool):
    """Return minus log likelihood (rescaled)."""
    if np.any(np.isnan(x)):
        return 1000.

    Nobs = df.shape[0]
    t = df['v0'].to_numpy().reshape(Nobs, 1)

    m_scale, n_scale = xGASS_auxiliary.get_slope_intercept(x[0], x[1])
    m_shape, n_shape = xGASS_auxiliary.get_slope_intercept(x[2], x[3])
    m_zero, n_zero = xGASS_auxiliary.get_slope_intercept(x[4], x[5])

    m_out_scale, n_out_scale = xGASS_auxiliary.get_slope_intercept(0., x[6])
    m_out_shape, n_out_shape = xGASS_auxiliary.get_slope_intercept(0., x[7])
    m_out_frac, n_out_frac = xGASS_auxiliary.get_slope_intercept(0., x[8])

    loc_true = np.zeros((2, Nobs, 1))
    loc_true[0, :] = t

    scale_true = np.zeros((2, Nobs, 1))
    scale_true[0, :] = 1e-2
    scale_true[1, :] = 10**(m_scale * (t-10) + n_scale)

    scale_zero = 1e-2 * np.ones((Nobs, 1))

    shape_true = [[],
                  [10**(m_shape * (t-10) + n_shape),
                   scale_zero,
                   leopy.stats.inv_logit(m_zero * (t-10) + n_zero),
                   np.log(10)*10**(  # ln(10) to allow input in dex
                    m_out_shape * (t-10) + n_out_shape),
                   10**(m_out_scale * (t-10) + n_out_scale),
                   leopy.stats.inv_logit(m_out_frac * (t-10) + n_out_frac)]
                  ]

    p_xy = like.p(loc_true, scale_true, shape_true=shape_true, pool=pool)

    if np.any(p_xy == 0):
        m_ln_p_xy = 1000.
    else:
        m_ln_p_xy = -np.sum(np.log(p_xy))
        m_ln_p_xy /= np.float(Nobs)

    if 0:
        print('[', end='')
        print(', '.join(map(lambda _: '{:.5g}'.format(_), x)), end='')
        print('] ', end='')
        print(m_ln_p_xy)

    return m_ln_p_xy

bounds = scipy.optimize.Bounds(
    [np.arctan(-10), -2, np.arctan(-10), -2, np.arctan(-10),
     -np.inf, 0, -np.inf, -10.],
    [np.arctan(10), np.inf, np.arctan(10), np.inf, np.arctan(10),
     np.inf, np.inf, np.inf, leopy.stats.logit(0.30)])

try:
    from schwimmbad import MultiPool
    pool = MultiPool()
    print('Parallel execution on ' + str(pool.size) + ' processes')
except ImportError as error:
    print('Serial execution as module `schwimmbad` was not found')
    pool = None

my_print_fun = auxiliary.MyPrintFun()
my_take_step = auxiliary.MyTakeStep(stepsize=1)
minimizer_options = {'disp': True, 'ftol': 1e-8}
minimizer_kwargs = {'method': method, 'bounds': bounds,
                    'options': minimizer_options, 'args': (pool,)}

# -- Step 4. Run Maximum Likelihood analysis (basinhopping)
np.random.seed(1)
print('Running the ML analysis - this may take a while')
optres = scipy.optimize.basinhopping(f_mlnlike, start, niter=n_basinhopping,
                                     T=T, interval=3,
                                     minimizer_kwargs=minimizer_kwargs,
                                     take_step=my_take_step,
                                     callback=my_print_fun)
print('[', end='')
print(', '.join(map(lambda _: '{:.5g}'.format(_), optres.x)), end='')
print('] ', end='')
print('{}'.format(optres.fun))

if pool:
    pool.close()

# -- Step 5. Output results
print('-- Likelihood maximization of variable ' + ly + ' --')
print(', '.join(map(lambda x: '{:.3g}'.format(x), optres.x)))

fit_result = optres.x

m_scale, n_scale = xGASS_auxiliary.get_slope_intercept(
    fit_result[0], fit_result[1])
m_shape, n_shape = xGASS_auxiliary.get_slope_intercept(
    fit_result[2], fit_result[3])
m_zero, n_zero = xGASS_auxiliary.get_slope_intercept(
    fit_result[4], fit_result[5])

xx = np.linspace(max(min(df['v0']), sel_lgMstar[0]),
                 min(max(df['v0']), sel_lgMstar[1]), 100)
shape_xx = 10**(m_shape * (xx-10) + n_shape)
list_res_scatter = []
for r in [1, 2]:
    scatter_up_xx, _ = xGASS_auxiliary.calc_scatter(shape_xx, r=r)
    aux = pd.DataFrame({'scatter': scatter_up_xx, 'lg_M10': (xx-10)})
    list_res_scatter.append(
        smf.ols('scatter ~ lg_M10', data=aux).fit())

# print table
# m_zero, n_zero, m_SFR, n_SFR, m_scatter(r=1), n_scatter(r=1) + for r=2
print('-- var = {} -- '.format(ly))
print('{} -- {}'.format(*sel_lgMstar), end='')
print(' & {} & {} & {} & '.format(df.shape[0], Ncen, Nmis), end='')
print('{:.2g} & {:.3g} & '.format(
    m_shape + m_scale, n_shape + n_scale), end='')
for ir, r in enumerate([1, 2]):
    print('{:.2g} & {:.2g} & '.format(
        list_res_scatter[ir].params['lg_M10'],
        list_res_scatter[ir].params['Intercept']),
        end='')
print('{:.3g} & {:.3g}\n'.format(m_zero, n_zero), end='')
