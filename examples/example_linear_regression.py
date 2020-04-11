"""Example of linear regression of normally distributed data

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

import matplotlib.pyplot as plt
import matplotlib.patches
plt.ion()

savefig = True

np.random.seed(2)
fontsize=17

print('Example: Linear regression of normally distributed data')

Nobs = 100

m = 2.   # true slope
n = 0.5  # true intercept
s = 0.4  # intrinsic scatter

uncert_x = 1. * np.ones(Nobs) # observational uncertainty of x
uncert_y = 0.7 * np.ones(Nobs) # observational uncertainty of y

min_x = -1.5
max_x = 1.5

## -- create data set

x_true = scipy.stats.uniform.rvs(size=Nobs, loc=min_x, scale=max_x-min_x)
y_true = np.array(m*x_true+n) + s * scipy.stats.norm.rvs(size=Nobs)

# assume here uncorrelated errors for x and y
e_x = uncert_x * scipy.stats.multivariate_normal.rvs(size=Nobs)
e_y = uncert_y * scipy.stats.multivariate_normal.rvs(size=Nobs)
x = x_true + e_x
y = y_true + e_y

## --- plot data set

plt.figure()
hs = []
ls = []

h = plt.errorbar(x, y, yerr=uncert_y, xerr=uncert_x, fmt='o', zorder=1)
hs.append(h)
ls.append('Observed data')

h, = plt.plot(x_true, y_true, '.k')
hs.append(h)
ls.append('true data w/ intrinsic scatter')

h, = plt.plot([min_x, max_x], m * np.array([min_x, max_x]) + n, '--k',
              lw=2, alpha=1., zorder=2)
hs.append(h)
ls.append('true data w/o intrinsic scatter')


## -- linear regression (Maximum likelihood with leopy)
import leopy

df = pd.DataFrame(np.array([x, y, uncert_x, uncert_y]).T,
                  columns=['v0', 'v1', 'e_v0', 'e_v1'])

obs = leopy.Observation(df, 'test', verbosity=0)

## -- set up Likelihood and find maximum likelihood parameters
like = leopy.Likelihood(obs, p_true='norm', p_cond=[None, 'norm'],
                        verbosity=-1)

def f_lnlike(p, pool):
    print(p)

    # p are the three parameters of the fit
    # the slope (p[0])
    # the intercept (p[1])
    # and the intrinsic scatter (p[2])

    Nmod = 200
    dt = np.linspace(-4, 4, Nmod)  # in units of meas. uncert.
    et = df['e_v0'].to_numpy().reshape(Nobs, 1)  # meas. uncert.
    # t is unknown x_true
    t = np.outer(df['v0'].to_numpy().reshape(Nobs, 1), np.ones(Nmod))
    t += np.outer(et, dt)

    # one has to use the correct prior for x_true or results will be biased
    pt = ((min_x < t) & (t <= max_x))/(max_x - min_x)

    m = p[0]
    n = p[1]
    s = p[2]

    loc_true = np.ones((2, Nobs, Nmod))
    loc_true[0, :] = t
    loc_true[1, :] = m * t + n

    scale_true = [et, s]

    p_xy = like.p(loc_true, scale_true, pool=pool)

    p_x = scipy.stats.norm.pdf(dt) / et
    p_y_x = (scipy.integrate.simps(p_xy * pt, t, axis=1) /
             scipy.integrate.simps(p_x * pt, t, axis=1))
    p_res = p_y_x

    if  np.any(p_res <= 0):
        ln_p_res = -1e3
    else:
        ln_p_res = np.sum(np.log(p_res))

    return ln_p_res

def f_mlnlike(p, pool):
    return -f_lnlike(p, pool)

bounds = scipy.optimize.Bounds([-5, -5, 1e-3],
                               [5, 5, 10.])


print('Running ML optimization ...')
from schwimmbad import MultiPool
pool = MultiPool()
optres = scipy.optimize.minimize(f_mlnlike, [1., 1., 0.2],
                                 bounds=bounds, method='SLSQP',
                                 args=(pool,), options={
                                 'disp': True, 'ftol': 1e-12})

plt.text(0.03, 0.98, r'$N_{{\rm obs}}$ = {}'.format(Nobs),
         transform=plt.gca().transAxes, horizontalalignment='left',
         verticalalignment='top')
plt.text(0.03, 0.93, 'True parameters: m={}, n={}, s={}'.format(m, n, s),
         transform=plt.gca().transAxes, horizontalalignment='left',
         verticalalignment='top')
plt.text(0.03, 0.88, 'LEO-Py estimate: m={:.3g}, n={:.3g}, s={:.3g}'.format(
         *optres.x), transform=plt.gca().transAxes, horizontalalignment='left',
         verticalalignment='top')

print('Maximum likelihood parameters: {}'.format(*optres.x))
pool.close()

## -- plot fit results
xx = np.linspace(np.min(x), np.max(x), 100)
h, = plt.plot(xx, optres.x[0] * xx + optres.x[1], '-',
              color='lightcoral', lw=2, zorder=3)
hs.append(h)
ls.append('LEO-Py ML fit')

plt.xlabel('x', fontsize=fontsize)
plt.ylabel('y', fontsize=fontsize)

plt.legend(hs, ls, loc='lower right', markerfirst=False)

plt.tick_params(axis='both', which='major', direction='inout',
                labelsize=fontsize-2)
plt.tick_params(axis='both', which='minor', direction='inout',
                labelsize=fontsize-6)
plt.minorticks_on()

if savefig:
    plt.savefig('example_linear_regression.pdf')
