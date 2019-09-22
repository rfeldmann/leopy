"""Example of data with missing and censored values and lognormal distribution

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

savefig = False
optimize = True

np.random.seed(1)
fontsize=17

print('Example: Linear regression of correlated, normally distributed data '
      'with correlated observational errors.')

dist = scipy.stats.norm
Ndata = 300
rho = 0.

m = 1.  # true slope
n = 0.0  # true intercept

rho_c_s = [0., -0.8]

yobs_lower = -np.inf  # lower detection limit for censoring

min_t = -1.5
max_t = 1.5

t = scipy.stats.uniform.rvs(size=Ndata, loc=min_t, scale=max_t-min_t)  # latent variable
loc = np.array((t, m*t+n)).T

scale = np.array([0.001, 1.])
R = np.array([[1., rho], [rho, 1.]])

random_state = np.random.get_state()

for rho_c, loc_legend, key_left in zip(
    rho_c_s, ['upper left', 'upper right'], [False, True]):

    np.random.set_state(random_state)

    ## -- create "true" observational data

    print('-- x sample --')
    x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)
    print('{}'.format(pd.DataFrame(x).describe()))
    rho_x_sample = np.corrcoef(x.T)[0, 1]
    print('rho(x) = {}\n'.format(rho_x_sample))

    print('-- y sample --')
    y = dist.ppf(scipy.stats.norm.cdf(x), loc=loc, scale=scale)
    print('{}'.format(pd.DataFrame(y).describe()))
    rho_y_sample = np.corrcoef((y-loc).T)[0, 1]
    print('rho(y) = {}\n'.format(rho_y_sample))

    ## -- add observational error

    #sigma_c = [0.5, 2.]
    sigma_c = [1., 1.5]
    #sigma_c = [1.5, 3.]
    e_yobs = np.zeros_like(y)
    e_yobs[:, 0] = sigma_c[0]
    e_yobs[:, 1] = sigma_c[1]

    error_yobs = np.zeros_like(y)
    R_c = np.zeros((Ndata, 2, 2))
    cov_c = np.zeros_like(R_c)

    for i in range(Ndata):
        R_c[i] = np.array([[1., rho_c], [rho_c, 1]])
        cov_c[i] = np.diag(sigma_c).dot(R_c[i].dot(np.diag(sigma_c)))
        error_yobs[i, :] = scipy.stats.multivariate_normal.rvs(cov=cov_c[i])

    print('-- y_obs sample (y w/ error) --')
    yobs_all = y + error_yobs
    print('{}'.format(pd.DataFrame(yobs_all).describe()))
    rho_yobs_sample = np.corrcoef((yobs_all-loc).T)[0, 1]
    print('rho(y_obs) = {}\n'.format(rho_yobs_sample))

    ## -- censor some observations
    yobs = np.copy(yobs_all)
    cen_yobs = np.zeros(Ndata, dtype=bool)
    lim_yobs = np.zeros(Ndata)
    sel = (yobs[:, 1] < yobs_lower)
    lim_yobs[sel] = yobs_lower
    cen_yobs[sel] = True
    yobs[sel, 1] = 0.
    n_cen = np.sum(cen_yobs == True)

    ## -- complete cases
    print('-- complete cases of y_obs sample --')
    sel = np.all(~np.isnan(yobs), axis=1) & (cen_yobs == False)
    yobs_cc = yobs[sel]
    e_yobs_cc = e_yobs[sel]
    cen_yobs_cc = cen_yobs[sel]
    lim_yobs_cc = lim_yobs[sel]
    n_cc = np.sum(sel)
    print('{}'.format(pd.DataFrame(yobs_cc).describe()))
    rho_yobs_cc_sample = np.corrcoef((yobs_cc-loc[sel]).T)[0, 1]
    print('rho(y_obs) = {}\n'.format(rho_yobs_cc_sample))

    print('{} total cases, {} complete, {} censored'.format(Ndata, n_cc, n_cen))

    ## --- plotting

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.patches
    plt.ion()

    if key_left:
        align = 'left'
        align_x = 0.02
    else:
        align = 'right'
        align_x = 0.98

    plt.figure(figsize=(6,6))
    ls = []
    hs = []

    if n_cen > 0:
        n_cen_label = r', $N_{{\rm cen}}={}$'.format(n_cen)
    else:
        n_cen_label = ''
    plt.text(align_x, 0.96,
             r'$N_{{\rm tot}}={}${}'.format(Ndata, n_cen_label),
             fontsize=fontsize-4, color='k',
             horizontalalignment=align, transform=plt.gca().transAxes)
    if rho_c == 0:
        aux = 'uncorrelated errors'
    else:
        aux = 'error corr. = {}'.format(rho_c)
    plt.text(align_x, 0.91, aux, fontsize=fontsize-5, color='k',
             horizontalalignment=align, transform=plt.gca().transAxes)

    # show censored data
    sel = cen_yobs
    n_sel = np.sum(sel)
    if n_sel > 0:
        col_text = [0, 0.6, 0.6]
        col = [0, 0.7, 0.7]
        plt.text(0.8, 0.03, 'censored y', fontsize=fontsize-6,
                 color=col_text, horizontalalignment='left',
                 transform=plt.gca().transAxes)
        for i, s in enumerate(sel):
            if not s:
                continue
            plt.annotate("", xy=(yobs[i, 0], lim_yobs[i]),
                         xytext=(yobs[i, 0], lim_yobs[i]-2.),
                         arrowprops=dict(arrowstyle="<|-", color=col))
            plt.plot(yobs[i, 0], lim_yobs[i], marker='1', color=col, markersize=4)

    plt.xlim([-6, 6])
    if n_cen > 0:
        plt.ylim([yobs_lower-2, 9])
    else:
        plt.ylim([-7, 9.5])
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('y', fontsize=fontsize)

    plt.tick_params(axis='both', which='major', direction='inout',
                    labelsize=fontsize-2)
    plt.tick_params(axis='both', which='minor', direction='inout',
                    labelsize=fontsize-6)
    plt.minorticks_on()
    plt.tight_layout()

    xx = np.linspace(np.min(yobs[:, 0]), np.max(yobs[:, 0]), 100)

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    for yy, col, linestyle, l in zip([yobs_cc],
                       ['navy'],
                       ['-.'],
                       [r'least squares regression']):
        df = pd.DataFrame(yy, columns=['x', 'y'])
        res = smf.ols('y ~ x', data=df).fit()
        slope = res.params['x']
        intercept = res.params['Intercept']
        print(res.summary2())
        h, = plt.plot(xx, intercept + slope * xx, ls=linestyle, color=col, lw=2)
        hs.append(h)
        ls.append(l)

    h, = plt.plot(xx, m * xx + n, '--k', lw=2, alpha=0.5)
    hs.append(h)
    ls.append('true model')

    # plot data points and error ellipses
    sel = ~cen_yobs
    for i, s in enumerate(sel):
        if not s:
            continue
        # see https://www.xarg.org/2018/04/
        # how-to-plot-a-covariance-error-ellipse/
        s = -2 * np.log((1-0.6827)/2)  # to show 68% error ellipse (1sigma)
        evs, evecs = np.linalg.eigh(s * cov_c[i])  # EV with max eigenvalue
        evec = evecs[:, -1]
        ell=matplotlib.patches.Ellipse(
            (yobs[i, 0], yobs[i, 1]), height=np.sqrt(evs[0]),
            width=np.sqrt(evs[1]),
            angle=np.arctan2(evec[1], evec[0])*180/np.pi, lw=1)
        plt.gca().add_artist(ell)
        ell.set_facecolor([0.9, 0.9, 1.])
        ell.set_edgecolor('b')
        ell.set_alpha(0.05)

    ## -- linear regression (Maximum likelihood with leopy)
    import leopy

    (ly, ley, lceny, llimy) = (yobs, e_yobs, cen_yobs, lim_yobs)
    df = pd.DataFrame(np.array([ly[:, 0], ly[:, 1], ley[:, 0], ley[:, 1],
                                lceny, llimy, rho_c * np.ones(Ndata)]).T,
                      columns=['v0', 'v1', 'e_v0', 'e_v1', 'c_v1', 'l_v1',
                               'r_v0_v1'])

    obs = leopy.Observation(df, 'test', verbosity=0)

    ## -- set up Likelihood and find maximum likelihood parameters
    like = leopy.Likelihood(obs, p_true='norm', p_cond='norm',
                            verbosity=-1)

    def f_mlnlike(x):
        #print(x)
        nt = 200
        t = np.linspace(min_t, max_t, nt)

        m = x[0]
        n = x[1]

        loc_true = np.ones((2, 1, nt))
        loc_true[0, :] = t
        loc_true[1, :] = m * t + n

        scale_true = [scale[0], x[2]]

        p_xy = like.p(loc_true, scale_true)

        m_ln_p_xy = -np.sum(np.log(scipy.integrate.simps(p_xy, axis=1)))

        #p_x = like.p(loc_true, scale_true, vars=[0])
        #m_ln_p_x = -np.sum(np.log(scipy.integrate.simps(p_x, axis=1)))
        m_ln_p_x = 0.

        return m_ln_p_xy - m_ln_p_x

    bounds = scipy.optimize.Bounds(
        [-np.inf, -np.inf, 1e-3],
        [np.inf, np.inf, 10.])
    print('Running ML optimization ...')
    if optimize:
        optres = scipy.optimize.minimize(f_mlnlike, [0., 0., 1.],
                                         bounds=bounds, method='SLSQP',
                                         options={'disp': True,
                                         'ftol': 1e-12})

        print('Maximum likelihood parameters: [{:.3g} {:.3g} '
              '{:.3g}]'.format(*optres.x))

        ## -- plot fit results
        xx = np.linspace(np.min(yobs[:, 0]), np.max(yobs[:, 0]), 100)
        h, = plt.plot(xx, optres.x[0] * xx + optres.x[1], '-',
                      color='lightcoral', lw=2)
        hs.append(h)
        ls.append('ML estimate')

    # data points
    h, = plt.plot(yobs[sel, 0], yobs[sel, 1], '.', color='b', alpha=0.8)
    hs.append(h)
    ls.append(r'observed $(x^{\rm obs}, y^{\rm obs})$')

    h, = plt.plot(y[sel, 0], y[sel, 1], '.', color='k',
                  markerfacecolor='None', alpha=0.4)
    hs.append(h)
    ls.append(r'true data $(x, y)$')

    if loc_legend == 'upper right':
        markerfirst = False
    else:
        markerfirst = True
    plt.legend(hs, ls, loc=loc_legend, markerfirst=markerfirst)

    if savefig:
        plt.savefig('lin_regression_rc{}.pdf'.format(rho_c))
