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

import leopy
import auxiliary

savefig = True
showfig = True
key_left = True
optimize = False

fontsize=17

analytic_joint_pdf = True
N_mock = int(3e7)

np.random.seed(2)

# --- Data set (created with Ndata=200 and seed=2) ---
#
# population parameters: [0.0 0.0 1 2 0.5 1.5 0.9]
# population statistics (mean, std, corr): [1.13 6.16 0.604 17.9 0.9]
# sample statistics full data (mean, std, corr): n=200 [1.12 7.1 0.621 16.4 0.906]
# sample statistics (w/ missing as NaN): n=200 [1.04 6.69 0.655 16.6 N/A]
# sample statistics (without any NaNs): n=147 [1.04 7.26 0.655 19.2 0.791]
# sample statistics (complete cases]): n=46 [1.76 23.2 0.682 28.4 0.791]
# 200 total cases, 46 complete, 101 censored, 53 incomplete, 0 completely missing
#
# --- Result of ML parameter estimation (Ndata = 200) ---
#
# case 0: proper modeling of missing and censored data
# [-0.023716, 0.29896, -0.28695, 0.19859, 2.6028] 2.3917777845992307
# Maximum likelihood parameters: [0 0 0.947 1.99 0.516 1.58 0.931]
# Maximum likelihood statistics: [1.08 6.93 0.598 23.1 0.931]
#
# case 1: dropping missing data
# [-0.064101, -0.08104, -0.24516, 0.35151, 3.2766] 2.095509727700858
# Maximum likelihood parameters: [0 0 0.863 0.83 0.569 2.25 0.964]
# Maximum likelihood statistics: [1.01 10.3 0.627 129 0.964]
#
# case 2: dropping missing and censored data (complete cases)
# [0.2168, 1.1177, -0.4228, 0.019954, 1.6785] 4.424127154180376
# Maximum likelihood parameters: [0 0 1.65 13.1 0.378 1.05 0.843]
# Maximum likelihood statistics: [1.77 22.7 0.693 32 0.843]
#
# case 3: no correlations
# [-0.056388, 0.28618, -0.24826, 0.21061] 2.8229116169916955
# Maximum likelihood parameters: [0 0 0.878 1.93 0.565 1.62]
# Maximum likelihood statistics: [1.03 7.23 0.631 26]
#
# case 4: complete sample (no censoring or missing data)
# [-0.0047164, 0.31322, -0.27256, 0.20332, 2.284] 2.6232688303439544
# Maximum likelihood parameters: [0 0 0.989 2.06 0.534 1.6 0.908]
# Maximum likelihood statistics: [1.14 7.36 0.655 25.3 0.908]
#
# case 5: same as 1 + replace censored data by the detection limit
#                     and adjust error to error at detection limit
# [-0.043375, 0.56639, -0.24941, 0.0058382, 1.3669] 3.044143593110131
# Maximum likelihood parameters: [0 0 0.905 3.68 0.563 1.01 0.797]
# Maximum likelihood statistics: [1.06 6.16 0.648 8.25 0.797]
#
#  --- Result of ML parameter estimation (Ndata = 800) ---
# need more basinhops
#
# case 0: proper modeling of missing and censored data
# [-0.00060418, 0.30104, -0.32184, 0.14967, 2.6804] 2.27595827739213
# Maximum likelihood parameters: [0 0 0.999 2 0.477 1.41 0.936]
# Maximum likelihood statistics: [1.12 5.42 0.565 13.6 0.936]
#
# case 1: dropping missing data
# [-0.043825, -0.1532, -0.27946, 0.32951, 2.9829] 1.924364087479337
# Maximum likelihood parameters: [0 0 0.904 0.703 0.525 2.14 0.952]
# Maximum likelihood statistics: [1.04 6.87 0.585 66.9 0.952]
#
# case 2: dropping missing and censored data (complete cases)
# [0.22945, 1.0511, -0.43975, -0.01844, 1.3284] 4.421377507381259
# Maximum likelihood parameters: [0 0 1.7 11.2 0.363 0.958 0.791]
# Maximum likelihood statistics: [1.81 17.8 0.681 21.8 0.791]
#
# case 3: no correlations
# [-0.03163, 0.28737, -0.28403, 0.16957] 2.6654912649924136
# Maximum likelihood parameters: [0 0 0.93 1.94 0.52 1.48]
# Maximum likelihood statistics: [1.06 5.77 0.593 16.2]
#
# case 4: complete sample (no censoring or missing data)
# [-0.008969, 0.23264, -0.31943, 0.18534, 2.1654] 2.512092730827868
# Maximum likelihood parameters: [0 0 0.98 1.71 0.479 1.53 0.897]
# Maximum likelihood statistics: [1.1 5.53 0.558 17 0.897]
#
# case 5: same as 1 + replace censored data by the detection limit
#                     and adjust error to error at detection limit
# [-0.02058, 0.52484, -0.27286, -0.028219, 1.5212] 2.819296682409833
# Maximum likelihood parameters: [0 0 0.954 3.35 0.534 0.937 0.821]
# Maximum likelihood statistics: [1.1 5.19 0.631 6.16 0.821]

print('Example: Partially missing, censored, and correlated data with '
      'observational errors and a lognormal distribution')

dist = scipy.stats.lognorm
Ndata = 200
# Ndata = 800  # convergence test
rho = 0.9
R = np.array([[1., rho], [rho, 1.]])
loc_true = np.array([0., 0.])
scale_true = np.array([1, 2])
shape_true = np.array([0.5, 1.5])

print('population parameters: [{} {} {} {} {} {} {}]'.format(
    *loc_true, *scale_true, *shape_true, rho))

mean_pop = scale_true * np.exp(0.5*shape_true**2) + loc_true
std_pop = (mean_pop - loc_true) * np.sqrt(np.exp(shape_true**2)-1.)
print('population statistics (mean, std, corr): [{:.3g} {:.3g} {:.3g} {:.3g} '
      '{:.3g}]'.format(
    *mean_pop, *std_pop, rho))

## -- create observational data
x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)

rho_x_sample = np.corrcoef(x.T)[0, 1]

y = dist.ppf(scipy.stats.norm.cdf(x),
             shape_true, loc=loc_true, scale=scale_true)
y_true = np.copy(y)
ey = np.zeros_like(y)
sigma_c = [0.1, 0.1]
ey[:, 0] = sigma_c[0]
ey[:, 1] = sigma_c[1] * y_true[:, 1]
y[:, 0] += ey[:, 0] * np.random.randn(Ndata)
y[:, 1] += ey[:, 1] * np.random.randn(Ndata)

ey_full = np.copy(ey)
y_full = np.copy(y)

print('sample statistics full data (mean, std, corr): n={} [{:.3g} {:.3g} '
      '{:.3g} {:.3g} {:.3g}]'.format(
    len(y), np.nanmean(y[:, 0]), np.nanmean(y[:, 1]),
    np.nanstd(y[:, 0]), np.nanstd(y[:, 1]), rho_x_sample))

# censor non-missing data (variable 2)
ceny = np.zeros(Ndata, dtype=bool)
limy = np.zeros(Ndata)
sel = (y[:, 1] < 2) & (~np.isnan(y[:, 0])) # 0.4*mean_pop[1]
limy[sel] = 2 # 0.4*mean_pop[1]
ceny[sel] = True
y[sel, 1] = 0.

# missing data:
# complicated missingness:
# - data with y < 2 is never missing
# - data with y >=2 is missing s.t. lower y values mare missing more often
# - but data is still missing at random (MAR)!
def logistic(x):
    res = np.ones_like(x)
    sel = x < 20
    res[sel] = np.exp(x[sel]) / (np.exp(x[sel]) + 1.)
    return res
m0 = (scipy.stats.bernoulli.rvs(logistic(2-0.2*y[:, 1])).astype(bool)
      & (y[:, 1]>limy))  # for col 0
y[m0, 0] = np.float('NaN')

print('sample statistics (w/ missing as NaN): n={} [{:.3g} {:.3g} {:.3g} {:.3g} '
      'N/A]'.format(
    len(y), np.nanmean(y[:, 0]), np.nanmean(y[:, 1]),
    np.nanstd(y[:, 0]), np.nanstd(y[:, 1])))

# complete cases + censored = don't contain NaN's
ycc = y[np.all(~np.isnan(y), axis=1)]
eycc = ey[np.all(~np.isnan(y), axis=1)]
cenycc = ceny[np.all(~np.isnan(y), axis=1)]
limycc = limy[np.all(~np.isnan(y), axis=1)]

print('sample statistics (without any NaNs): n={} [{:.3g} {:.3g} {:.3g} {:.3g}'
    ' {:.3g}]'.format(
    len(ycc), np.nanmean(ycc[:, 0]), np.nanmean(ycc[:, 1]),
    np.nanstd(ycc[:, 0]), np.nanstd(ycc[:, 1]), np.corrcoef(ycc.T)[0, 1]))

# complete cases, w/o censored
ycc2 = y[np.all(~np.isnan(y), axis=1) & (ceny == False)]
eycc2 = ey[np.all(~np.isnan(y), axis=1) & (ceny == False)]
cenycc2 = ceny[np.all(~np.isnan(y), axis=1) & (ceny == False)]
limycc2 = limy[np.all(~np.isnan(y), axis=1) & (ceny == False)]

print('sample statistics (complete cases]): n={} [{:.3g} {:.3g} {:.3g} {:.3g} '
    '{:.3g}]'.format(
    len(ycc2), np.nanmean(ycc2[:, 0]), np.nanmean(ycc2[:, 1]),
    np.nanstd(ycc2[:, 0]), np.nanstd(ycc2[:, 1]), np.corrcoef(ycc2.T)[0, 1]))

# complete cases, censored data replaced by limits
sel = (cenycc == True)
ycc3 = np.copy(ycc)
ycc3[sel, 1] = limycc[sel]
eycc3 = eycc
eycc3[sel, 1] = sigma_c[1] * limycc[sel]

print('sample statistics (complete cases]): n={} [{:.3g} {:.3g} {:.3g} {:.3g} '
    '{:.3g}]'.format(
    len(ycc3), np.nanmean(ycc3[:, 0]), np.nanmean(ycc3[:, 1]),
    np.nanstd(ycc3[:, 0]), np.nanstd(ycc3[:, 1]), np.corrcoef(ycc3.T)[0, 1]))

ncc = np.sum(np.all(~np.isnan(y), axis=1) & (ceny == False))
ncen = np.sum(ceny == True)
nic = np.sum(np.any(np.isnan(y), axis=1))
ncm = np.sum(np.all(np.isnan(y), axis=1))
print('{} total cases, {} complete, {} censored, {} incomplete, {} completely '
      'missing'.format(Ndata, ncc, ncen, nic, ncm))

for iirun, irun in enumerate([4]):  # enumerate([0, 1, 2, 3, 4, 5]):

    correlated = True
    if irun == 0:
        print('--- Using all data (incl. censored & missing) ---')
        ly = y
        ley = ey
        lceny = ceny
        llimy = limy
        ML_result = np.array([-0.023716, 0.29896, -0.28695, 0.19859, 2.6028])
        ylim_xhist = [0, 1.2]
        xlim_yhist = [0, 1.2]
        ylim_total = [0.1, 500]
        label = 'Proper data handling'
    elif irun == 1:
        print('--- Exclude missing data ---')
        ly = ycc
        ley = eycc
        lceny = cenycc
        llimy = limycc
        ML_result = np.array([-0.064101, -0.08104, -0.24516, 0.35151, 3.2766])
        ylim_xhist = [0, 1.2]
        xlim_yhist = [0, 1.]
        ylim_total = [0.1, 500]
        label = 'no missing data'
    elif irun == 2:
        print('--- Exclude missing and censored data ---')
        ly = ycc2
        ley = eycc2
        lceny = cenycc2
        llimy = limycc2
        ML_result = np.array([0.2168, 1.1177, -0.4228, 0.019954, 1.6785])
        ylim_xhist = [0, 1.2]
        xlim_yhist = [0, 1.]
        ylim_total = [0.1, 500]
        label = 'no missing and censored data'
    elif irun == 3:
        print('--- Using all data but no correlations ---')
        ly = y
        ley = ey
        lceny = ceny
        llimy = limy
        correlated = False
        ML_result = np.array([-0.056388, 0.28618, -0.24826, 0.21061])
        ylim_xhist = [0, 1.2]
        xlim_yhist = [0, 1.2]
        ylim_total = [0.1, 500]
        label = 'no data correlations'
    elif irun == 4:
        print('--- Using full sample (perfect knowledge)')
        ly = y_full
        ley = ey_full
        lceny = None
        llimy = None
        ML_result = np.array([-0.0047164, 0.31322, -0.27256, 0.20332, 2.284])
        ylim_xhist = [0, 1.2]
        xlim_yhist = [0, 1.]
        ylim_total = [0.1, 500]
        label = 'Complete sample'
    elif irun == 5:
        print('--- Exclude missing data & set censored data to limit ---')
        ly = ycc3
        ley = eycc3
        lceny = None
        llimy = None
        ML_result = np.array([-0.043375, 0.56639, -0.24941, 0.0058382, 1.3669])
        ylim_xhist = [0, 1.2]
        xlim_yhist = [0, 3]
        ylim_total = [0.1, 500]
        label = 'no missing data, censored at limit'


    if irun < 4:
        df = pd.DataFrame(np.array([ly[:, 0], ly[:, 1], ley[:, 0], ley[:, 1],
                                    lceny, llimy]).T,
                          columns=['v0', 'v1', 'e_v0', 'e_v1', 'c_v1', 'l_v1'])
    else:
        df = pd.DataFrame(
            np.array([ly[:, 0], ly[:, 1], ley[:, 0], ley[:, 1]]).T,
            columns=['v0', 'v1', 'e_v0', 'e_v1'])

    if showfig:

        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.patches
        import matplotlib.transforms
        plt.ion()

        fig = plt.figure(figsize=(8, 8))
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
        grid.update(wspace=0., hspace=0.)
        main_ax = fig.add_subplot(grid[1:, :-1])
        y_hist = fig.add_subplot(grid[1:, -1], sharey=main_ax)
        x_hist = fig.add_subplot(grid[0, :-1], sharex=main_ax)
        x_hist.xaxis.set_tick_params(labelleft=False)
        y_hist.yaxis.set_tick_params(labelbottom=False)

        # x histogram
        plt.sca(x_hist)
        ls = []
        hs = []
        sel = ~np.isnan(ly[:, 0])
        hist, bin_edges = np.histogram(ly[sel, 0], bins=np.linspace(-1, 4, 20),
                                       density=True)
        bin_center = 0.5*(bin_edges[1:] + bin_edges[:-1])
        h, = plt.step(bin_edges[1:], hist, color='steelblue', alpha=0.8, lw=2)
        hs.append(h)
        ls.append('binned observations')
        plt.fill_between(bin_edges[1:], hist, step='pre', color='b', alpha=0.1)
        xx = np.linspace(-1, 5, 200)
        h, = plt.plot(
            xx, scipy.stats.lognorm.pdf(
                xx, shape_true[0], scale=scale_true[0]),
                '--k', alpha=0.5, lw=2)
        hs.append(h)
        ls.append('true distribution')
        h, = plt.plot(xx, scipy.stats.lognorm.pdf(
            xx, 10**ML_result[2], scale=10**ML_result[0]),
            color='lightcoral', lw=2)
        hs.append(h)
        ls.append('ML estimate')
        plt.ylabel('pdf', fontsize=fontsize)
        plt.yticks([0, 0.5, 1., 1.5, 2., 2.5, 3.],
                   labels=['0', '', '1', '', '2', '', '3'])
        plt.ylim(ylim_xhist)
        plt.tick_params(axis='both', which='major', direction='inout',
                        labelsize=fontsize-2)
        plt.tick_params(axis='both', which='minor', direction='inout',
                        labelsize=fontsize-6)
        plt.minorticks_on()
        plt.legend(hs, ls, loc='upper right', markerfirst=False)

        # y histogram
        plt.sca(y_hist)
        bins=np.logspace(-2, 3, 20)
        sel = ~np.isnan(ly[:, 1]) & (ly[:, 1] > bins[0]) & (ly[:, 1] < bins[-1])
        hist, bin_edges = np.histogram(ly[sel, 1], bins=bins, density=True)
        bin_center = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        plt.step(hist * bin_center * np.log(10.), bin_edges[:-1],
                 color='steelblue', alpha=0.8, lw=2)
        plt.fill_between(hist * bin_center * np.log(10.), bin_edges[:-1],
                         step='pre', color='b', alpha=0.1)
        xx = np.logspace(-1, 4, 100)
        plt.plot(
            scipy.stats.lognorm.pdf(xx, shape_true[1], scale=scale_true[1])
                * xx * np.log(10.), xx,
            '--k', alpha=0.5, lw=2)
        plt.plot(
            scipy.stats.lognorm.pdf(
                xx, 10**ML_result[3], scale=10**ML_result[1])
                    * xx * np.log(10.), xx, color='lightcoral', lw=2)

        plt.xlabel('pdf', fontsize=fontsize)
        plt.xticks([0, 0.5, 1., 1.5, 2., 2.5, 3.],
                   labels=['0', '', '1', '', '2', '', '3'])
        plt.xlim(xlim_yhist)
        plt.tick_params(axis='both', which='major', direction='inout',
                        labelsize=fontsize-2)
        plt.tick_params(axis='both', which='minor', direction='inout',
                        labelsize=fontsize-6)
        plt.minorticks_on()

        # scatter plot
        plt.sca(main_ax)

        if key_left:
            align = 'left'
            align_x = 0.02
        else:
            align = 'right'
            align_x = 0.98

        ls = []
        hs = []

        Nobs = len(ly)
        Nmis = np.sum(np.isnan(ly[:, 0]))
        if lceny is not None:
            Ncen = np.sum(lceny)
        else:
            Ncen = 0

        plt.text(align_x, 0.96,
                 r'$N_{{\rm tot}}={}, N_{{\rm miss}}={}, N_{{\rm cen}}={}$'.format(
                 Nobs, Nmis, Ncen), fontsize=fontsize-4, color='k',
                 horizontalalignment=align, transform=plt.gca().transAxes)
        plt.text(align_x, 0.915, 'PDF(x) ~ lognorm(x, {})'.format(
            shape_true[0]), fontsize=fontsize-6,
            color='k', horizontalalignment=align, transform=plt.gca().transAxes)
        plt.text(align_x, 0.875, 'PDF(y) ~ lognorm(y/2, {})/2'.format(
            shape_true[1]), fontsize=fontsize-6,
            color='k', horizontalalignment=align, transform=plt.gca().transAxes)
        plt.text(align_x, 0.835, r'data correlation = {}'.format(
            rho), fontsize=fontsize-6,
            color='k', horizontalalignment=align, transform=plt.gca().transAxes)
        plt.text(align_x, 0.795, r'uncorrelated errors', fontsize=fontsize-6,
            color='k', horizontalalignment=align, transform=plt.gca().transAxes)

        plt.text(0.98, 0.02, label, fontsize=fontsize-2,
            color='k', horizontalalignment='right',
            transform=plt.gca().transAxes)

        # show data with x-NaNs
        sel = np.isnan(ly[:, 0])
        if np.sum(sel) > 0:
            col_text = [0.6, 0., 0.6]
            col = [0.7, 0., 0.7]
            plt.text(4.3, 1., 'missing x', fontsize=fontsize-6, color=col_text,
                     horizontalalignment='right')
            rstate = np.random.get_state()
            for i, s in enumerate(sel):
                if not s:
                    continue
                x_r = 0.1*(np.random.rand()-0.5) + 4.2
                plt.plot(x_r, ly[i, 1], '.', color=col,
                         markersize=3)
            np.random.set_state(rstate)
            plt.annotate("", xy=(4.2+0.2, 1.5), xytext=(4.2-0.2, 1.5),
                         arrowprops=dict(arrowstyle="<|-|>",
                                         color=col))

        # show censored data
        sel = lceny
        nsel = np.sum(sel)
        if nsel and nsel > 0:
            col_text = [0, 0.6, 0.6]
            col = [0, 0.7, 0.7]
            plt.text(1.9, 1.5, 'censored y', fontsize=fontsize-6,
                     color=col_text, horizontalalignment='left')
                     #transform=plt.gca().transAxes)
            for i, s in enumerate(sel):
                if not s:
                    continue
                plt.annotate("", xy=(ly[i, 0], llimy[i]), xytext=(ly[i, 0], llimy[i]-1.),
                             arrowprops=dict(arrowstyle="<|-",
                                             color=col))
                plt.plot(ly[i, 0], llimy[i], '.', color=col, markersize=4)

        if lceny is not None:
            sel = ~lceny
        else:
            sel = np.ones(len(ly), dtype=bool)

        for i, s in enumerate(sel):
            if not s:
                continue
            # see https://www.xarg.org/2018/04/
            # how-to-plot-a-covariance-error-ellipse/
            s = -2 * np.log((1-0.6827)/2)  # to show 68% error ellipse (1sigma)
            cov_c = np.diag(ley[i, :]).dot(np.eye(2).dot(np.diag(ley[i, :])))
            evs, evecs = np.linalg.eigh(s * cov_c)  # EV with max eigenvalue
            evec = evecs[:, -1]
            ell=matplotlib.patches.Ellipse(
                (ly[i, 0], ly[i, 1]), height=np.sqrt(evs[0]),
                width=np.sqrt(evs[1]),
                angle=np.arctan2(evec[1], evec[0])*180/np.pi, lw=1)
            plt.gca().add_artist(ell)
            ell.set_facecolor([0.9, 0.9, 1.])
            ell.set_edgecolor('b')
            ell.set_alpha(0.1)

        h, = plt.plot(ly[sel, 0], ly[sel, 1], '.', color='b', alpha=0.8)
        hs.append(h)
        ls.append(r'observed $(x^{\rm obs}, y^{\rm obs})$')

        h, = plt.plot(y_true[:, 0], y_true[:, 1], '.', color='k',
                      markerfacecolor='None', alpha=0.4)
        hs.append(h)
        ls.append(r'true data $(x, y)$')

        # true 2d distribution
        ## -- create observational data
        if analytic_joint_pdf:
            def pdf(x, lgy):
                shape = x.shape
                x = x.flatten()
                lgy = lgy.flatten()
                obs = leopy.Observation({'v0': x, 'v1': 10**lgy}, 'true_pdf',
                                        verbosity=-1)
                like = leopy.Likelihood(obs, p_true='lognorm', p_cond=None,
                                        verbosity=-1)
                return (like.p(loc_true, scale_true,
                               shape_true=shape_true, R_true=R)
                        * 10**lgy[:, None] * np.log(10.)).reshape(shape)
            contour_levels, xc, lgyc, H = auxiliary.compute_confidence_levels_pdf(
                pdf, levels=[0.6827, 0.9545], bins=200, x_range=[-0.5, 4],
                y_range=[-3, 3])
        else:
            z_mock = scipy.stats.multivariate_normal.rvs(cov=R, size=N_mock)
            y_mock = dist.ppf(scipy.stats.norm.cdf(z_mock),
                         shape_true, loc=loc_true, scale=scale_true)
            contour_levels, xc, lgyc, H = auxiliary.compute_confidence_levels(
                y_mock[:, 0], np.log10(y_mock[:, 1]), levels=[0.6827, 0.9545],
                bins=400)

        contour_levels = np.sort(contour_levels)
        plt.contour(xc, 10**lgyc, H.T, levels=contour_levels,
                    linestyles='dashed', colors='k', alpha=0.5,
                    linewidths=2)

        # 2d distribution for ML estimate
        ## -- create observational data
        if correlated:
            ML_rho = leopy.stats.inv_logit(ML_result[4])
        else:
            ML_rho = 0.
        ML_R = np.array([[1., ML_rho], [ML_rho, 1.]])
        if analytic_joint_pdf:
            def pdf(x, lgy):
                shape = x.shape
                x = x.flatten()
                lgy = lgy.flatten()
                obs = leopy.Observation({'v0': x, 'v1': 10**lgy}, 'true_pdf',
                                        verbosity=-1)
                like = leopy.Likelihood(obs, p_true='lognorm', p_cond=None,
                                        verbosity=-1)
                return (like.p([0, 0], 10**ML_result[0:2],
                              shape_true=10**ML_result[2:4], R_true=ML_R)
                        * 10**lgy[:, None] * np.log(10.)).reshape(shape)
            contour_levels, xc, lgyc, H = auxiliary.compute_confidence_levels_pdf(
                pdf, levels=[0.6827, 0.9545], bins=200, x_range=[-0.5, 4],
                y_range=[-3, 3])
        else:
            z_mock = scipy.stats.multivariate_normal.rvs(cov=ML_R, size=N_mock)
            y_mock = dist.ppf(scipy.stats.norm.cdf(z_mock),
                         10**ML_result[2:4], loc=[0, 0],
                         scale=10**ML_result[0:2])
            contour_levels, xc, lgyc, H = auxiliary.compute_confidence_levels(
                y_mock[:, 0], np.log10(y_mock[:, 1]), levels=[0.6827, 0.9545],
                bins=400)

        contour_levels = np.sort(contour_levels)
        plt.contour(xc, 10**lgyc, H.T, levels=contour_levels,
                    linestyles='solid', colors='lightcoral',
                    linewidths=2)

        plt.yscale('log')
        plt.ylim(ylim_total)
        plt.xlim([-0.1, 4.5])

        plt.xlabel('x', fontsize=fontsize)
        plt.ylabel('y', fontsize=fontsize)
        plt.yticks([1, 10, 100], labels=['1', '10', '100'])

        plt.tick_params(axis='both', which='major', direction='inout',
                        labelsize=fontsize-2)
        plt.tick_params(axis='both', which='minor', direction='inout',
                        labelsize=fontsize-6)
        plt.minorticks_on()
        plt.tight_layout()

        plt.legend(hs, ls, loc='upper right', markerfirst=False,
                   handletextpad=-0.1)

        if savefig:
            plt.savefig('joint_probability_{}.pdf'.format(irun))

    if optimize:

        obs = leopy.Observation(df, 'joint probability')

        ## -- set up Likelihood and find maximum likelihood parameters
        like = leopy.Likelihood(obs, p_true='lognorm', p_cond='norm',
                                verbosity=0)

        if correlated:
            def f_mlnlike(x, *args):

                if np.any(np.isnan(x)):
                    return 1000.

                df = args[0]
                Nobs = df.shape[0]

                if 0:
                    loc_true = x[0:2]
                    scale_true = 10**x[2:4]
                    shape_true = 10**x[4:6]
                    rho = leopy.stats.inv_logit(x[6])
                else:
                    loc_true = [0, 0]
                    scale_true = 10**x[0:2]
                    shape_true = 10**x[2:4]
                    rho = leopy.stats.inv_logit(x[4])

                R = np.array([[1., rho], [rho, 1.]])
                pp = like.p(loc_true, scale_true, shape_true=shape_true, R_true=R)
                if np.sum(pp==0) > 0:
                    return 1000.
                else:
                    return -np.sum(np.log(pp))/Nobs

            # bounds = scipy.optimize.Bounds(
            #    [-5, -5, -2., -2., -2., -2., -7.],
            #    [5, 5, 2., 2., 2., 2., 7.])
            bounds = scipy.optimize.Bounds(
                [-2., -2., -2., -2., -7.],
                [2., 2., 2., 2., 7.])
            #start = [0., 0., 0., 0., 0., 0., 0.]
            start = [
                np.log10(1.), np.log10(2.), np.log10(0.5), np.log10(1.5), 2.2]

        else:
            def f_mlnlike(x, *args):

                if np.any(np.isnan(x)):
                    return 1000.

                df = args[0]
                Nobs = df.shape[0]

                if 0:
                    loc_true = x[0:2]
                    scale_true = 10**x[2:4]
                    shape_true = 10**x[4:6]
                else:
                    loc_true = [0, 0]
                    scale_true = 10**x[0:2]
                    shape_true = 10**x[2:4]
                pp = like.p(loc_true, scale_true, shape_true=shape_true)
                if np.sum(pp==0) > 0:
                    return 1000
                else:
                    return -np.sum(np.log(pp))/Nobs

            bounds = scipy.optimize.Bounds(
                [-2., -2., -2., -2.],
                [2., 2., 2., 2.])

            #tart = [0., 0., 0., 0.]
            start = [np.log10(1.), np.log10(2.), np.log10(0.5), np.log10(1.5)]

        print('Running ML optimization - This may take a little bit ...')
        import auxiliary

        T = 0.01
        my_print_fun = auxiliary.MyPrintFun()
        my_take_step = auxiliary.MyTakeStep(stepsize=1)
        minimizer_options = {'disp': True, 'ftol': 1e-12}
        minimizer_kwargs = {'method': 'SLSQP', 'bounds': bounds,
                            'options': minimizer_options, 'args': (obs.df,)}

        optres = scipy.optimize.basinhopping(
            f_mlnlike, start, niter=20, T=T, interval=3,
            minimizer_kwargs=minimizer_kwargs, take_step=my_take_step,
            callback=my_print_fun)

        print('[', end='')
        print(', '.join(map(lambda _: '{:.5g}'.format(_), optres.x)), end='')
        print('] ', end='')
        print('{}'.format(optres.fun))

        if 0:
            loc_opt = optres.x[0:2]
            scale_opt = 10**optres.x[2:4]
            shape_opt = 10**optres.x[4:6]
            if correlated:
                rho_opt = leopy.stats.inv_logit(optres.x[6])
        else:
            loc_opt = [0, 0]
            scale_opt = 10**optres.x[0:2]
            shape_opt = 10**optres.x[2:4]
            if correlated:
                rho_opt = leopy.stats.inv_logit(optres.x[4])

        print('Maximum likelihood parameters: '
              '[{:.3g} {:.3g} {:.3g} {:.3g} {:.3g} {:.3g}'.format(
                *loc_opt, *scale_opt, *shape_opt))
        if correlated:
            print(' {:.3g}]'.format(rho_opt))
        else:
            print(']')

        mean_opt = scale_opt * np.exp(0.5*shape_opt**2) + loc_opt
        std_opt = (mean_opt - loc_opt) * np.sqrt(np.exp(shape_opt**2)-1.)
        print('Maximum likelihood statistics: '
              '[{:.3g} {:.3g} {:.3g} {:.3g}'.format(*mean_opt, *std_opt))
        if correlated:
            print(' {:.3g}]'.format(rho_opt))
        else:
            print(']')
