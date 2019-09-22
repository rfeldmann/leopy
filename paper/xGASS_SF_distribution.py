"""Analysis of the xGASS data set.

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
import scipy.integrate

import xGASS_auxiliary

import sys
sys.path.append("..") # Adds higher directory to python modules path.
import leopy.stats

savefig = True

model_color = 'lightcoral'
model_alpha = 0.8
model_ls = '--'

f_detection = 2
fs = r'{:g}\times'.format(f_detection)

fontsize = 17

print('SFRs distribution in xGASS relative to SF sequence')

# -- Step 1. Read the data
catalog_filename = 'xGASS.csv'

df = pd.read_csv(catalog_filename, comment='#', na_values='.')
df = df.rename(columns={
    'lgMstar': 'v0', 'SFR': 'v1',
    'e_SFR': 'e_v1', 'SFR_censored': 'c_v1', 'SFR_limit': 'l_v1'})

ylim = [0, 1]
if 0:
    sel_lgMstar = [9, 10]
    fit_result = [0.51043, -0.83939, 0.14565, 0.7706, 0.75496, -1.0976,
                  1.2514, -0.72357, -1.6778]
    ylim = [0, 1.7]

if 0:
    sel_lgMstar = [9, 10.5]
    fit_result = [0.6964, -0.16138, -0.12652, 0.12532, 1.2754, -0.56014,
                  1.1055, -1.0889, -4.5444]
    ylim = [0, 1.2]

if 1:
    sel_lgMstar = [9, 11]
    fit_result = [0.61447, -0.1615, -0.098252, 0.09682, 1.053, -0.87677,
                  1.1217, -1.3584, -5.4383]

if 0:
    sel_lgMstar = [9, 11.5]
    fit_result = [0.61763, -0.1453, -0.18351, 0.046835, 0.86896, -1.1914,
                  1.1026, -1.3548, -5.6141]

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches
plt.ion()

lx = 'v0'
savelabel = '_{}_{}_fs{}'.format(*sel_lgMstar, f_detection)

ly = 'v{}'.format(1)
lsy = 'e_{}'.format(ly)
lcy = 'c_{}'.format(ly)
lly = 'l_{}'.format(ly)

#np.random.seed(2)
#df = df.sample(frac=0.3)

df = df[(df[lx]>=sel_lgMstar[0]) & (df[lx]<=sel_lgMstar[1])]
x, y, sy, limy = df[[lx, ly, lsy, lly]].to_numpy().astype(float).T
ceny = df[lcy].to_numpy().astype(bool)

xx = np.linspace(min(x), max(x), 100)

m_scale, n_scale = xGASS_auxiliary.get_slope_intercept(fit_result[0], fit_result[1])
loc = loc_xx = 0
scale = 10**(m_scale * (x-10) + n_scale)
scale_xx = 10**(m_scale * (xx-10) + n_scale)

m_shape, n_shape = xGASS_auxiliary.get_slope_intercept(fit_result[2], fit_result[3])
shape = 10**(m_shape * (x-10) + n_shape)
shape_xx = 10**(m_shape * (xx-10) + n_shape)

m_zero, n_zero = xGASS_auxiliary.get_slope_intercept(fit_result[4], fit_result[5])
f_zero = leopy.stats.inv_logit(m_zero * (x-10) + n_zero)
f_zero_xx = leopy.stats.inv_logit(m_zero * (xx-10) + n_zero)

# mean and percentiles of main component
mu = scipy.stats.gamma.mean(shape, loc=loc, scale=scale)
mu_xx = scipy.stats.gamma.mean(shape_xx, loc=loc_xx, scale=scale_xx)

sel = (~np.isnan(y) | ceny)
x = x[sel]
y = y[sel]
mu = mu[sel]
sy = sy[sel]
ceny = ceny[sel]
limy = limy[sel]
Ndata = np.sum(sel)

plt.figure(figsize=(6, 6))
hs = []
ls = []

density = True
# the SFR values below the detection limit (SFR < f * e(SFR)) are not
# believable. Show only fraction of galaxies with SFR < f*e(SFR)
sel_SFR = (~ceny) & (y >= f_detection * sy)
hist, bin_edges = np.histogram(
    np.log10(y[sel_SFR]/mu[sel_SFR]),
    bins='auto', density=density)
KS_rvs = np.log10(y[sel_SFR]/mu[sel_SFR])

bin_center = 0.5*(bin_edges[1:] + bin_edges[:-1])
hist *= np.sum(sel_SFR)/len(sel_SFR)
plt.step(bin_center, hist, where='mid', color='steelblue', alpha=0.8, lw=2)
plt.fill_between(bin_center, hist, step='mid', color='b', alpha=0.1)

plt.bar(-3., height = 1.-np.sum(sel_SFR)/len(sel_SFR), width=1.,
        color='steelblue', alpha=0.8)
plt.text(-2.9, 1.-np.sum(sel_SFR)/len(sel_SFR)+0.03,
         'xGASS:\n ${\\rm SFR}<'+fs+'\\Delta{}SFR$', fontsize=fontsize-9,
         color='steelblue', horizontalalignment='center')
# hs.append((plt.Rectangle((0,0),0.1,0.1,color='b', alpha=0.2),
#            matplotlib.lines.Line2D((0,0),(1,1),color='steelblue',
#            alpha=0.8)))
hs.append(matplotlib.lines.Line2D(
    (0,0), (1,1), color='steelblue', alpha=0.8, lw=2))
ls.append(r'xGASS: ${\rm SFR}\geq'+fs+r'\Delta{}{\rm SFR}$')

# generate mock sample from fit
Nmock = 1000000
np.random.seed(1)
x_mock = np.random.choice(x, size=(Nmock))

shape_mock = 10**(m_shape * (x_mock-10) + n_shape)
scale_mock = 10**(m_scale * (x_mock-10) + n_scale)
f_zero_mock = leopy.stats.inv_logit(m_zero * (x_mock-10) + n_zero)
# next line is same as mu_mock = shape_mock * scale_mock
mu_mock = scipy.stats.gamma.mean(shape_mock, loc=0, scale=scale_mock)

scale_zero = 1e-2

out_scale = 10**fit_result[6]
out_shape = 10**fit_result[7] * np.log(10)
out_frac = leopy.stats.inv_logit(fit_result[8])

# -- Create mock data

leopy.stats.zi_gamma_lognorm.set_return_index(True)
y_mock, halfnorm_index, lognorm_index = leopy.stats.zi_gamma_lognorm.rvs(
    shape_mock, scale_zero, f_zero_mock, out_shape, out_scale, out_frac,
    scale=scale_mock)
leopy.stats.zi_gamma_lognorm.set_return_index(False)

y_true_mock = np.copy(y_mock)

# compute SFR uncertainty
e_y_mock = xGASS_auxiliary.SFR_error(x_mock, y_mock)

# add errors
error_y_mock = scipy.stats.norm.rvs(scale=e_y_mock)
y_mock += error_y_mock

# -- Plot 'observed' mock data

def mock_pdf(x, normalize=True):
    """PDF of mock SFRs w/ errors relative to SF sequence."""
    sel_SFR_mock = (y_mock >= f_detection * e_y_mock)
    fsel = np.sum(sel_SFR_mock)/len(sel_SFR_mock)
    hist, bin_edges = np.histogram(
        np.log10(y_mock[sel_SFR_mock]/mu_mock[sel_SFR_mock]),
        np.linspace(-5.5, 2., 500), density=True)
    bin_center = 0.5*(bin_edges[1:] + bin_edges[:-1])
    hist = np.convolve(hist, np.ones((5,))/5., mode='same')
    if not normalize:
        hist *= fsel
    return np.interp(x, bin_center, hist), fsel

def mock_cdf(x, normalize=True):
    """CDF of mock SFRs w/ errors relative to SF sequence."""
    sel_SFR_mock = (y_mock >= f_detection * e_y_mock)
    fsel = np.sum(sel_SFR_mock)/len(sel_SFR_mock)
    hist, bin_edges = np.histogram(
        np.log10(y_mock[sel_SFR_mock]/mu_mock[sel_SFR_mock]),
        np.linspace(-5.5, 2., 500), density=True)
    bin_center = 0.5*(bin_edges[1:] + bin_edges[:-1])
    hist = np.convolve(hist, np.ones((5,))/5., mode='same')
    add = 0
    if not normalize:
        hist *= fsel
        add = (1-fsel)
    cum_hist = scipy.integrate.cumtrapz(hist, bin_center, initial=0) + add
    return np.interp(x, bin_center, cum_hist)

# naive KS test on non-zero component
p = scipy.stats.kstest(KS_rvs, mock_cdf)[1]
plt.text(0.02, 0.76, r'xGASS vs Mock: $p_{{\rm KS}} = {:.2f}$'.format(p),
         fontsize=fontsize-6, horizontalalignment='left',
         verticalalignment='center', transform=plt.gca().transAxes)

max_pdf = np.max(hist)
xx = np.linspace(-5.5, 2., 500)
pdf, fsel = mock_pdf(xx, False)
plt.plot(xx, pdf, color='mediumseagreen', lw=2, alpha=.9, ls='-.')
import matplotlib.lines
hs.append(matplotlib.lines.Line2D((0,0),(1,1), color='seagreen', ls='-.', lw=2,
                                  alpha=0.8))
ls.append('Mock: SFR$\geq'+fs+'\\Delta{}SFR$')
plt.bar(-4.5, height = 1.-fsel, width=1.,
        alpha=0.5, color='seagreen')
plt.text(-4.6, 1.-fsel+0.035,
         'Mock:\n ${\\rm SFR}<'+fs+'\\Delta{}SFR$', fontsize=fontsize-9,
         color='seagreen', horizontalalignment='center')

# -- plot ML estimates
if 0:
    # select mock objects that are in the gamma component
    sel_SFR_mock = np.logical_and(np.logical_not(lognorm_index),
                                  np.logical_not(halfnorm_index))
if 1:
    # select mock objects that are in the gamma or starburst component
    sel_SFR_mock = np.logical_not(halfnorm_index)

hist, bin_edges = np.histogram(
    np.log10(y_true_mock[sel_SFR_mock]/mu_mock[sel_SFR_mock]),
    np.linspace(-5.5, 2., 200), density=density)
bin_center = 0.5*(bin_edges[1:] + bin_edges[:-1])
hist = np.convolve(hist, np.ones((5,))/5., mode='same')
hist *= np.sum(sel_SFR_mock)/len(sel_SFR_mock)
plt.plot(bin_center, hist, model_ls, color=model_color, lw=2, alpha=model_alpha)
sel_SFR_mock = np.logical_and(np.logical_not(lognorm_index), halfnorm_index)
plt.bar(-6, height = np.sum(sel_SFR_mock)/len(sel_SFR_mock), width=1.,
        alpha=model_alpha, color=model_color)
plt.text(-6, np.sum(sel_SFR_mock)/len(sel_SFR_mock)+0.035,
         'Model:\nSFR=0', fontsize=fontsize-9, color='k',
         horizontalalignment='center', alpha=0.5)
hs.append(matplotlib.lines.Line2D((0,0),(1,1), ls=model_ls, color=model_color,
                                  alpha=model_alpha))
ls.append(r'Model: SFR$>0$')

if 0:
    # select mock objects that are in lognormal component
    sel_SFR_mock = lognorm_index.astype(bool)
    hist, bin_edges = np.histogram(
        np.log10(y_true_mock[sel_SFR_mock]/mu_mock[sel_SFR_mock]),
        np.linspace(-5.5, 2., 200), density=density)
    bin_center = 0.5*(bin_edges[1:] + bin_edges[:-1])
    hist = np.convolve(hist, np.ones((5,))/5., mode='same')
    hist *= np.sum(sel_SFR_mock)/len(sel_SFR_mock)
    plt.plot(bin_center, hist, ':', color=model_color, lw=2, alpha=model_alpha)

    hs.append(matplotlib.lines.Line2D((0,0),(1,1), ls=':', color=model_color,
                                      alpha=model_alpha))
    ls.append(r'Model: starbursts')


# s = r'GASS: $N_{{\rm tot}}={}$'.format(Ndata)
s = 'xGASS'
s = s + '\n${}\\leq{{}}\\lg{{}}\,M_{{\\rm star}}/M_\\odot\\leq{{}}{}$'.format(
    *sel_lgMstar)
plt.text(0.98, 0.98, s, fontsize=fontsize-4, color='k',
         horizontalalignment='right', verticalalignment='top',
         transform=plt.gca().transAxes)

plt.xlabel(r'$\lg{}$ SFR / SFR$_{\rm MS}$', fontsize=fontsize)
plt.ylabel('pdf', fontsize=fontsize)

plt.tick_params(axis='both', which='major', direction='inout',
                labelsize=fontsize-2)
plt.tick_params(axis='both', which='minor', direction='inout',
                labelsize=fontsize-6)
plt.minorticks_on()
plt.tight_layout()

plt.ylim(ylim)
plt.xlim([-6.8, 2.])

loc_legend = 'upper left'
if 'right' in loc_legend:
    markerfirst = False
else:
    markerfirst = True
plt.legend(hs, ls, loc=loc_legend, markerfirst=markerfirst)

if savefig:
    plt.savefig('xGASS_SF_distribution{}.pdf'.format(savelabel))
