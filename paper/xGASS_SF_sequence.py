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

import statsmodels.api as sm
import statsmodels.formula.api as smf

import auxiliary
import xGASS_auxiliary

savefig = True

np.random.seed(6)

fontsize=17

if 0:
    sel_lgMstar = [9, 10]
    fit_result = [0.51043, -0.83939, 0.14565, 0.7706, 0.75496, -1.0976,
                  1.2514, -0.72357, -1.6778]

if 0:
    sel_lgMstar = [9, 10.5]
    fit_result = [0.6964, -0.16138, -0.12652, 0.12532, 1.2754, -0.56014,
                  1.1055, -1.0889, -4.5444]

if 1:
    sel_lgMstar = [9, 11]
    fit_result = [0.61447, -0.1615, -0.098252, 0.09682, 1.053, -0.87677,
                  1.1217, -1.3584, -5.4383]

if 0:
    sel_lgMstar = [9, 11.5]
    fit_result = [0.61763, -0.1453, -0.18351, 0.046835, 0.86896, -1.1914,
                  1.1026, -1.3548, -5.6141]

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches
plt.ion()

print('Plotting the SFR -- Mstar relation in the xGASS data set')

# -- Step 1. Read the data
catalog_filename = 'xGASS.csv'

df = pd.read_csv(catalog_filename, comment='#', na_values='.')
df = df.rename(columns={
    'lgMstar': 'v0', 'SFR': 'v1',
    'e_SFR': 'e_v1', 'SFR_censored': 'c_v1', 'SFR_limit': 'l_v1'})

Ndata = df.shape[0]
dist = scipy.stats.gamma

# -- Step 2. Compute the SFR - Mstar relation
lx = 'v0'
xlabel = r'$\lg M_{\rm star}$'
ylabel = r'SFR'
savelabel = 'SF_sequence_{}_{}'.format(*sel_lgMstar)
missing_y = 33
ylim = [1e-2, 150]

ly = 'v{}'.format(1)
lsy = 'e_{}'.format(ly)
lcy = 'c_{}'.format(ly)
lly = 'l_{}'.format(ly)

x, y, sy = df[[lx, ly, lsy]].to_numpy().T
if lly in df:
    ceny = df[lcy].to_numpy().T
    limy = df[lly].to_numpy().T
sx = 0.01 * np.ones_like(x)

xx = np.linspace(max(min(x), sel_lgMstar[0]), min(max(x), sel_lgMstar[1]), 100)

m_scale, n_scale = xGASS_auxiliary.get_slope_intercept(
    fit_result[0], fit_result[1])
m_shape, n_shape = xGASS_auxiliary.get_slope_intercept(
    fit_result[2], fit_result[3])
m_zero, n_zero = xGASS_auxiliary.get_slope_intercept(
    fit_result[4], fit_result[5])
loc_xx = 0
scale_xx = 10**(m_scale * (xx-10) + n_scale)
shape_xx = 10**(m_shape * (xx-10) + n_shape)
mu_xx = dist.mean(shape_xx, loc=loc_xx, scale=scale_xx)
scatter_up_xx, scatter_down_xx = xGASS_auxiliary.calc_scatter(shape_xx, r=1)

print('Slope of lgSFR - lgMstar relation (excl. zeros) is {:.4g}'.format(
    m_shape + m_scale))
print('Offset of lgSFR - lgMstar relation (excl. zeros) is {:.4g}'.format(
    n_shape + n_scale))

df_scatter = pd.DataFrame({'scatter': scatter_up_xx, 'lg_M10': (xx-10)})
res_scatter = smf.ols('scatter ~ lg_M10', data=df_scatter).fit()
print(res_scatter.params)

plt.figure(figsize=(6, 6))
hs = []
ls = []

# -- Step 3. Plot data and errors
if lly in df:
    sel = np.logical_and(~np.isnan(y), ceny!=True )
else:
    sel = ~np.isnan(y)
plt.errorbar(x[sel], y[sel], sy[sel], fmt='none', ecolor='b', alpha=0.1,
             capthick=2, capsize=2)
h, = plt.plot(x[sel], y[sel], '.', color='b', alpha=0.4)

ML_color = 'lightcoral'
# ML_color = 'seagreen'
h, = plt.plot(xx, mu_xx, '-', color=ML_color, lw=2)
hs.append(h)
ls.append(r'peak position')

plt.plot(xx, mu_xx * 10**scatter_up_xx, '--', color=ML_color)
h, = plt.plot(xx, mu_xx * 10**(-scatter_down_xx), '--', color=ML_color)
hs.append(h)
ls.append(r'$\Delta_{+/-}$')

# show (non-censored) data with y-NaNs (lgMstar values are complete)
if lly in df:
    sel = np.logical_and(np.isnan(y), ceny!=True )
else:
    sel = np.isnan(y)
if np.sum(sel) > 0:
    col_text = [0.6, 0., 0.6]
    col = [0.7, 0., 0.7]
    for i, s in enumerate(sel):
        if not s:
            continue
        y_r = missing_y * 10**(0.1*(np.random.rand()-0.5))
        plt.plot(x[i], y_r, '.', color=col,
                 markersize=3)
    plt.text(0.02, 0.78, 'missing SFR', fontsize=fontsize-6, color=col_text,
             horizontalalignment='left', transform=plt.gca().transAxes)

# show censored data
if lly in df:
    sel = (ceny==True)
    nsel = np.sum(sel)
else:
    sel = np.isnan(y)
    nsel = 0
if nsel > 0:
    col_text = [0, 0.6, 0.6]
    col = [0, 0.7, 0.7]
    for i, s in enumerate(sel):
        if not s:
            continue
        h=plt.annotate("", xy=(x[i], limy[i]), xytext=(x[i], ylim[0]),
                       arrowprops=dict(arrowstyle="<|-",
                                       color=col, alpha=0.2, lw=2))
        plt.plot(x[i], limy[i], '.', color=col, markersize=4)
    plt.text(0.02, 0.06, 'censored SFR', fontsize=fontsize-6,
             color=col_text, horizontalalignment='left',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))

plt.ylim(ylim)
plt.yscale('log')

s = 'xGASS'
plt.text(0.02, 0.975, s, fontsize=fontsize-4, color='k',
         horizontalalignment='left', verticalalignment='top',
         transform=plt.gca().transAxes)

s = (r'$\lg{{}}\,{{\rm SFR}}_{{\rm MS}} = '
     r'{:.2f}\,\lg{{}}\,(M_{{\rm star}}/10^{{10}} M_\odot) '
     r'{:+.2f}$'.format(m_shape + m_scale, n_shape + n_scale))
plt.text(0.98, 0.98, s, fontsize=fontsize-5, color='k',
horizontalalignment='right', verticalalignment='top',
transform=plt.gca().transAxes)

s = r'scatter ($\Delta{{}}_+$) $\sim{{}} {:.2f}$ dex'.format(
    res_scatter.params['Intercept'])
plt.text(0.98, 0.935, s, fontsize=fontsize-4, color='k',
horizontalalignment='right', verticalalignment='top',
transform=plt.gca().transAxes)

plt.xlabel(xlabel, fontsize=fontsize)
plt.ylabel(ylabel, fontsize=fontsize)

plt.yticks([1e-2, 1e-1, 1, 10, 100], labels=['0.01', '0.1', '1', '10', '100'])

plt.tick_params(axis='both', which='major', direction='inout',
                labelsize=fontsize-2)
plt.tick_params(axis='both', which='minor', direction='inout',
                labelsize=fontsize-6)
plt.minorticks_on()
plt.tight_layout()

loc_legend = 'lower right'
if 'right' in loc_legend:
    markerfirst = False
else:
    markerfirst = True
plt.legend(hs, ls, loc=loc_legend, markerfirst=markerfirst)

if savefig:
    plt.savefig('xGASS_{}.pdf'.format(savelabel))

# print table
# m_zero, n_zero, m_SFR, n_SFR, m_scatter, n_scatter
print('{:.2g} & {:.2g} & '.format(
    m_shape + m_scale, n_shape + n_scale), end='')
print('{:.2g} & {:.2g} & '.format(
    res_scatter.params['lg_M10'], res_scatter.params['Intercept']),
    end='')
print('{:.3g} & {:.3g}\n'.format(m_zero, n_zero), end='')
