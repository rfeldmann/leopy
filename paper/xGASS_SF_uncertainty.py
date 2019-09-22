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
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

plt.ion()

savefig = True

fontsize = 16

print('Uncertainty of SFRs in xGASS')

df = pd.read_csv('xGASS.csv', comment='#', na_values='.')

plt.figure(figsize=(6, 6))

# excl. galaxies without SFR uncert.
sel = ((df['SFR_case'] == 1) | (df['SFR_case'] == 2)) & (df['e_SFR'] > 5e-3)
df = df.loc[sel]

# drop entries with NaN in SFR, e_SFR, or Mstar
sel = np.any(np.isnan(df[['SFR', 'lgMstar', 'e_SFR']]), axis=1)
df = df.loc[~sel]

df['lg_SFR'] = np.log10(df['SFR'])
df['lg_e_SFR'] = np.log10(df['e_SFR'])
df['lg_Mstar10'] = df['lgMstar'] - 10.

mod = smf.ols(formula='lg_e_SFR ~ lg_SFR * lg_Mstar10', data=df)
res = mod.fit()

print(res.summary())

hs = []
hls = []

xx = np.logspace(-2.5, 1.5, 100)
h, = plt.plot(xx, xx, '--k')
hs.append(h)
hls.append('1:1')

lgMstar_bins = np.linspace(9, 11.5, 6)
colors = ['b','c', 'g', 'm', [1., 0, 0]]
for i in reversed(range(len(lgMstar_bins)-1)):
    lgMstar_min = lgMstar_bins[i]
    lgMstar_max = lgMstar_bins[i+1]
    sel = (df['lgMstar'] >= lgMstar_min) & (df['lgMstar'] < lgMstar_max)
    lgMstar_avg = np.mean(df.loc[sel, 'lgMstar'])
    df.loc[sel, 'lgMstar'] = lgMstar_avg
    df.loc[sel, 'lg_Mstar10'] = lgMstar_avg - 10
    h, = plt.plot(df.loc[sel, 'SFR'], df.loc[sel,'e_SFR'], 'o',
             color=colors[i], markersize=4, alpha=0.3);
    hs.append(h)
    hls.append(r'${} < \lg{{}}\,M_{{\rm star}} < {}$'.format(
               lgMstar_min, lgMstar_max))

    plt.plot(df.loc[sel, 'SFR'],
                  10**res.predict(df.loc[sel, ['lg_SFR', 'lg_Mstar10']]),
                  '-', color=colors[i])

    if i == 0:
        h, = plt.plot([-10, -9], [0, 1], '-', color=[0.5, 0.5, 0.5])
        hs.append(h)
        s = r'$y = {:.3f} x_1 {:+.3f} x_2 {:+.3f} x_1 x_2 {:+.3f}$'.format(
            res.params['lg_SFR'], res.params['lg_Mstar10'],
            res.params['lg_SFR:lg_Mstar10'], res.params['Intercept'])
        hls.append(s)
        s = (r'$y = \lg\,\Delta{}{\rm SFR}/M_\odot{}{\rm yr}^{-1}, '
             r'x_1 = \lg\,{\rm SFR}/M_\odot{}{\rm yr}^{-1}, '
             r'x_2 = \lg{}\,M_{\rm star}/10^{10}M_\odot$')
        plt.text(0.98, 0.02, s, fontsize=9.2, transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom')
        h, = plt.plot([-10, -9], [0, 1], lw=0, visible=False, marker=None)
        hs.append(h)
        hls.append('')


plt.xlabel(r'SFR [ $M_\odot$ yr$^{-1}$ ]', fontsize=fontsize)
plt.ylabel(r'$\Delta{}\,$SFR [ $M_\odot$ yr$^{-1}$ ]', fontsize=fontsize)
plt.xscale('log')
plt.yscale('log')

plt.legend(hs, hls, loc='lower right', frameon=False,
           markerfirst=False, fontsize=10,
           title='                                                                    xGASS')

plt.minorticks_on()
plt.gca().tick_params(axis='both', which='major', labelsize=fontsize-2)
plt.gca().tick_params(axis='both', which='minor', labelsize=fontsize-6)

plt.xlim([3e-3, 100])
plt.ylim([1e-2, 1])

if savefig:
    plt.savefig('xGASS_SF_uncertainty.pdf', bbox_inches='tight')
