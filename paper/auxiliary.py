"""Auxiliary functions for the likelihood maximization and plotting.

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
import time

class MyPrintFun(object):
    """Print info about basinhopping."""

    def __init__(self):
        self.time = time.time()
        self.iteration = 0
        self.minimum = np.inf
        self.x = None

    def __call__(self, x, f, accepted):
        """Return basinhopping info."""
        time_new = time.time()
        print('Iteration {} needed {:.3f} s'.format(
            self.iteration, time_new - self.time))
        self.iteration += 1
        self.time = time_new
        print('[', end='')
        print(', '.join(map(lambda _: '{:.3g}'.format(_), x)), end='')
        print('] ', end='')
        print('Minimum {:.5f} accepted: {}'.format(f, int(accepted)))

        if accepted and f < self.minimum:
            self.minimum = f
            self.x = x

        if self.x is not None:
            print('Accepted minimum so far: {:.5f} at '.format(self.minimum),
                  end='')
            print('[', end='')
            print(', '.join(map(lambda _: '{:.3g}'.format(_), self.x)),
                  end='')
            print('] ')


class MyTakeStep(object):
    """Set step size of basinhopping."""

    def __init__(self, stepsize=1.0):
        self.stepsize = stepsize

    def __call__(self, x):
        """Set new step size."""
        s = self.stepsize
        print('Current stepsize: {:.3g}'.format(s))
        x += np.random.uniform(-s, s, x.shape)
        return x

def compute_confidence_levels(x, y, levels=[0.6827, 0.9545, 0.9973], bins=20):
    """Compute confidence levels for a set of points"""

    # Make a 2d normed histogram
    min_points_per_bin = 10

    if np.isscalar(bins):
        lbins = min(bins, np.sqrt(len(x) / min_points_per_bin))
    else:
        lbins = bins
    H, xedges, yedges=np.histogram2d(x, y, bins=lbins, normed=True)

    # fraction of probability in regions above limit
    def frac_prob(limit):
        w = np.where(H>=limit)
        dx = xedges[w[0]+1] - xedges[w[0]]
        dy = yedges[w[1]+1] - yedges[w[1]]
        return np.sum(H[w]*dx*dy)

    def objective(limit, target):
        return frac_prob(limit) - target

    assert(np.isclose(frac_prob(0), 1.0))

    # Find levels by integrating histogram to objective
    limits = np.linspace(H.min(), H.max(), 1000)
    frac_probs = np.zeros_like(limits)
    for ilimit, limit in enumerate(limits):
        frac_probs[ilimit] = frac_prob(limit)
    ls = np.interp(levels, np.flipud(frac_probs), np.flipud(limits))

    xcenters = 0.5*(xedges[1:] + xedges[:-1])
    ycenters = 0.5*(yedges[1:] + yedges[:-1])

    return np.array(ls), xcenters, ycenters, H

def compute_confidence_levels_pdf(pdf, levels=[0.6827, 0.9545, 0.9973],
                                  bins=20, x_range=[0, 1], y_range=[0, 1]):
    """Compute confidence levels for a given prob. distribution function"""

    dx = (x_range[1] - x_range[0]) / bins
    dy = (y_range[1] - y_range[0]) / bins
    xe = np.linspace(x_range[0], x_range[1], bins)
    ye = np.linspace(y_range[0], y_range[1], bins)
    xc = 0.5*(xe[1:] + xe[:-1])
    yc = 0.5*(ye[1:] + ye[:-1])

    X, Y = np.meshgrid(xc, yc, indexing='ij')
    H = pdf(X, Y)

    # fraction of probability in regions above limit
    def frac_prob(limit):
        w = np.where(H>=limit)
        return np.sum(H[w])*dx*dy

    # make sure 2d-histogram contains most of the probability
    if frac_prob(0) < 0.95 or frac_prob(0) > 1.0+1e-10:
        print(frac_prob(0))
        raise RuntimeError('2d histogram does not integrate to ~1, please use '
                           'more bins and/or a larger range.')

    # renormalize:
    H /= frac_prob(0)

    def objective(limit, target):
        return frac_prob(limit) - target

    # Find levels by integrating histogram to objective
    if 0:
        ls = []
        for level in levels:
                ls.append(scipy.optimize.bisect(
                    objective, H.min(), H.max(), args=(level,)))
    else:
        limits = np.linspace(H.min(), H.max(), 1000)
        frac_probs = np.zeros_like(limits)
        for ilimit, limit in enumerate(limits):
            frac_probs[ilimit] = frac_prob(limit)
        ls = np.interp(levels, np.flipud(frac_probs), np.flipud(limits))

    return np.array(ls), xc, yc, H
