"""Additional statistics functionality.

This module is part of LEO-Py --
Likelihood Estimation of Observational data with Python

Copyright 2019 University of Zurich, Robert Feldmann

This module makes use of the source code of the open-source scipy.stats module.
The class doc-strings in this module include some text from the documentation
of scipy.stats functions. Please see
Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools
for Python, 2001-, http://www.scipy.org
"""
# -- Scipy Copyright notice --
#
# Copyright © 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright © 2003-2013 SciPy Developers.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#     Neither the name of Enthought nor the names of the SciPy Developers may be
#     used to endorse or promote products derived from this software without
#     specific prior written permission.
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#     TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#     PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE
#     LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#     CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#     SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#     INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#     CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#     ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.
#
# --- End of Scipy Copyright notice
#
# LEO-Py is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LEO-Py is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LEO-Py. If not, see <https://www.gnu.org/licenses/>.
import numpy as np
import scipy.stats
import warnings

from leopy.misc import AccuracyWarning


class zi_gamma_gen(scipy.stats.rv_continuous):
    r"""A (pseudo) zero-inflated gamma continous random variable.

    Subclass of `scipy.stats.rv_continuous`.

    This class implements a mixture model of a gamma and a half-normal
    distribution. The latter can be interpreted as a zero-inflated component
    provided :math:`s \ll 1`.

    Notes
    -----
    The probability density function for `zi_gamma` is:

    .. math::

        f(x, a, s, z) = (1-z)\frac{x^{a-1} \exp(-x)}{\Gamma(a)}
                        + z \frac{\exp(-x^2/(2s^2))}{s\sqrt{2\pi}}

    for :math:`x \ge 0`, :math:`a > 0`, :math:`s > 0`, :math:`0 \le z \le 1`.
    Here :math:`\Gamma(a)` refers to the gamma function.

    `zi_gamma` takes ``a``, ``s``, and ``z`` as shape parameters for :math:`a`,
    :math:`s`, and :math:`z`.

    The probability density above is defined in the "standardized" form. To
    shift and/or scale the distribution use the ``loc`` and ``scale``
    parameters. Specifically, ``zi_gamma.pdf(x, a, s, z, loc, scale)`` is
    equivalent to ``zi_gamma.pdf(y, a, s, z) / scale`` with
    ``y = (x - loc) / scale``.

    """
    def __init__(self, *args, **kwargs):
        self._return_index = kwargs.pop('index', False)
        super().__init__(*args, **kwargs)
        self.a = 0
        self.b = np.inf

    def set_return_index(self, flag):
        """If set to True, rvs() will produce additional output"""
        self._return_index = flag

    def _pdf(self, x, a, s, z):
        return ((1-z)*scipy.stats.gamma._pdf(x, a)
                + z*scipy.stats.halfnorm._pdf(x/s)/s)

    def _cdf(self, x, a, s, z):
        return ((1-z)*scipy.stats.gamma._cdf(x, a)
                + z*scipy.stats.halfnorm._cdf(x/s))

    def _rvs(self, a, s, z):
        sz = self._size
        ind = self._return_index
        be = scipy.stats.bernoulli.rvs(z, size=sz)
        hn = scipy.stats.halfnorm.rvs(size=sz, scale=s)
        ga = scipy.stats.gamma.rvs(a, size=sz)
        if ind:
            return np.where(be, hn, ga), be
        else:
            return np.where(be, hn, ga)

    def _prob_component(self, x, a, s, z):
        p = np.zeros((2, len(x)))
        p[0, :] = (1-z)*scipy.stats.gamma._pdf(x, a)
        p[1, :] = z*scipy.stats.halfnorm._pdf(x/s)/s
        return p

    def _argcheck(self, *args):
        """Check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.

        """
        cond = 1
        cond = np.logical_and(cond, (np.asarray(args[0]) > 0))
        cond = np.logical_and(cond, (np.asarray(args[1]) > 0))
        cond = np.logical_and(
            np.logical_and(cond, (np.asarray(args[2]) >= 0)),
            (np.asarray(args[2]) <= 1))
        return cond

    def prob_component(self, x, *args, **kwds):
        """Calculate probability that x belongs to each component."""
        args, loc, scale = self._parse_args(*args, **kwds)
        p = self._prob_component((x-loc)/scale, *args)
        return p / np.sum(p, axis=0)

    def rvs_component(self, x, *args, **kwds):
        """Assign x to one of the components in a probabilistic fashion."""
        p = self.prob_component(x, *args, **kwds)
        m = np.zeros(len(x), dtype=int)
        for i in range(len(x)):
            try:
                m[i] = np.round(np.nonzero(
                    scipy.stats.multinomial.rvs(n=1, p=p[:, i], size=1))[1][0])
            except ValueError:
                m[i] = -1

        return m

zi_gamma = zi_gamma_gen(name='zi_gamma')

class zi_gamma_lognorm_gen(zi_gamma_gen):
    r"""A (pseudo) zero-inflated gamma + lognormal continuous random variable.

    Subclass of `scipy.stats.rv_continuous`.

    This class implements a mixture model of a gamma, a half-normal
    distribution, and a lognormal distribution. The half-normal distribution
    can be interpreted as a zero-inflated component provided :math:`s \ll 1`.
    A possible use of the lognormal distribution is to model outliers from the
    primary (zero-inflated gamma) distribution.

    Notes
    -----
    The probability density function for `zi_gamma_lognorm` is:

    .. math::
        f(x, a, s, z, A, S, Z) =
            (1-Z)\left[(1-z)\frac{x^{a-1} \exp(-x)}{\Gamma(a)}
                       + z \frac{\exp(-x^2/(2s^2))}{s\sqrt{2\pi}}\right] \\
            + Z \left[\frac{1}{S A x \sqrt{2\pi}}
                      \exp\left(-\frac{\log^2(x/S)}{2A^2}\right)\right]

    for :math:`x \ge 0`, :math:`a > 0`, :math:`s > 0`, :math:`0 \le z \le 1`,
    :math:`A > 0`, :math:`S > 0`, and :math:`0 \le Z \le 1`.
    Here :math:`\Gamma(a)` refers to the gamma function.

    `zi_gamma_lognorm` takes ``a``, ``s``, ``z``, ``A``, ``S``, and ``Z`` as
    shape parameters for :math:`a`, :math:`s`, :math:`z`, :math:`A`, :math:`S`,
    and :math:`Z`.

    The probability density above is defined in the "standardized" form. To
    shift and/or scale the distribution use the ``loc`` and ``scale``
    parameters.
    Specifically, ``zi_gamma_lognorm.pdf(x, a, s, z, A, S, Z, loc, scale)``
    is equivalent to ``zi_gamma_lognorm.pdf(y, a, s, z, A, S, Z) / scale`` with
    ``y = (x - loc) / scale``.

    A common parametrization for a lognormal random variable ``Y`` is in
    terms of the mean, ``mu``, and standard deviation, ``sigma``, of the
    unique normally distributed random variable ``X`` such that exp(X) = Y.
    This parametrization corresponds to ``A = sigma`` and ``S = exp(mu)``.

    """
    def __init__(self, *args, **kwargs):
        self._return_index = kwargs.pop('index', False)
        super().__init__(*args, **kwargs)

    def _pdf(self, x, a, s, z, A, S, Z):
        return ((1-Z)*super()._pdf(x, a, s, z)
                 + Z*scipy.stats.lognorm._pdf(x/S, A)/S)

    def _cdf(self, x, a, s, z, A, S, Z):
        return ((1-Z)*super()._cdf(x, a, s, z)
                + Z*scipy.stats.lognorm._cdf(x/S, A))

    def _rvs(self, a, s, z, A, S, Z):
        sz = self._size
        ind = self._return_index
        be = scipy.stats.bernoulli.rvs(Z, size=sz)
        ln = scipy.stats.lognorm.rvs(A, size=sz, scale=S)
        if ind:
            zi_ga, zi_ga_ind = super()._rvs(a, s, z)
            return np.where(be, ln, zi_ga), zi_ga_ind, be
        else:
            zi_ga = super()._rvs(a, s, z)
            return np.where(be, ln, zi_ga)

    def _prob_component(self, x, a, s, z, A, S, Z):
        p = np.zeros((3, len(x)))
        p[:2, :] = (1-Z)*super()._prob_component(x, a, s, z)
        p[2, :] = Z*scipy.stats.lognorm._pdf(x/S, A)/S
        return p

    def _argcheck(self, *args):
        """Check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.

        """
        cond = 1
        cond = np.logical_and(cond, (np.asarray(args[0]) > 0))
        cond = np.logical_and(cond, (np.asarray(args[1]) > 0))
        cond = np.logical_and(
            np.logical_and(cond, (np.asarray(args[2]) >= 0)),
            (np.asarray(args[2]) <= 1))
        cond = np.logical_and(cond, (np.asarray(args[3]) > 0))
        cond = np.logical_and(cond, (np.asarray(args[4]) > 0))
        cond = np.logical_and(
            np.logical_and(cond, (np.asarray(args[5]) >= 0)),
            (np.asarray(args[5]) <= 1))
        return cond

zi_gamma_lognorm = zi_gamma_lognorm_gen(name='zi_gamma_lognorm')


class gamma_lognorm_gen(scipy.stats.rv_continuous):
    r"""A gamma + lognormal continuous random variable.

    Subclass of `scipy.stats.rv_continuous`.

    This class implements a mixture model of a gamma and a lognormal
    distribution. A possible use of the lognormal distribution is to model
    outliers from the primary (gamma) distribution.

    Notes
    -----
    The probability density function for `gamma_lognorm` is:

    .. math::

        f(x, a, A, S, Z) =
            (1-Z)\left[\frac{x^{a-1} \exp(-x)}{\Gamma(a)}\right]
            + Z \left[\frac{1}{S A x \sqrt{2\pi}}
                      \exp\left(-\frac{\log^2(x/S)}{2A^2}\right)\right]

    for :math:`x \ge 0`, :math:`a > 0`, :math:`A > 0`, :math:`S > 0`, and
    :math:`0 \le Z \le 1`. Here :math:`\Gamma(a)` refers to the gamma function.

    `gamma_lognorm` takes ``a``, ``A``, ``S``, and ``Z`` as shape parameters
    for :math:`a`, :math:`A`, :math:`S`, and :math:`Z`.

    The probability density above is defined in the "standardized" form. To
    shift and/or scale the distribution use the ``loc`` and ``scale``
    parameters.
    Specifically, ``gamma_lognorm.pdf(x, a, A, S, Z, loc, scale)``
    is equivalent to ``gamma_lognorm.pdf(y, a, A, S, Z) / scale`` with
    ``y = (x - loc) / scale``.

    A common parametrization for a lognormal random variable ``Y`` is in
    terms of the mean, ``mu``, and standard deviation, ``sigma``, of the
    unique normally distributed random variable ``X`` such that exp(X) = Y.
    This parametrization corresponds to ``A = sigma`` and ``S = exp(mu)``.

    """
    def __init__(self, *args, **kwargs):
        self._return_index = kwargs.pop('index', False)
        super().__init__(*args, **kwargs)
        self.a = 0
        self.b = np.inf

    def set_return_index(self, flag):
        """If set to True, rvs() will produce additional output"""
        self._return_index = flag

    def _pdf(self, x, a, A, S, Z):
        return ((1-Z)*scipy.stats.gamma._pdf(x, a)
                 + Z*scipy.stats.lognorm._pdf(x/S, A)/S)

    def _cdf(self, x, a, A, S, Z):
        return ((1-Z)*scipy.stats.gamma._cdf(x, a)
                + Z*scipy.stats.lognorm._cdf(x/S, A))

    def _rvs(self, a, A, S, Z):
        sz = self._size
        ind = self._return_index
        be = scipy.stats.bernoulli.rvs(Z, size=sz)
        ln = scipy.stats.lognorm.rvs(A, size=sz, scale=S)
        ga = scipy.stats.gamma.rvs(a, size=sz)
        if ind:
            return np.where(be, ln, ga), be
        else:
            return np.where(be, ln, ga)

    def _prob_component(self, x, a, A, S, Z):
        p = np.zeros((2, len(x)))
        p[0, :] = (1-Z)*scipy.stats.gamma._pdf(x, a)
        p[1, :] = Z*scipy.stats.lognorm._pdf(x/S, A)/S
        return p

    def _argcheck(self, *args):
        """Check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.

        """
        cond = 1
        cond = np.logical_and(cond, (np.asarray(args[0]) > 0))
        cond = np.logical_and(cond, (np.asarray(args[1]) > 0))
        cond = np.logical_and(cond, (np.asarray(args[2]) > 0))
        cond = np.logical_and(
            np.logical_and(cond, (np.asarray(args[3]) >= 0)),
            (np.asarray(args[3]) <= 1))
        return cond

gamma_lognorm = gamma_lognorm_gen(name='gamma_lognorm')


def nearest_psd(A):
    """Compute symmetric positive-semidefinite matrix 'closest' to A.

    Parameters
    ----------
    A : array
        A real square matrix

    Returns
    -------
    array
        A symmetric positive approximant of A, i.e., a symmetric matrix that is
        PSD and closest in the Frobenius norm.

    Notes
    -----
    'Nearest' refers to the Frobenius norm.

    The implementation is based on the algorithm by Higham 1988, see
    `DOI:10.1016/0024-3795(88)90223-6
    <http://dx.doi.org/10.1016/0024-3795(88)90223-6>`_.

    Examples
    --------
    >>> A = np.array([[0, 0, 0], [1, 0, 0], [0, 1., 0]])
    >>> nearest_psd(A)
    array([[0.1767767 , 0.25      , 0.1767767 ],
           [0.25      , 0.35355339, 0.25      ],
           [0.1767767 , 0.25      , 0.1767767 ]])
    """
    B = (A + A.T)/2

    lam, Z = np.linalg.eigh(B)
    d = [max(_, 0) for _ in lam]

    XF = Z.dot(np.diag(d).dot(Z.T))

    return XF


def find_cdf_limits(q, f, a, b, args=(), exponent=1.0, maxiter=100,
                    return_iterations=False):
    """find arguments xl, xu of cdf f such that f(xl)<=q & f(xu)>=1-q"""
    # f is assumed to be a monoton. incr. function from (a,b) to [0, 1]

    # x has various ranges
    # y has range [0, 1]
    # g maps from y to x
    # g_inv maps from x to y

    # map from [0, 1] to the actual domain of f
    # two functions as loss of accuracy results in y != 1-(1-y)
    if np.isneginf(a) and np.isposinf(b):
        gs = [lambda y: np.log(y/(1.-y)),
              lambda y: np.log((1.-y)/y)]
    elif np.isneginf(a):
        gs = [lambda y: -(1.-y)/y + b,
              lambda y: -y/(1.-y) + b]
    elif np.isposinf(b):
        gs = [lambda y: y/(1.-y) + a,
              lambda y: (1.-y)/y + a]
    else:
        gs = [lambda y: y*(b-a) + a,
              lambda y: (1.-y)*(b-a) + a]

    def calc_bad(y, shape_params, limit_type, sel=np.array(False)):

        g = np.zeros_like(y)
        g[~sel] = gs[0](y[~sel])
        g[sel] = gs[1](y[sel])

        fval = np.array(f(g, *shape_params))
        limit = g * np.ones_like(fval)
        if limit_type == 0:
            bad = np.array((fval > q))
        else:
            bad = np.array((fval < 1-q))
        return limit, bad, fval

    # limit_type 0/1 is lower/upper limit
    for limit_type in range(2):

        limit, bad, fval = calc_bad(np.array(0.5), args, limit_type)

        limit = np.atleast_1d(limit)
        bad = np.atleast_1d(bad)

        bad_initial = bad
        not_flipped = np.ones_like(bad)

        for i_n, n in enumerate(range(2, maxiter)):
            y0 = 2**(-n**exponent)
            if y0 == 0:
                break

            y = y0 * np.ones(np.sum(not_flipped))
            if limit_type == 0:
                sel = ~bad_initial[not_flipped]
            else:
                sel = bad_initial[not_flipped]

            sh = [np.atleast_1d(_)[not_flipped] for _ in args]
            limit_new, bad_new, fval = calc_bad(
                y, sh, limit_type, sel)

            sel = ~bad_new

            ind_notflipped = np.where(not_flipped)[0]

            limit_sel = ind_notflipped[sel]
            limit_notsel = ind_notflipped[~sel]
            limit[limit_sel] = limit_new[sel]

            sel1 = ~bad_new & bad_initial[not_flipped]
            sel2 = bad_new & ~bad_initial[not_flipped]
            not_flipped[ind_notflipped[sel1 | sel2]] = False

            nn = np.sum(not_flipped)

            if nn == 0:
                break

            limit[limit_notsel] = limit_new[~sel]

        if limit_type == 0:
            lower_limit = limit
            n_lower_limit = n
        else:
            upper_limit = limit
            n_upper_limit = n

        if nn > 0:
            if limit_type == 0:
                limit_type_string = 'lower'
            else:
                limit_type_string = 'upper'
            warnings.warn('Maximum number of iterations ({}) exceeded '
                          'while determining {} limit (n={})'.format(
                          maxiter, limit_type_string, nn),
                          AccuracyWarning)

    if return_iterations:
        return lower_limit, upper_limit, n_lower_limit, n_upper_limit
    else:
        return lower_limit, upper_limit


def logit(x):
    """Logit function."""
    return np.log(x / (1.-x))


def inv_logit(x):
    """logistic function (inverse of logit)."""
    #the below equals 1./(1.+np.exp(-x)) but is slightly faster
    return 0.5*(1. + np.tanh(0.5*x))


if __name__ == '__main__':

    import doctest
    doctest.testmod()
