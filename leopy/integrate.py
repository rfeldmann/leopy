"""Compute a definite integral using Gaussian quadrature.

Copyright 2019 University of Zurich, Robert Feldmann

The content of this file modifies scipy/integrate/quadrature.py that is part of
the scipy.integrate package.

Please see
Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools
for Python, 2001-, http://www.scipy.org

The modifications enable fixed_quad and quadrature to operate on vectorized
limits. It also includes modifications to make it compatible with python3.

This file may be replaced with direct calls to the scipy.integrate functions
in future versions.
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
import scipy.special
import warnings

from leopy.misc import AccuracyWarning


def _cached_roots_legendre(n):
    """
    Cache roots_legendre results to speed up calls of the fixed_quad
    function.
    """
    if n in _cached_roots_legendre.cache:
        return _cached_roots_legendre.cache[n]

    _cached_roots_legendre.cache[n] = scipy.special.roots_legendre(n)
    return _cached_roots_legendre.cache[n]

_cached_roots_legendre.cache = dict()

def fixed_quad(func, a, b, args=(), n=5):
    """
    Compute a definite integral using fixed-order Gaussian quadrature.

    Integrate `func` from `a` to `b` using Gaussian quadrature of
    order `n`.

    Parameters
    ----------
    func : callable
        A Python function or method to integrate (must accept vector inputs).
        If integrating a vector-valued function, the returned array must have
        shape ``(..., len(x))``.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function, if any.
    n : int, optional
        Order of quadrature integration. Default is 5.

    Returns
    -------
    val : float
        Gaussian quadrature approximation to the integral
    none : None
        Statically returned value of None


    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    dblquad : double integrals
    tplquad : triple integrals
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    romb : integrators for sampled data
    simps : integrators for sampled data
    cumtrapz : cumulative integration for sampled data
    ode : ODE integrator
    odeint : ODE integrator

    Examples
    --------
    >>> from leopy import integrate
    >>> f = lambda x: x**8
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=4)
    (array([0.11108844]), None)
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=5)
    (array([0.11111111]), None)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=4)
    (array([0.99999998]), None)
    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=5)
    (array([1.]), None)
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0

    """
    x, w = _cached_roots_legendre(n)
    x = np.real(x)

    if np.any(np.logical_or(np.isinf(a), np.isinf(b))):
        raise ValueError("Gaussian quadrature is only available for "
                         "finite limits.")

    # y = (b-a)*(x+1)/2 + a
    y = np.outer(b-a, 0.5*(x+1)) + np.atleast_2d(a).T
    return (b-a)/2.0 * np.sum(w*func(y, *args), axis=-1), None

def quadrature(func, a, b, args=(), tol=1.49e-8, rtol=1.49e-8, maxiter=10,
               n_iter=4):
    """
    Compute a definite integral using fixed-tolerance Gaussian quadrature.

    Integrate `func` from `a` to `b` using Gaussian quadrature
    with absolute tolerance `tol`.

    Parameters
    ----------
    func : function
        A Python function or method to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function.
    tol, rtol : float, optional
        Iteration stops when error between last two iterates is less than
        `tol` OR the relative change is less than `rtol`.
    maxiter : int, optional
        2**(`n_iter` + `maxiter`) is maximum order of Gaussian quadrature.
    n_iter : int, optional
        2**`n_iter` is minimum order of Gaussian quadrature.

    Returns
    -------
    val : float
        Gaussian quadrature approximation (within tolerance) to integral.
    err : float
        Difference between last two estimates of the integral.

    See also
    --------
    romberg: adaptive Romberg quadrature
    fixed_quad: fixed-order Gaussian quadrature
    quad: adaptive quadrature using QUADPACK
    dblquad: double integrals
    tplquad: triple integrals
    romb: integrator for sampled data
    simps: integrator for sampled data
    cumtrapz: cumulative integration for sampled data
    ode: ODE integrator
    odeint: ODE integrator

    Examples
    --------
    >>> from leopy import integrate
    >>> f = lambda x: x**8
    >>> integrate.quadrature(f, 0.0, 1.0)
    (array([0.11111111]), array([9.85322934e-16]))
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> integrate.quadrature(np.cos, 0.0, np.pi/2)
    (array([1.]), array([7.77156117e-16]))
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0
    """
    if not isinstance(args, tuple):
        args = (args,)

    a = np.array(a)
    b = np.array(b)
    bad = np.atleast_1d(a < b)

    nbad = np.sum(bad)

    val = np.zeros_like(bad, dtype=np.float)
    err = np.zeros_like(val)
    newval = np.zeros_like(val)

    for i_n, n in enumerate(range(n_iter - 1, (n_iter + 1 + maxiter))):

        if nbad == 0:
            break

        ngood = bad.size - nbad
        if ngood == 0:
            newval = fixed_quad(func, a, b, (*args,), 2**n)[0]
        else:
            newval[bad] = fixed_quad(
                func, a[bad], b[bad], (bad, *args), 2**n)[0]

        if i_n == 0:
            err[bad] = np.inf
        else:
            err[bad] = abs(newval[bad]-val[bad])
        val[bad] = newval[bad]

        bad[bad] = np.logical_not(np.logical_or(
            err[bad] < tol, err[bad] < rtol*abs(val[bad]))).flatten()

        nbad = np.sum(bad)

    if nbad > 0:
        warnings.warn(
            '{} entries not converged. Largest error = {:e}. '
            'Consider increasing maxiter (currently set to {})'.format(
                nbad, np.max(err), maxiter),
            AccuracyWarning)

    return val, err
