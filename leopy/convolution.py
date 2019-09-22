"""Continuous random variable marginalized over a joint probability.

This module is part of LEO-Py --
Likelihood Estimation of Observational data with Python

Copyright 2019 University of Zurich, Robert Feldmann
"""
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
import scipy.integrate
import scipy.stats

import warnings

import leopy.integrate
import leopy.stats


class Convolution(scipy.stats.rv_continuous):
    r"""Continuous random variable marginalized over a joint probability.

    Subclass of `scipy.stats.rv_continuous`.
    It computes the probability density of random variable (RV) X from the
    probability density of a RV Y and the conditional probability density of
    X given Y, i.e.

    .. math::
        p_X(x) = \int_{-\infty}^{\infty} dy p_{X|Y}(x,y) p_Y(y).
    """

    def __init__(self, p_XY, p_Y, atol=1e-5, rtol=1e-5, eps=1e-5,
                 n_iter=8, maxiter=5, cdf_maxiter=100,
                 log_mode=True, verbosity=0, **kwargs):
        r"""Initialize self.

        Parameters
        ----------
        p_XY : object
            A continuous random variable describing p_X|Y(Y). Instance of
            class `scipy.stats.rv_continuous`.
        p_Y : object
            A continuous random variable describing p_Y(Y). Instance of
            class `scipy.stats.rv_continuous`.
        atol : float
            Absolute tolerance parameter of the quadrature routine involved
            in the numerical calculation of the convolution. Choose wisely
            as smaller values can significantly increase computation time.
            The parameter must be a positive number (default 1e-5).
        rtol : float
            Relative tolerance parameter of the quadrature routine involved
            in the numerical calculation of the convolution. Choose wisely
            as smaller values can significantly increase computation time.
            The parameter must be a positive number (default 1e-5).
        eps : float
            To speed up the convolution, restrict the integral to a range
            [Y1, Y2], with the property that p_Y.cdf(x1) >= eps and
            p_Y.cdf(x2) <= eps. Choose wisely as smaller values can
            increase computation time.
            The parameter must be a positive number (default is 1e-5).
        n_iter : int
            The initial number of points used in the integration step is
            2**n_iter. It is increased dynamically to satisfy the `atol` and
            `rtol` constraints. Consider choosing a larger `n_iter` if p_Y
            has strongly localized peaks on a scale much smaller than scale_Y.
            (the default is 8, i.e., 256 integration points).
        maxiter : int
            To archieve the desired `atol and `rtol`, `n_iter` may be increased
            (each time by 1). `maxiter` is the maximum number of times
            n_iter can be increased this way. If maxiter is reached and the
            `atol` and `rtol` limits are not satisfied, an AccuracyWarning is
            raised (maxiter is >= 0, default is 5).
        cdf_maxiter : int
            Maximum number of iterations to find the points where the CDF is
            below `eps` and above 1-`eps`. Needs to be larger than 10
            (default is 100).
        log_mode : bool
            If True, integration is done in log-space provided p_Y.a >= 0.
            This may provide a speed-up for certain distributions
            (default True).
        verbosity : int
            Level of verbosity (default is 0, i.e., no additional output)
        \*\*kwargs : Additional keyword arguments
            `\*\*kwargs` is passed to `scipy.stats.rv_continuous`.

        Returns
        -------
        Convolution
            Instance of class `Convolution`.

        Notes
        -----
        Convolutions are computed analytically if both p_Y and p_XY are
        normal distributions, speeding up the computation. In this case,
        the parameters `atol`, `rtol`, `eps, `n_iter`, `maxiter`, `log_mode`
        are not used. Setting self.name to anything other than 'norm' will
        force a numerical calculation. This can be used to compare the
        numerical and analytical result in the case of the convolution of
        normal distributions.

        Examples
        --------
        >>> import scipy.stats
        >>> a = Convolution(scipy.stats.norm, scipy.stats.lognorm)
        >>> a.pdf([-1, 0, 10.], 2., 1., 0., 2.)
        array([0.06118279, 0.15266689, 0.01466218])
        >>> a.cdf([-1, 0, 10.], 2., 1., 0., 2.)
        array([0.03449056, 0.14173643, 0.7884726 ])
        >>> round(a.median(2., 1., 0., 2.), 5)
        2.2706
        >>> b = Convolution(scipy.stats.norm, scipy.stats.norm)
        >>> b.median(2., 1., 1.5)
        1.0
        >>> c = Convolution(scipy.stats.norm, scipy.stats.norm)
        >>> c.name = 'composite'  # enforce numerical computation
        >>> round(c.median(2., 1., 1.5), 5)
        1.0
        """
        self.p_XY = p_XY        # probability distribution of X given Y
        self.p_Y = p_Y          # probability distribution of Y

        self.atol = atol
        self.rtol = rtol
        self.eps = eps
        self.n_iter = n_iter
        self.maxiter = maxiter
        self.cdf_maxiter = cdf_maxiter

        assert atol > 0 and rtol > 0 and eps > 0 and n_iter > 0
        assert maxiter >= 0
        assert cdf_maxiter > 10

        self.log_mode = log_mode
        self.verbosity = verbosity

        if self.p_XY.name == 'norm' and self.p_Y.name == 'norm':
            name = 'norm'
            # name = 'composite'
            shapes_str = 'scale_xy, loc_y, scale_y'
        else:
            name = 'composite'
            shapes_str = ''
            if self.p_XY.shapes:
                for s in self.p_XY.shapes.split(','):
                    shapes_str += '{}_xy, '.format(s)
            if self.p_Y.shapes:
                for s in self.p_Y.shapes.split(','):
                    shapes_str += '{}_y, '.format(s)
            shapes_str += 'scale_xy, loc_y, scale_y'

        self.shapes_arg = (self.p_XY.numargs, self.p_Y.numargs)
        self.shapes_numargs = self.p_XY.numargs + self.p_Y.numargs

        kwargs['shapes'] = shapes_str
        kwargs['name'] = name
        super(Convolution, self).__init__(**kwargs)

    def _updated_ctor_param(self):
        """Register parameters for proper pickling during multiprocessing."""
        dct = self._ctor_param.copy()
        dct['p_XY'] = self.p_XY
        dct['p_Y'] = self.p_Y
        dct['atol'] = self.atol
        dct['rtol'] = self.rtol
        dct['eps'] = self.eps
        dct['n_iter'] = self.n_iter
        dct['maxiter'] = self.maxiter
        dct['cdf_maxiter'] = self.cdf_maxiter
        dct['log_mode'] = self.log_mode
        dct['verbosity'] = self.verbosity
        return dct

    def _argcheck(self, *args):
        """Check for correct values on args and keywords."""
        shape_params_XY = args[:self.shapes_arg[0]]
        shape_params_Y = args[self.shapes_arg[0]:self.shapes_numargs]
        return np.logical_and(np.logical_and(
            self.p_Y._argcheck(*shape_params_Y),        # p_Y shape params
            self.p_XY._argcheck(*shape_params_XY)),     # p_XY shape params
            args[self.shapes_numargs] > 0)              # p_XY scale param

    def _broadcast(self, x, scale_XY, loc_Y, scale_Y, *shape_args):
        """Auxiliary function to broadcast & vectorize params."""
        (x, scale_XY, loc_Y, scale_Y, *shape_args) = np.broadcast_arrays(
            x, scale_XY, loc_Y, scale_Y, *shape_args)

        broadcast_shape = x.shape
        broadcast_ndim = x.ndim
        broadcast_num = int(np.prod(x.shape))

        if broadcast_ndim > 1:
            x = x.flatten()
            scale_XY = scale_XY.flatten()
            loc_Y = loc_Y.flatten()
            scale_Y = scale_Y.flatten()
            if self.shapes_numargs > 0:
                for i in range(self.shapes_numargs):
                    shape_args[i] = shape_args[i].flatten()

        elif broadcast_ndim < 1:
            x = np.atleast_1d(x)
            scale_XY = np.atleast_1d(scale_XY)
            loc_Y = np.atleast_1d(loc_Y)
            scale_Y = np.atleast_1d(scale_Y)
            if self.shapes_numargs > 0:
                for i in range(self.shapes_numargs):
                    shape_args[i] = np.atleast_1d(shape_args[i])

        return (broadcast_shape, broadcast_num, x, scale_XY, loc_Y, scale_Y,
                *shape_args)

    def _quad_conv(self, is_pdf, X, scale_XY, loc_Y, scale_Y, *shape_args):
        """PDF of RV X computed by convolution - do not call directly."""
        (broadcast_shape, broadcast_num, X, scale_XY, loc_Y, scale_Y,
            *shape_args) = self._broadcast(
                X, scale_XY, loc_Y, scale_Y, *shape_args)

        shape_params_XY = shape_args[:self.shapes_arg[0]]
        shape_params_Y = shape_args[
            self.shapes_arg[0]:self.shapes_numargs]

        result = np.zeros(broadcast_shape)

        log_mode = False
        if self.log_mode and self.p_Y.a >= 0:
            log_mode = True

        def f(yhat, sel=None):

            if sel is not None:
                sh_y = [_[sel, None] for _ in shape_params_Y]
                sh_xy = [_[sel, None] for _ in shape_params_XY]
                sxy = scale_XY[sel, None]
                sy = scale_Y[sel, None]
                loc_y = loc_Y[sel, None]
                x = X[sel, None]
            else:
                sh_y = [_[:, None] for _ in shape_params_Y]
                sh_xy = [_[:, None] for _ in shape_params_XY]
                sxy = scale_XY[:, None]
                sy = scale_Y[:, None]
                loc_y = loc_Y[:, None]
                x = X[:, None]

            if log_mode:
                yhat = np.exp(yhat)

            y = yhat * sy + loc_y
            xhat = (x - y)/sxy

            p_Y = self.p_Y._pdf

            if is_pdf:
                p_XY = self.p_XY._pdf
            else:
                p_XY = self.p_XY._cdf

            if log_mode:
                return (p_XY(xhat, *sh_xy) * p_Y(yhat, *sh_y) * yhat)
            else:
                return (p_XY(xhat, *sh_xy) * p_Y(yhat, *sh_y))

        # limit of yhat = (y - loc_Y)/scale_Y
        # set limits based on cdf of self.p_Y, i.e., find range
        # where self.p_Y.pdf is important
        lY, uY = leopy.stats.find_cdf_limits(
            self.eps, self.p_Y._cdf, self.p_Y.a, self.p_Y.b,
            args=shape_params_Y, maxiter=self.cdf_maxiter)
        lY = np.atleast_1d(lY)
        uY = np.atleast_1d(uY)

        # limit of xhat = (x-y)/sxy
        # convert into limit on yhat
        lXY_raw, uXY_raw = leopy.stats.find_cdf_limits(
            self.eps, self.p_XY._cdf, self.p_XY.a, self.p_XY.b,
            args=shape_params_XY, maxiter=self.cdf_maxiter)
        uXY = (X - lXY_raw * scale_XY - loc_Y)/scale_Y
        lXY = (X - uXY_raw * scale_XY - loc_Y)/scale_Y
        # ensure uXY and lXY are within domain of p_Y
        uXY = np.maximum(self.p_Y.a, np.minimum(self.p_Y.b, uXY))
        lXY = np.maximum(self.p_Y.a, np.minimum(self.p_Y.b, lXY))

        lower_limit = np.maximum(lY, lXY)
        upper_limit = np.minimum(uY, uXY)

        add = 0.
        if not is_pdf:
            # need to re-add part of integral where CDF_XY is ~ 1
            add = (self.p_XY.cdf(uXY_raw, *shape_params_XY) * (
                self.p_Y.cdf(lower_limit, *shape_params_Y)
                - self.p_Y.cdf(lY, *shape_params_Y)))

        if log_mode:
            lower_limit = np.log(lower_limit+1e-100)
            upper_limit = np.log(upper_limit+1e-100)

        result = leopy.integrate.quadrature(
            f, lower_limit, upper_limit,
            tol=self.atol, rtol=self.rtol,
            n_iter=self.n_iter, maxiter=self.maxiter)[0] + add

        if is_pdf:
            result = np.maximum(0., result)
            return result.reshape(broadcast_shape) / scale_XY
        else:
            result = np.minimum(1., np.maximum(0., result))
            return result.reshape(broadcast_shape)

    def _pdf(self, x, *args):
        """PDF of random variable X."""
        scale_XY, loc_Y, scale_Y = args[
            self.shapes_numargs:self.shapes_numargs+3]

        if self.name == 'norm':
            scale = np.sqrt(scale_Y**2 + scale_XY**2)
            return scipy.stats.norm._pdf((x - loc_Y)/scale) / scale

        return self._quad_conv(True, x, scale_XY, loc_Y, scale_Y,
                               *args[:self.shapes_numargs])

    def _cdf(self, x, *args):
        """CDF of random variable X."""
        scale_XY, loc_Y, scale_Y = args[
            self.shapes_numargs:self.shapes_numargs+3]

        if self.name == 'norm':
            return scipy.stats.norm.cdf(
                x, loc=loc_Y, scale=np.sqrt(scale_Y**2 + scale_XY**2))

        return self._quad_conv(False, x, scale_XY, loc_Y, scale_Y,
                               *args[:self.shapes_numargs])

    def _ppf_conv(self, q, scale_XY, loc_Y, scale_Y, *shape_args):
        """PPF of RV X computed by convolution - do not call directly."""
        (broadcast_shape, broadcast_num, q, scale_XY, loc_Y, scale_Y,
            *shape_args) = self._broadcast(
                q, scale_XY, loc_Y, scale_Y, *shape_args)

        def f(x, q, scale_XY, loc_Y, scale_Y, *shape_args):
            return self._quad_conv(
                False, x, scale_XY, loc_Y, scale_Y, *shape_args) - q

        def fprime(x, q, scale_XY, loc_Y, scale_Y, *shape_args):
            return self._quad_conv(
                True, x, scale_XY, loc_Y, scale_Y, *shape_args)

        out = np.zeros(broadcast_num)
        for i, (lq, lscale_XY, lloc_Y, lscale_Y, *lshape) in enumerate(
                zip(q, scale_XY, loc_Y, scale_Y, *shape_args)):
            out[i] = scipy.optimize.newton(
                f, x0=lloc_Y, fprime=fprime,
                args=(lq, lscale_XY, lloc_Y, lscale_Y, *lshape))

        return out.reshape(broadcast_shape)

    def _ppf(self, q, *args):
        """PPF (inverse of CDF) of RV X."""
        scale_XY, loc_Y, scale_Y = args[
            self.shapes_numargs:self.shapes_numargs+3]

        if self.name == 'norm':
            return scipy.stats.norm.ppf(
                q, loc=loc_Y, scale=np.sqrt(scale_Y**2 + scale_XY**2))

        return self._ppf_conv(q, scale_XY, loc_Y, scale_Y,
                              *args[:self.shapes_numargs])


if __name__ == '__main__':
    import doctest
    doctest.testmod()
