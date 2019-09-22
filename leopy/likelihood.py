"""Compute Likelihood of parameters for observational data set.

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
import scipy.stats

import warnings

import leopy

class Likelihood:
    """Compute likelihood, i.e., probability of data set given model params."""

    def __init__(self, obs, p_true='norm', p_cond=None, verbosity=0,
                 **kwargs):
        r"""Initialize self.

        Parameters
        ----------
        obs : object
            Instance of class `Observation` containing the observed data.
        p_true : object or str
            Instance of class `scipy.stats.rv_continuous` describing the
            probability distribution of the true values of a given observable.
            The member functions _pdf(), _cdf(), and _ppf() are used during the
            likelihood calculation.
            If `p_true` is a string, it assumes it is a function of the same
            name defined in in scipy.stats. (the default is 'norm').
        p_cond : object or str or None
            Same as `p_true` but describing the conditional probability
            distribution of the observed values given the true values of a
            given observable. If set to None, the conditional probability is
            assumed to be a delta function, i.e., p_obs = p_true
            (the default is None).
        verbosity : int
            Level of verbosity (default is 0, i.e., no additional output)
        **kwargs : type
            `**kwargs` is passed to an instance of `Convolution`.

        Returns
        -------
        Likelihood
            Instance of class `Likelihood`.

        Examples
        --------
        >>> import pandas as pd
        >>> from leopy import Observation, Likelihood
        >>> d = {'v0': [1, 2], 'e_v0': [0.1, 0.2],
        ...      'v1': [3, 4], 'e_v1': [0.1, 0.1]}
        >>> obs = Observation(pd.DataFrame(d), 'testdata')
        Reading dataset 'testdata' and extracting 2 variables (['v0', 'v1'])
        Errors of different observables are assumed to be uncorrelated
        >>> l = Likelihood(obs, p_true='lognorm', p_cond='norm')
        >>> l.p([0.5, 0.7], [1, 2], shape_true=[[1.4], [2.]])
        array([[0.0441545],
               [0.0108934]])
        """
        self.obs = obs
        self.verbosity = verbosity

        # set self.p_true & self.p_cond
        for ip, p in enumerate([p_true, p_cond]):
            if type(p) in [list, tuple]:
                assert len(p) == obs.num_var
                sp = []
                for _p in p:
                    if type(_p) == str:
                        sp.append(eval('scipy.stats.{}'.format(_p)))
                    else:
                        sp.append(_p)
            else:
                if type(p) == str:
                    sp = [eval('scipy.stats.{}'.format(p))] * obs.num_var
                else:
                    sp = [p] * obs.num_var
            if ip == 0:
                self.p_true = sp
            elif ip == 1:
                self.p_cond = sp

        # set self.p_obs
        self.p_obs = []
        for var in range(obs.num_var):
            if self.p_cond[var] is not None:
                self.p_obs.append(leopy.Convolution(
                    self.p_cond[var], self.p_true[var], verbosity=verbosity,
                    **kwargs))
            else:
                self.p_obs.append(self.p_true[var])

    def _normal_partial_integral(self, sel_nocen, sel_cen, cov,
                                 limit_type, return_log=False):
        """Compute prob. of partially integrated multivariate Gaussian."""
        if len(sel_nocen) > 0:
            S2 = cov[np.ix_(sel_nocen, sel_nocen)]

        if len(sel_cen) > 0:
            U2 = cov[np.ix_(sel_cen, sel_cen)]
            if np.sum(limit_type == -1) > 0:
                A = np.diag(limit_type)
                U2 = A.dot(U2.dot(A))
            else:
                A = np.array(1)
            if len(sel_nocen) > 0:
                T2 = cov[np.ix_(sel_nocen, sel_cen)].dot(A)
                T2T = np.transpose(T2)  # A.dot(cov[np.ix_(sel_cen,sel_nocen)])
                T2T_S2inv = T2T.dot(np.linalg.inv(S2))
                U2 -= T2T_S2inv.dot(T2)

        # limit_type states whether limit is lower (+1) or upper (-1).
        # it is assumed to be a list of length sum(sel_cen)
        limit_type = limit_type.reshape(limit_type.shape[0], 1)

        if len(sel_cen) > 0 and len(sel_nocen) > 0:
            if return_log:
                def f(z):
                    z1 = z[sel_nocen]
                    z2 = z[sel_cen] * limit_type
                    z2_hat = T2T_S2inv.dot(z1)
                    return (scipy.stats.multivariate_normal.logpdf(
                        z1.T, cov=S2, allow_singular=False)
                        + scipy.stats.multivariate_normal.logcdf(
                        (z2 - z2_hat).T, cov=U2, allow_singular=False))
            else:
                def f(z):
                    z1 = z[sel_nocen]
                    z2 = z[sel_cen] * limit_type
                    z2_hat = T2T_S2inv.dot(z1)

                    return (scipy.stats.multivariate_normal.pdf(
                        z1.T, cov=S2, allow_singular=False)
                        * scipy.stats.multivariate_normal.cdf(
                        (z2 - z2_hat).T, cov=U2, allow_singular=False))
        elif len(sel_cen) > 0:
            if return_log:
                def f(z):
                    z2 = z[sel_cen] * limit_type
                    return scipy.stats.multivariate_normal.logcdf(
                        z2.T, cov=U2, allow_singular=False)
            else:
                def f(z):
                    z2 = z[sel_cen] * limit_type
                    return scipy.stats.multivariate_normal.cdf(
                        z2.T, cov=U2, allow_singular=False)

        elif len(sel_nocen) > 0:
            if return_log:
                def f(z):
                    return scipy.stats.multivariate_normal.logpdf(
                        z[sel_nocen].T, cov=S2, allow_singular=False)
            else:
                def f(z):
                    return scipy.stats.multivariate_normal.pdf(
                        z[sel_nocen].T, cov=S2, allow_singular=False)
        else:
            if return_log:
                def f(z):
                    return -np.inf*np.ones(z.shape[1:]).T
            else:
                def f(z):
                    return np.zeros(z.shape[1:]).T
        return f

    def _normal_partial_integral_uncorr(self, sel_nocen, sel_cen, limit_type):
        """Compute prob. of part. integrated, uncorr. multivar. Gaussian."""
        limit_type = limit_type.reshape(limit_type.shape[0], 1)

        if len(sel_nocen) > 0:
            normal_pdf = scipy.stats.norm._pdf

        if len(sel_cen) > 0:
            normal_cdf = scipy.stats.norm._cdf

        if len(sel_cen) > 0 and len(sel_nocen) > 0:
            def f(z):
                z1 = z[sel_nocen]
                z2 = z[sel_cen] * limit_type
                return (np.prod(normal_pdf(z1.T), axis=-1)
                        * np.prod(normal_cdf(z2.T), axis=-1))
        elif len(sel_cen) > 0:
            def f(z):
                z2 = z[sel_cen] * limit_type
                return np.prod(normal_cdf(z2.T), axis=-1)
        elif len(sel_nocen) > 0:
            def f(z):
                return np.prod(normal_pdf(z[sel_nocen].T), axis=-1)
        else:
            def f(z):
                return np.zeros(z.shape[1:]).T

        return f

    def _compute_S_matrix(self, R, Rc, t_diag):
        """Compute correlation matrix for observed variables."""
        t_diagc = np.sqrt(1 - t_diag**2)
        S = R * np.outer(t_diag, t_diag) + Rc * np.outer(t_diagc, t_diagc)
        np.fill_diagonal(S, 1)  # to be save
        return S

    def p(self, loc_true, scale_true, vars=None, offset=0, rescale_cond=None,
          obs=None, R_true=None, shape_true=None, shape_cond=None,
          return_log=False, aggregate_S='mean', pool=None):
        """Compute likelihood of model parameters.

        This function computes the probability of finding the observational
        data for variables listed by `vars` given the distribution parameters
        `loc_true` (loc), `scale_true` (scale), `shape_true` (shape) of the
        true data, the parameters `self.obs.ev` (scale) and `shape_cond`
        (shape) of the conditional distribution between observed and true
        data variables, and the correlation between true data variables
        as described by the correlation matrix `R_true`. When interpreted as
        a function of model parameters for a given observational data set, the
        return value of p() is the likelihood of the given parameter values.

        In the following, Nobs (=self.obs.Nobs) refers to the number of
        observations, Nvar (=self.obs.num_var) is the number of variables per
        observation as defined by the self.obs or `obs`,
        Npar (=scipy.stats.<distribution>.numargs) is the number of shape
        parameters of a distribution, and Nmod is the number of simultaneous
        model calculations.

        The distribution parameters `loc_true` and `scale_true` are lists of
        length Nvar containing numpy arrays of a shape that can be broadcasted
        to shape (Nobs, Nmod). In the simplest case, `loc_true` and
        `scale_true` are lists of Nvar floats, i.e., they provide overall
        the location and scale parameters for the 1-dimensional distributions
        (p_true) of each variable.

        The distribution parameters `shape_true` and `shape_cond` are lists
        of length Nvar containing numpy arrays of a shape that can be
        broadcasted to shape (Npar, Nobs, Nmod).

        The final dimension (Nmod) of the distribution parameters allows for
        simultaneous model calculations which increases computational
        efficiency.

        Parameters
        ----------
        loc_true : list of arrays
            The location parameter of the p_true distribution.
        scale_true : list of arrays
            The scale parameter of the p_true distribution.
        vars : list of int or None
            Variables considered when computing the likelihood. `vars` is a
            list of integers with 0 referring to the first variable and Nvar-1
            referring to the last variable (see class Observation for details).
            Distribution parameters have to specified for all Nvar variables,
            not only for the ones in `vars`. (the default is None, implying all
            variables are considered).
        offset : array_like or float
            Offset for the observed data variables. This parameter can be a
            number, an array of length Nvar, or an array of shape
            (Nvar, Nobs). In the latter case, the offset
            is observation specific. (the default is 0, i.e., no offset).
        rescale_cond : array_like or float
            Rescale measurement uncertainties by this factor. This parameter
            can be a number, an array of length Nvar, or an array of shape
            (Nvar, Nobs). In the latter case, the rescaling
            is observation specific. (the default is None, i.e., no rescaling).
        obs : instance of class `Observation` or None
            If provided, this observational data set is used instead of the
            one given during the initialization of `Likelihood`. `obs` needs to
            have the same number of variables as the one provided during the
            Likelihood initialization. (the default is None).
        R_true : array or None
            Correlation matrix between true data variables. R_true should be
            an array of shape (Nvar, Nvar) or None. In
            the latter case, R_true is assumed to be the identity matrix, i.e.,
            different variables are uncorrelated (the default is None).
        shape_true : list of list of arrays or None
            Shape parameters of the p_true distribution. Please refer to the
            documentation of the p_true distribution to see whether and how
            many shape parameters are required (the default is None, implying
            no shape parameters are provided).
        shape_cond : list of list of arrays or None
            Shape parameters of the p_cond distribution. Please refer to the
            documentation of the p_cond distribution to see whether and how
            many shape parameters are required (the default is None, implying
            no shape parameters are provided).
        return_log : bool
            If true, return ln of the likelihood (the default is False).
        aggregate_S : str or None
            If set to 'mean', 'max', or 'min', the correlation matrix
            between observed data values is computed in an aggregated way
            for reasons of computational efficiency (only matters if Nmod > 1).
            If set to None the correlation matrix is computed for each of the
            `Nmod` models separately (the default is 'mean').
        pool : object or None
            If pool object is provided, the likelihood calculation is
            parallelized by dividing the data set in equal chunks and by
            computing the likelihood for each chunk on a seperate task/process.
            The pool object needs to support a map function. The functionality
            has been tested with the multiprocessing and MPI pools provided by
            the schwimmbad python package. If set to None, the likelihood
            calculation is not explicitly parallelized (default is None).

        Returns
        -------
        array
            Probability of the observational data given the model parameters.
            The function returns an array of shape (Nobs, Nmod).
        """
        if not obs and not self.obs:
            raise ValueError('Parameter `obs` needs to be provided during '
                             'initialization of instance of class '
                             '`Likelihood` or when calling `p()`.')
        if not obs:
            obs = self.obs

        if not vars:  # use all variables
            vars = list(range(obs.num_var))

        offset = np.array(offset)

        if R_true is None:
            R_true = np.identity(obs.num_var)
        else:
            R_true = np.array(R_true)
            assert R_true.ndim == 2
            assert np.allclose(np.diag(R_true), 1.)
            assert np.allclose(R_true, R_true.T)
            assert np.all(R_true) >= -1 and np.all(R_true) <= 1

        def convert_to_list_of_array(x, name, depth=1):
            if x is None:
                x = [np.array([])] * obs.num_var
                return x

            if not isinstance(x, (list, tuple)):
                try:
                    x = list(x)
                except TypeError:
                    raise TypeError('Parameter {} needs to be a list/tuple or '
                                    'convertable to a list.'.format(name))

            if len(x) != obs.num_var:
                raise TypeError('Parameter {} needs to be a list/tuple of '
                                'length {}.'.format(name, obs.num_var))

            if depth == 1:
                x = [np.array(e) for e in x]
            elif depth == 2:
                x = [np.atleast_1d(e) for e in x]

            return x

        loc_true = convert_to_list_of_array(loc_true, 'loc_true')
        scale_true = convert_to_list_of_array(scale_true, 'scale_true')
        shape_true = convert_to_list_of_array(shape_true, 'shape_true', 2)
        shape_cond = convert_to_list_of_array(shape_cond, 'shape_cond', 2)

        # check consistency of loc_true and scale_true shapes
        Nmod = 1
        for ivar, var in enumerate(vars):
            for s, l in zip([loc_true[var], scale_true[var]],
                            ['loc_true', 'scale_true']):
                if ((s.ndim > 0)
                        and (s.shape[0] != obs.Nobs) and (s.shape[0] != 1)):
                    raise ValueError(
                        '{} for variable {} should be numpy array of '
                        'shape ({},...) or (1,...) but found {}'.format(
                            l, var, obs.Nobs, s.shape))
                elif s.ndim > 1:
                    if ((Nmod != s.shape[1])
                            and (Nmod != 1) and (s.shape[1] != 1)):
                        raise ValueError(
                            'Inconsistent number of simultaneous model '
                            'calculations (Nmod) found in variable {} of {}: '
                            '{} != {}'.format(var, l, Nmod, s.shape[1]))
                    else:
                        Nmod = s.shape[1]

        # check consistency of shape_cond and shape_true shapes
        for ivar, var in enumerate(vars):
            for p, s, l in zip([self.p_cond[var], self.p_true[var]],
                               [shape_cond[var], shape_true[var]],
                               ['_cond', '_true']):
                if not p:
                    if s.size > 0:
                        raise ValueError(
                            'shape{} is not empty for variable {}'
                            ' but no parameters are required'.format(l, var))
                elif (s.ndim > 0) and (s.shape[0] != p.numargs):
                    raise ValueError('shape{} provides {} parameters for '
                                     'variable {} but {} are required'.format(
                                        l, s.shape[0], var, p.numargs))
                else:
                    for ss in s:
                        if ss.ndim > 0:
                            if ((ss.shape[0] != obs.Nobs)
                                    and (ss.shape[0] != 1)):
                                raise ValueError(
                                    'shape{} for variable {} should be numpy '
                                    'array of shape ({},{},...) or ({},1,...) '
                                    'but found {}'.format(
                                        l, var, p.numargs, obs.Nobs,
                                        p.numargs, ss.shape))
                        elif ss.ndim > 1:
                            if ((Nmod != ss.shape[1])
                                    and (Nmod != 1) and (ss.shape[1] != 1)):
                                raise ValueError(
                                    'Inconsistent number of simultaneous '
                                    'model calculations (Nmod) found in '
                                    'variable {} of {}: {} != {}'.format(
                                        var, l, Nmod, ss.shape[1]))
                            else:
                                Nmod = ss.shape[1]

        Nobs = obs.Nobs     # No. of observations

        v = obs.v - offset
        ev = obs.ev
        if rescale_cond:
            ev = ev * rescale_cond

        cv = obs.cv
        lim = obs.lim - offset
        limt = obs.limt

        # check uncertainties are provided for all non-missing values unless
        # p_cond == None (i.e., no measurement uncertainties)
        for var in vars:
            if self.p_cond[var] is not None:
                sel = np.isnan(ev[:, var]) & (cv[:, var])
                if np.sum(sel) > 0:
                    raise ValueError(
                        'Variable \'{}\' has a censored value but '
                        'no uncertainty is provided.'.format(var))
                sel = (np.isnan(ev[:, var])
                       & (~cv[:, var] & ~np.isnan(v[:, var])))
                if np.sum(sel) > 0:
                    raise ValueError(
                        'Variable \'{}\' has a regular value but '
                        'no uncertainty is provided.'.format(var))

        # broadcast distribution parameters
        for ivar, var in enumerate(vars):
            loc_true[var] = np.broadcast_to(loc_true[var], (Nobs, Nmod))
            scale_true[var] = np.broadcast_to(scale_true[var], (Nobs, Nmod))
            if self.p_true[var]:
                Npar = self.p_true[var].numargs
                if Npar > 0:
                    if shape_true[var].ndim == 0:
                        # same shapes for Npar x Nobs x Nmod
                        shape_true[var] = shape_true[var][None, None, None]
                    elif shape_true[var].ndim == 1:
                        # same shapes for Nobs x Nmod
                        shape_true[var] = shape_true[var][:, None, None]
                    elif shape_true[var].ndim == 2:
                        # same shapes for Nmod
                        shape_true[var] = shape_true[var][:, :, None]
                    shape_true[var] = np.broadcast_to(
                        shape_true[var], (Npar, Nobs, Nmod))
                else:
                    shape_true[var] = shape_true[var].reshape(Npar, Nobs, Nmod)
            if self.p_cond[var]:
                Npar = self.p_cond[var].numargs
                if Npar > 0:
                    if shape_cond[var].ndim == 0:
                        # same shapes for Npar x Nobs x Nmod
                        shape_cond[var] = shape_cond[var][None, None, None]
                    elif shape_cond[var].ndim == 1:
                        # same shapes for Nobs x Nmod
                        shape_cond[var] = shape_cond[var][:, None, None]
                    elif shape_cond[var].ndim == 2:
                        # same shapes for Nmod
                        shape_cond[var] = shape_cond[var][:, :, None]
                    shape_cond[var] = np.broadcast_to(
                        shape_cond[var], (Npar, Nobs, Nmod))
                else:
                    shape_cond[var] = shape_cond[var].reshape(Npar, Nobs, Nmod)

        for ivar, var in enumerate(vars):
            obs_cen = cv[:, var]
            v[obs_cen, var] = lim[obs_cen, var]

        if Nobs == 0:

            p = np.zeros((0, Nmod))

        elif pool is not None:

            size = max(pool.size, 1)

            splits = np.array_split(range(Nobs), size)

            chunks = []
            for rank in range(size):

                if len(splits[rank]) == 0:
                    continue

                j0 = splits[rank][0]
                j1 = splits[rank][-1]+1

                l_loc_true = []
                l_scale_true = []
                l_shape_true = []
                l_shape_cond = []
                for var in range(obs.num_var):
                    if var in vars:
                        l_loc_true.append(loc_true[var][j0:j1, :])
                        l_scale_true.append(scale_true[var][j0:j1, :])
                        if self.p_true[var]:
                            l_shape_true.append(shape_true[var][:, j0:j1, :])
                        else:
                            l_shape_true.append([])
                        if self.p_cond[var]:
                            l_shape_cond.append(shape_cond[var][:, j0:j1, :])
                        else:
                            l_shape_cond.append([])
                    else:
                        l_loc_true.append([])
                        l_scale_true.append([])
                        l_shape_true.append([])
                        l_shape_cond.append([])

                chunks.append(
                    [Nmod, vars, v[j0:j1], ev[j0:j1], cv[j0:j1], limt[j0:j1],
                     l_loc_true, l_scale_true, l_shape_cond, l_shape_true,
                     R_true, obs.Rc[j0:j1],
                     aggregate_S, obs.correlated_errors])

            try:
                p = np.concatenate(pool.map(self._p_arg_list, chunks))
            except TypeError:
                p = np.concatenate(list(pool.map(self._p_arg_list, chunks)))

        else:
            p = self._p(Nmod, vars, v, ev, cv, limt,
                        loc_true, scale_true, shape_cond, shape_true,
                        R_true, obs.Rc, aggregate_S, obs.correlated_errors)

        if return_log:
            return np.log(p)
        else:
            return p

    def _p_arg_list(self, chunks):
        return self._p(*chunks)

    def _p(self, Nmod, vars, v, scale_cond, cv, limit_types,
           loc_true, scale_true, shape_cond, shape_true, R, Rc, aggregate_S,
           correlated_errors):
        """Core routine to compute likelihood - Please call p() instead."""

        Nobs = v.shape[0]
        Nvar = len(vars)

        vars_array = np.array(vars)
        R_true = R[np.ix_(vars_array, vars_array)]

        correlated_errors = Rc is not None

        f_obs = np.zeros((Nobs, Nvar, Nmod))
        F_obs = np.zeros_like(f_obs)

        p_cond_plus = np.zeros_like(f_obs)
        p_cond_minus = np.zeros_like(f_obs)
        x_plus = np.zeros_like(f_obs)

        prob = np.ones((Nobs, Nmod))
        eps = 1e-300

        z = np.ones((Nobs, Nvar, Nmod))
        dzdv = np.ones_like(z)
        scale = np.ones_like(z)

        trivial_R = np.all(np.isclose(R_true, np.identity(Nvar)))
        if self.verbosity > 0:
            print('Latent variables are correlated: {}'.format(
                not trivial_R))
        if self.verbosity > 0:
            print('Measurement errors are correlated: {}'.format(
                correlated_errors))
        correlated_observables = True
        if trivial_R and not correlated_errors:
            correlated_observables = False
        if self.verbosity > 0:
            print('Observed variables are correlated: {}'.format(
                correlated_observables))

        for ivar, var in enumerate(vars):

            obs_cen = cv[:, var]

            obs_nomiss = ~np.isnan(v[:, var])
            num_nomiss = np.sum(obs_nomiss)
            obs_nocen = (np.logical_not(obs_cen) & obs_nomiss)

            t_v = v[obs_nomiss, var].reshape(num_nomiss, 1)
            t_lt = loc_true[var][obs_nomiss]
            # t_sc = np.outer(scale_cond[obs_nomiss, var], np.ones(Nmod))
            t_sc = scale_cond[obs_nomiss, var].reshape(num_nomiss, 1)

            if self.p_obs[var].name == 'norm':

                if self.p_cond[var] is None:
                    scale[obs_nomiss, ivar] = scale_true[var][obs_nomiss]
                else:
                    scale[obs_nomiss, ivar] = np.sqrt(
                        (scale_cond[obs_nomiss, var]**2).reshape(num_nomiss, 1)
                        + scale_true[var][obs_nomiss]**2)

                # compute z (standard normal distributed observable)
                # z = inv_Phi(CDF_obs(y)) and dz/dy = PDF_obs(y)/phi(z)
                z[obs_nomiss, ivar] = (t_v - t_lt) / scale[obs_nomiss, ivar]
                dzdv[obs_nocen, ivar] = 1. / scale[obs_nocen, ivar]

            else:
                if self.p_cond[var] is None:

                    F_obs[obs_nomiss, ivar] = self.p_obs[var].cdf(
                        t_v, *shape_true[var][:, obs_nomiss], t_lt,
                        scale_true[var][obs_nomiss])

                    f_obs[obs_nomiss, ivar] = self.p_obs[var].pdf(
                        t_v, *shape_true[var][:, obs_nomiss], t_lt,
                        scale_true[var][obs_nomiss])

                else:

                    f_obs[obs_nomiss, ivar] = self.p_obs[var].pdf(
                        t_v, *shape_cond[var][:, obs_nomiss],
                        *shape_true[var][:, obs_nomiss],
                        t_sc, t_lt, scale_true[var][obs_nomiss])

                    F_obs[obs_nomiss, ivar] = self.p_obs[var].cdf(
                        t_v, *shape_cond[var][:, obs_nomiss],
                        *shape_true[var][:, obs_nomiss],
                        t_sc, t_lt, scale_true[var][obs_nomiss])

                # compute z (standard normal distributed observable)
                # z = inv_Phi(CDF_obs(y)) and dz/dy = PDF_obs(y)/phi(z)
                z[obs_nomiss, ivar] = scipy.stats.norm.ppf(
                    F_obs[obs_nomiss, ivar])
                dzdv[obs_nocen, ivar] = (
                    (eps + f_obs[obs_nocen, ivar])
                    / (eps + scipy.stats.norm._pdf(z[obs_nocen, ivar])))

            if ((not trivial_R) and (self.p_cond[var] is not None)
                    and (self.p_obs[var].name != 'norm') and num_nomiss > 0):
                # auxiliary data to compute t_diag (true-obs correlation)
                # define x_plus, p_cond_plus, p_cond_minus
                true_plus = t_v
                x_plus[obs_nomiss, ivar] = scipy.stats.norm.ppf(
                    self.p_true[var].cdf(
                        true_plus, *shape_true[var][:, obs_nomiss],
                        loc=loc_true[var][obs_nomiss],
                        scale=scale_true[var][obs_nomiss]))

                true_minus = self.p_true[var].ppf(
                    scipy.stats.norm.cdf(-x_plus[obs_nomiss, ivar]),
                    *shape_true[var][:, obs_nomiss],
                    loc=loc_true[var][obs_nomiss],
                    scale=scale_true[var][obs_nomiss])

                #loc_xy, scale_xy, *shape_xy = self.p_obs[var].param_map_XY(
                #    true_plus, t_sc, *shape_cond[var][:, obs_nomiss])
                loc_xy, scale_xy, *shape_xy = (true_plus,
                    t_sc, *shape_cond[var][:, obs_nomiss])
                p_cond_plus[obs_nomiss, ivar] = self.p_cond[var].pdf(
                    t_v, *shape_xy, loc=loc_xy, scale=scale_xy)

                #loc_xy, scale_xy, *shape_xy = self.p_obs[var].param_map_XY(
                #    true_minus, t_sc, *shape_cond[var][:, obs_nomiss])
                loc_xy, scale_xy, *shape_xy = (
                    true_minus, t_sc, *shape_cond[var][:, obs_nomiss])
                p_cond_minus[obs_nomiss, ivar] = self.p_cond[var].pdf(
                    t_v, *shape_xy, loc=loc_xy, scale=scale_xy)

        t_diag = np.zeros((Nvar, Nmod))

        for i in range(Nobs):

            nomiss = ~np.isnan(v[i, vars_array])
            num_nomiss = np.sum(nomiss)

            if num_nomiss == 0:
                continue

            # non-missing variables
            sel_nomiss = nomiss.nonzero()[0]

            # non-censored and non-missing variables (counting relative
            # to non-missing variables)
            sel_nocen = (
                np.logical_not(cv[i, vars_array[nomiss]])).nonzero()[0]

            # censored and non-missing variables (counting relative
            # to non-missing variables)
            sel_cen = (cv[i, vars_array[nomiss]]).nonzero()[0]

            # type (upper/lower) of detections limits of censored variables
            limit_type = limit_types[i, sel_cen]

            # -- No correlation among the variables --
            if not correlated_observables:

                # Split into non-censored and censored variables
                psi = self._normal_partial_integral_uncorr(sel_nocen, sel_cen,
                                                           limit_type)

                # find models that contain one of more z=+-inf => p=0
                lz = z[i, sel_nomiss]
                sel = (
                    (np.sum(np.isneginf(lz), axis=0) > 0)
                    | (np.sum(np.isposinf(lz[sel_nocen]), axis=0) > 0))
                prob[i, sel] = 0.

                sel = ~sel
                num_sel = np.sum(sel)
                if num_sel > 0:
                    l_z = z[i][np.ix_(sel_nomiss, sel)].reshape(
                        num_nomiss, num_sel)
                    l_dzdv = dzdv[i][np.ix_(sel_nomiss, sel)].reshape(
                        num_nomiss, num_sel)
                    prob[i, sel] = psi(l_z) * np.prod(l_dzdv, axis=0)

            else:  # -- Correlation between variables --

                Rc_true = Rc[i][np.ix_(vars_array, vars_array)]

                # Compute t_diag: correlation between true and observed vars
                for ivar, var in enumerate(vars):

                    if ivar not in sel_nomiss:
                        continue

                    if self.p_cond[var] is None:

                        t_diag[ivar, :] = 1

                    elif self.p_obs[var].name == 'norm':

                        t_diag[ivar, :] = scale_true[var][i] / scale[i, ivar]

                    else:
                        with np.errstate(divide='ignore', invalid='ignore',
                                         over='ignore'):
                            lnp = np.log(p_cond_plus[i, ivar]
                                         / p_cond_minus[i, ivar])
                            zx = z[i, ivar]*x_plus[i, ivar]
                            aux = lnp / zx

                        t_diag[ivar] = 0.

                        sel = np.isinf(aux)
                        t_diag[ivar, sel] = np.sign(aux[sel])
                        sel = np.isnan(lnp)
                        t_diag[ivar, sel] = 0.
                        sel = np.isnan(aux) & ~np.isnan(lnp)
                        t_diag[ivar, sel] = (np.sign(lnp[sel])
                                             * np.sign(zx[sel]))

                        sel = (np.isinf(aux)) | (np.isnan(aux))
                        aux[sel] = 0.

                        sel = (aux > 0)
                        t_diag[ivar, sel] = (
                            -1/aux[sel] + np.sqrt(1/aux[sel]**2 + 1))
                        sel = (aux < 0)
                        t_diag[ivar, sel] = (
                            -1/aux[sel] - np.sqrt(1/aux[sel]**2 + 1))

                        if (np.any(np.isnan(t_diag[ivar]))
                            or np.any(t_diag[ivar] < -1)
                                or np.any(t_diag[ivar] > 1)):
                            warntxt = ('Likelihood computed correlation '
                                       'outside allowed range {}!'.format(
                                            t_diag[ivar]))
                            warnings.warn(warntxt, RuntimeWarning)
                            import pdb
                            pdb.set_trace()

                if aggregate_S in ['mean', 'max', 'min']:

                    if aggregate_S == 'mean':  # average
                        t_diag_agg = np.mean(t_diag, axis=1)
                    elif aggregate_S == 'max':
                        t_diag_agg = np.max(t_diag, axis=1)
                    elif aggregate_S == 'min':
                        t_diag_agg = np.min(t_diag, axis=1)

                    # Correlation between observed variables
                    S = self._compute_S_matrix(
                        R_true[np.ix_(sel_nomiss, sel_nomiss)],
                        Rc_true[np.ix_(sel_nomiss, sel_nomiss)],
                        t_diag_agg[sel_nomiss])

                    psi = self._normal_partial_integral(
                        sel_nocen, sel_cen, S, limit_type)

                    # probability of the observations given model params
                    lz = z[i, sel_nomiss]
                    sel = (
                        (np.sum(np.isneginf(lz), axis=0) > 0)
                        | (np.sum(np.isposinf(lz[sel_nocen]), axis=0) > 0))
                    prob[i, sel] = 0.

                    sel = ~sel
                    num_sel = np.sum(sel)
                    if num_sel > 0:
                        l_z = z[i][np.ix_(sel_nomiss, sel)].reshape(
                            num_nomiss, num_sel)
                        l_dzdv = dzdv[i][np.ix_(sel_nomiss, sel)].reshape(
                            num_nomiss, num_sel)
                        prob[i, sel] = psi(l_z) * np.prod(l_dzdv, axis=0)

                else:

                    for k in range(Nmod):

                        lz = z[i, sel_nomiss]
                        if (np.any(np.isneginf(lz[:, k]))
                                or np.any(np.isposinf(lz[sel_nocen, k]))):
                            prob[i, k] = 0.
                            continue

                        # Correlation between observed variables
                        S = self._compute_S_matrix(
                            R_true[np.ix_(sel_nomiss, sel_nomiss)],
                            Rc_true[np.ix_(sel_nomiss, sel_nomiss)],
                            t_diag[sel_nomiss, k])

                        psi = self._normal_partial_integral(
                            sel_nocen, sel_cen, S, limit_type)

                        # probability of the observations given model params
                        prob[i, k] = (psi(z[i, sel_nomiss, k])
                                      * np.prod(dzdv[i, sel_nomiss, k]))

        return prob


if __name__ == '__main__':

    import doctest
    doctest.testmod()
