"""Simple test cases for the leopy.likelihood module.

   This module is part of LEO-Py -- \
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
import sys
import time

import pytest

import leopy

@pytest.fixture
def obs_norm_no_error():
    np.random.seed(10)
    dist = scipy.stats.norm
    Ndata = 100
    rho = 0.5
    R = np.array([[1., rho], [rho, 1.]])
    loc_true = [-1., 2.]
    scale_true = [1., 3.]
    x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)
    y = dist.ppf(scipy.stats.norm.cdf(x), loc=loc_true, scale=scale_true)
    y *= scale_true / np.std(y, axis=0)
    y += loc_true - np.mean(y, axis=0)
    ey = np.zeros_like(y)
    df = pd.DataFrame(np.array([y[:, 0], y[:, 1], ey[:, 0], ey[:, 1]]).T,
                      columns=['v0', 'v1', 'e_v0', 'e_v1'])
    return leopy.Observation(df, 'test', verbosity=0)

@pytest.fixture
def obs_lognorm_no_error():
    np.random.seed(14)
    dist = scipy.stats.lognorm
    Ndata = 100
    rho = 0.5
    R = np.array([[1., rho], [rho, 1.]])
    loc_true = np.array([0., 2.])
    scale_true = np.array([1., 3.])
    shape_true = np.array([0.5, 1.5])
    x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)
    y = dist.ppf(scipy.stats.norm.cdf(x),
                 shape_true, loc=loc_true, scale=scale_true)
    ey = np.zeros_like(y)
    df = pd.DataFrame(np.array([y[:, 0], y[:, 1], ey[:, 0], ey[:, 1]]).T,
                      columns=['v0', 'v1', 'e_v0', 'e_v1'])
    return leopy.Observation(df, 'test', verbosity=0)

@pytest.fixture
def obs_norm_cen():
    np.random.seed(16)
    dist = scipy.stats.norm
    Ndata = 200
    rho = 0.5
    R = np.array([[1., rho], [rho, 1.]])
    loc_true = np.array([0.5, 1.5])
    scale_true = np.array([1., 2.5])
    x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)
    y = dist.ppf(scipy.stats.norm.cdf(x),
                 loc=loc_true, scale=scale_true)
    y_true = np.copy(y)
    ey = np.zeros_like(y)
    ey[:, 0] = 0.2
    ey[:, 1] = 0.1
    y[:, 0] += ey[:, 0] * np.random.randn(Ndata)
    y[:, 1] += ey[:, 1] * np.random.randn(Ndata)

    lower_limit = 0.3*np.random.randn(Ndata)+0.3
    lower_limit[lower_limit<0.05] = 0.05
    upper_limit = 0.6*np.random.randn(Ndata)+2.5
    upper_limit[upper_limit<0.2] = 0.2
    cy = np.zeros_like(y).astype(bool)
    ly = -np.infty*np.ones_like(y)
    uy = np.infty*np.ones_like(y)
    for i in range(2):
        sel = y[:, i] < lower_limit
        y[sel, i] = float('NaN')
        cy[sel, i] = True
        ly[sel, i] = lower_limit[sel]
        sel = y[:, i] > upper_limit
        y[sel, i]=float('NaN')
        cy[sel, i] = True
        uy[sel, i] = upper_limit[sel]
    df = pd.DataFrame(np.array([y[:, 0], y[:, 1], ey[:, 0], ey[:, 1],
                                cy[:, 0], cy[:, 1], ly[:, 0], ly[:, 1],
                                uy[:, 0], uy[:, 1]]).T,
                  columns=['v0', 'v1', 'e_v0', 'e_v1', 'c_v0', 'c_v1', 'l_v0',
                           'l_v1', 'u_v0', 'u_v1'])
    return leopy.Observation(df, 'test', verbosity=0)

@pytest.fixture
def obs_norm_cen_uncorr():
    np.random.seed(16)
    dist = scipy.stats.norm
    Ndata = 200
    rho = 0.
    R = np.array([[1., rho], [rho, 1.]])
    loc_true = np.array([0.5, 1.5])
    scale_true = np.array([1., 2.5])
    x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)
    y = dist.ppf(scipy.stats.norm.cdf(x),
                 loc=loc_true, scale=scale_true)
    y_true = np.copy(y)
    ey = np.zeros_like(y)
    ey[:, 0] = 0.2
    ey[:, 1] = 0.1
    y[:, 0] += ey[:, 0] * np.random.randn(Ndata)
    y[:, 1] += ey[:, 1] * np.random.randn(Ndata)

    lower_limit = 0.3*np.random.randn(Ndata)+0.3
    lower_limit[lower_limit<0.05] = 0.05
    upper_limit = 0.6*np.random.randn(Ndata)+2.5
    upper_limit[upper_limit<0.2] = 0.2
    cy = np.zeros_like(y).astype(bool)
    ly = -np.infty*np.ones_like(y)
    uy = np.infty*np.ones_like(y)
    for i in range(2):
        sel = y[:, i] < lower_limit
        y[sel, i] = float('NaN')
        cy[sel, i] = True
        ly[sel, i] = lower_limit[sel]
        sel = y[:, i] > upper_limit
        y[sel, i]=float('NaN')
        cy[sel, i] = True
        uy[sel, i] = upper_limit[sel]
    df = pd.DataFrame(np.array([y[:, 0], y[:, 1], ey[:, 0], ey[:, 1],
                                cy[:, 0], cy[:, 1], ly[:, 0], ly[:, 1],
                                uy[:, 0], uy[:, 1]]).T,
                  columns=['v0', 'v1', 'e_v0', 'e_v1', 'c_v0', 'c_v1',
                           'l_v0', 'l_v1', 'u_v0', 'u_v1'])
    return leopy.Observation(df, 'test', verbosity=0)

@pytest.fixture
def obs_norm_MAR():
    np.random.seed(16)
    dist = scipy.stats.norm
    Ndata = 100
    rho = 0.5
    R = np.array([[1., rho], [rho, 1.]])
    loc_true = np.array([0., 2.])
    scale_true = np.array([1., 3.])
    x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)
    y = dist.ppf(scipy.stats.norm.cdf(x),
                 loc=loc_true, scale=scale_true)
    y_true = np.copy(y)
    ey = np.zeros_like(y)
    ey[:, 0] = 0.2
    ey[:, 1] = 0.1
    y[:, 0] += ey[:, 0] * np.random.randn(Ndata)
    y[:, 1] += ey[:, 1] * np.random.randn(Ndata)
    def logistic(x):
        return np.exp(x) / (np.exp(x) + 1.)
    m1 = scipy.stats.bernoulli.rvs(logistic(y[:, 0]-1.)).astype(bool)  # for col 1
    m0 = scipy.stats.bernoulli.rvs(logistic(y[:, 1]-2.)).astype(bool)  # for col 0
    y[m1, 1] = np.float('NaN')
    y[m0, 0] = np.float('NaN')
    df = pd.DataFrame(np.array([y[:, 0], y[:, 1], ey[:, 0], ey[:, 1]]).T,
                  columns=['v0', 'v1', 'e_v0', 'e_v1'])
    return leopy.Observation(df, 'test', verbosity=0)

@pytest.fixture
def obs_lognorm_MAR():
    np.random.seed(2)
    dist = scipy.stats.lognorm
    Ndata = 100
    rho = 0.5
    R = np.array([[1., rho], [rho, 1.]])
    loc_true = np.array([0., 2.])
    scale_true = np.array([1., 3.])
    shape_true = np.array([0.5, 1.5])
    x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)
    y = dist.ppf(scipy.stats.norm.cdf(x),
                 shape_true, loc=loc_true, scale=scale_true)
    y_true = np.copy(y)
    ey = np.zeros_like(y)
    ey[:, 0] = 0.2
    ey[:, 1] = 0.1
    y[:, 0] += ey[:, 0] * np.random.randn(Ndata)
    y[:, 1] += ey[:, 1] * np.random.randn(Ndata)
    def logistic(x):
        return np.exp(x) / (np.exp(x) + 1.)
    m1 = scipy.stats.bernoulli.rvs(logistic(y[:, 0]-2.)).astype(bool)  # for col 1
    m0 = scipy.stats.bernoulli.rvs(logistic(y[:, 1]-5.)).astype(bool)  # for col 0
    y[m1, 1] = np.float('NaN')
    y[m0, 0] = np.float('NaN')
    df = pd.DataFrame(np.array([y[:, 0], y[:, 1], ey[:, 0], ey[:, 1]]).T,
                  columns=['v0', 'v1', 'e_v0', 'e_v1'])
    return leopy.Observation(df, 'test', verbosity=0)

@pytest.fixture
def obs_norm_obscorr():
    np.random.seed(10)
    dist = scipy.stats.norm
    Ndata = 100
    rho = 0.5
    R = np.array([[1., rho], [rho, 1.]])
    loc_true = [-1., 2.]
    scale_true = [1., 3.]
    x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)
    y = dist.ppf(scipy.stats.norm.cdf(x), loc=loc_true, scale=scale_true)
    y *= scale_true / np.std(y, axis=0)
    y += loc_true - np.mean(y, axis=0)

    sigma_c = [1., 1.5]
    ey = np.zeros_like(y)
    ey[:, 0] = sigma_c[0]
    ey[:, 1] = sigma_c[1]
    rho_c = np.zeros(Ndata)
    error_y = np.zeros_like(y)
    for i in range(Ndata):
        rho_c[i] = 0.01*np.random.rand()-0.99
        R_c = np.array([[1., rho_c[i]], [rho_c[i], 1.]])
        cov_c = np.diag(sigma_c).dot(R_c.dot(np.diag(sigma_c)))
        error_y[i, :] = scipy.stats.multivariate_normal.rvs(cov=cov_c)
    y += error_y

    df = pd.DataFrame(np.array([y[:, 0], y[:, 1], ey[:, 0], ey[:, 1],
                                rho_c]).T,
                      columns=['v0', 'v1', 'e_v0', 'e_v1', 'r_v0_v1'])

    return leopy.Observation(df, 'test', verbosity=0)

@pytest.fixture
def obs_lognorm_obscorr():
    np.random.seed(19)
    dist = scipy.stats.lognorm
    Ndata = 100
    rho = 0.5
    R = np.array([[1., rho], [rho, 1.]])
    loc_true = np.array([0., 2.])
    scale_true = np.array([1., 3.])
    shape_true = np.array([0.5, 1.5])
    x = scipy.stats.multivariate_normal.rvs(cov=R, size=Ndata)
    y = dist.ppf(scipy.stats.norm.cdf(x),
                 shape_true, loc=loc_true, scale=scale_true)

    sigma_c = [0.1, 0.2]
    ey = np.zeros_like(y)
    ey[:, 0] = sigma_c[0]
    ey[:, 1] = sigma_c[1]
    rho_c = np.zeros(Ndata)
    error_y = np.zeros_like(y)
    for i in range(Ndata):
        rho_c[i] = 0.99*2*(np.random.rand()-0.5)
        R_c = np.array([[1., rho_c[i]], [rho_c[i], 1.]])
        cov_c = np.diag(sigma_c).dot(R_c.dot(np.diag(sigma_c)))
        error_y[i, :] = scipy.stats.multivariate_normal.rvs(cov=cov_c)
    y += error_y

    df = pd.DataFrame(np.array([y[:, 0], y[:, 1], ey[:, 0], ey[:, 1],
                                rho_c]).T,
                      columns=['v0', 'v1', 'e_v0', 'e_v1', 'r_v0_v1'])
    return leopy.Observation(df, 'test', verbosity=0)

@pytest.mark.usefixtures('pool')
class TestLikelihood:

    def test_1(self):
        d = {'v0': [1, 2], 'e_v0': [0.1, 0.2],
             'v1': [3, 4], 'e_v1': [0.1, 0.1]}
        df = pd.DataFrame(d)
        obs = leopy.Observation(df, 'testdata', verbosity=0)
        like = leopy.Likelihood(obs, p_true='norm', verbosity=-1)
        stddev = [1, 2]
        mean = [0.5, 0.7]
        p = like.p(mean, stddev, pool=self.pool)
        p_v1 = scipy.stats.norm.pdf(df['v0']-mean[0], scale=stddev[0])
        p_v2 = scipy.stats.norm.pdf(df['v1']-mean[1], scale=stddev[1])
        assert np.all(np.isclose(p.T[0], p_v1*p_v2))

    def test_2(self):
        d = {'v0': [1., 2., -4.], 'e_v0': [0.1, 0.2, 0.3],
             'v1': [3., 4., 1.], 'e_v1': [0.1, 0.1, 0.1]}
        df = pd.DataFrame(d)
        obs = leopy.Observation(df, 'testdata', verbosity=0)
        like = leopy.Likelihood(obs, p_true='norm', verbosity=-1)
        R = np.array([[1, 0.6], [0.6, 1]])
        stddev = [1, 2]
        mean = [0.5, 0.7]
        cov = np.diag(stddev).dot(R.dot(np.diag(stddev)))
        p = like.p(mean, stddev, R_true=R, pool=self.pool)
        p_v1v2 = scipy.stats.multivariate_normal.pdf(
            df[['v0', 'v1']], mean=mean, cov=cov)
        assert np.all(np.isclose(p.T[0], p_v1v2))


    def test_3(self):
        d = {'v0': [1., 2., -4.], 'e_v0': [0.1, 0.2, 0.3],
             'v1': [3., 4., 1.], 'e_v1': [0.1, 0.1, 0.1]}
        df = pd.DataFrame(d)
        obs = leopy.Observation(df, 'testdata', verbosity=0)
        like = leopy.Likelihood(obs, p_true='norm', verbosity=-1)
        R = np.array([[1, -0.3], [-0.3, 1]])
        stddev = [1, 2]
        mean = [0.5, 0.7]
        cov = np.diag(stddev).dot(R.dot(np.diag(stddev)))
        p = like.p(mean, stddev, R_true=R, pool=self.pool)
        p_v1v2 = scipy.stats.multivariate_normal.pdf(
            df[['v0', 'v1']], mean=mean, cov=cov)
        assert np.all(np.isclose(p.T[0], p_v1v2))


    def test_4(self):
        d = {'v0': [1., 2., -4.], 'e_v0': [1e-6, 1e-6, 1e-6],
             'v1': [3., 4., 1.], 'e_v1': [1e-6, 1e-6, 1e-6]}
        df = pd.DataFrame(d)
        obs = leopy.Observation(df, 'testdata', verbosity=0)
        like = leopy.Likelihood(
            obs, p_true='norm', p_cond='norm', verbosity=-1)
        R = np.array([[1, -0.3], [-0.3, 1]])
        stddev = [1, 2]
        mean = [0.5, 0.7]
        cov = np.diag(stddev).dot(R.dot(np.diag(stddev)))
        p = like.p(mean, stddev, R_true=R, pool=self.pool)
        p_v1v2 = scipy.stats.multivariate_normal.pdf(
            df[['v0', 'v1']], mean=mean, cov=cov)
        assert np.all(np.isclose(p.T[0], p_v1v2))


    def test_5(self):
        d = {'v0': [1, 2], 'e_v0': [0.1, 0.2],
             'v1': [3, 4], 'e_v1': [0.1, 0.1]}
        obs = leopy.Observation(pd.DataFrame(d), 'testdata', verbosity=0)
        like = leopy.Likelihood(obs, p_true='lognorm', verbosity=-1)
        p = like.p(
            [0.5, 0.7], [1, 2], shape_true=[[1.4], [2.]], pool=self.pool)
        assert np.all(np.isclose(p, np.array([[0.0436189 ],
                                              [0.01067159]])))


    def test_6(self):
        d = {'v0': [1, 2], 'e_v0': [1e-6, 1e-6],
             'v1': [3, 4], 'e_v1': [1e-6, 1e-6]}
        obs = leopy.Observation(pd.DataFrame(d), 'testdata', verbosity=0)
        like = leopy.Likelihood(
            obs, p_true='lognorm', p_cond='norm', verbosity=-1)
        p = like.p(
            [0.5, 0.7], [1, 2], shape_true=[[1.4], [2.]], pool=self.pool)
        assert np.all(np.isclose(p, np.array([[0.0436189 ],
                                              [0.01067159]])))


    def test_7(self):
        d = {'v0': [1, 2], 'e_v0': [0.1, 0.2],
             'v1': [3, 4], 'e_v1': [0.1, 0.1]}
        obs = leopy.Observation(pd.DataFrame(d), 'testdata', verbosity=0)
        like = leopy.Likelihood(
            obs, p_true='lognorm', p_cond='norm', verbosity=-1)
        p = like.p(
            [0.5, 0.7], [1, 2], shape_true=[[1.4], [2.]], pool=self.pool)
        assert np.all(np.isclose(p, np.array([[0.04415356], [0.01089342]]),
                                 rtol=1e-5, atol=1e-5))


    def test_8(self):
        d = {'v0': [1., 2., 0.8], 'e_v0': [1e-6, 1e-6, 1e-6],
             'v1': [3., 4., 1.], 'e_v1': [1e-6, 1e-6, 1e-6]}
        df = pd.DataFrame(d)
        obs = leopy.Observation(df, 'testdata', verbosity=0)
        like = leopy.Likelihood(
            obs, p_true='lognorm', p_cond='norm', verbosity=-1)
        R = np.array([[1, -0.3], [-0.3, 1]])
        scale = [1, 2]
        loc = [0.5, 0.]
        shape = [[1], [1.5]]
        p = like.p(loc, scale, shape_true=shape, R_true=R, pool=self.pool)
        assert np.all(
            np.isclose(
                p, np.array([[0.05819145], [0.01415945], [0.12375991]]),
                rtol=1e-5, atol=1e-5))


    def test_9(self, obs_norm_no_error):
        like = leopy.Likelihood(obs_norm_no_error, p_true='norm',
                                verbosity=-1)
        def f_mlnlike(x):
            loc_true = x[0:2]
            scale_true = x[2:4]
            rho = x[4]
            R = np.array([[1., rho], [rho, 1.]])
            pp = like.p(loc_true, scale_true, R_true=R, pool=self.pool)
            if np.sum(pp==0) > 0:
                return np.inf
            else:
                return -np.sum(np.log(pp))

        bounds = scipy.optimize.Bounds(
            [-np.inf, -np.inf, 1e-3, 1e-3, 1e-3],
            [np.inf, np.inf, np.inf, np.inf, 1-1e-3])
        optres = scipy.optimize.minimize(f_mlnlike, [0., 0., 1., 1., 0.3],
                                         bounds=bounds, method='SLSQP',
                                         options={'disp': True, 'ftol': 1e-12})
        assert np.all(np.isclose(optres.x, [
            -1,  2,  1,  3,  0.4940357],  rtol=1e-5, atol=1e-5))


    def test_10(self, obs_lognorm_no_error):
        like = leopy.Likelihood(obs_lognorm_no_error, p_true='lognorm',
                                verbosity=-1)
        def f_mlnlike(x):
            print(x)
            loc_true = x[0:2]
            scale_true = x[2:4]
            shape_true = x[4:6].reshape(2, 1)
            rho = x[6]
            R = np.array([[1., rho], [rho, 1.]])
            pp = like.p(loc_true, scale_true, shape_true=shape_true, R_true=R,
                        pool=self.pool)
            if np.sum(pp==0) > 0:
                return np.inf
            else:
                return -np.sum(np.log(pp))

        bounds = scipy.optimize.Bounds(
            [-np.inf, -np.inf, 1e-3, 1e-3, 1e-3, 1e-3, -1+1e-3],
            [np.inf, np.inf, 10., 10., 10., 10., 1-1e-3])
        optres = scipy.optimize.minimize(
            f_mlnlike, [0., 0., 1., 1., 1., 1., 0.3],
            bounds=bounds, method='SLSQP',
            options={'disp': True, 'ftol': 1e-12})
        assert np.all(np.isclose(optres.x, [
            -0.01389813,  1.98866462,  1.17630436,  3.85686233,  0.53775924,
            1.47418086,  0.54154499], rtol=1e-5, atol=1e-5))

    def test_12(self, obs_norm_MAR):
        like = leopy.Likelihood(obs_norm_MAR, p_true='norm', p_cond='norm')
        def f_mlnlike(x):
            loc_true = x[0:2]
            scale_true = x[2:4]
            rho = x[4]
            R = np.array([[1., rho], [rho, 1.]])
            pp = like.p(loc_true, scale_true, R_true=R, pool=self.pool)
            if np.sum(pp==0) > 0:
                return np.inf
            else:
                return -np.sum(np.log(pp))

        bounds = scipy.optimize.Bounds(
            [-np.inf, -np.inf, 1e-3, 1e-3,  1e-3],
            [np.inf, np.inf, 10., 10., 1-1e-3])
        optres = scipy.optimize.minimize(f_mlnlike, [0., 0., 1., 1., 0.3],
                                         bounds=bounds, method='SLSQP',
                                         options={'disp': True, 'ftol': 1e-12})
        assert np.all(np.isclose(optres.x, [
            -0.17991379,  1.49608098,  0.98586541,  2.69842305,  0.44114192],
            rtol=1e-5, atol=1e-5))


    def test_13(self, obs_norm_cen):
        like = leopy.Likelihood(obs_norm_cen, p_true='norm', p_cond='norm')
        def f_mlnlike(x):
            loc_true = x[0:2]
            scale_true = x[2:4]
            rho = x[4]
            R = np.array([[1., rho], [rho, 1.]])
            pp = like.p(loc_true, scale_true, R_true=R, pool=self.pool)
            if np.sum(pp==0) > 0:
                return np.inf
            else:
                return -np.sum(np.log(pp))

        bounds = scipy.optimize.Bounds(
            [-np.inf, -np.inf, 1e-3, 1e-3,  1e-3],
            [np.inf, np.inf, 10., 10., 1-1e-3])
        optres = scipy.optimize.minimize(f_mlnlike, [0., 0., 1., 1., 0.3],
                                         bounds=bounds, method='SLSQP',
                                         options={'disp': True, 'ftol': 1e-12})
        assert np.all(np.isclose(optres.x, [
            0.47954307, 1.2705067,  0.88797593, 2.36476421, 0.52029972],
            rtol=1e-5, atol=1e-5))


    def test_14(self, obs_norm_cen_uncorr):
        like = leopy.Likelihood(
            obs_norm_cen_uncorr, p_true='norm', p_cond='norm')
        def f_mlnlike(x):
            loc_true = x[0:2]
            scale_true = x[2:4]
            pp = like.p(loc_true, scale_true, pool=self.pool)
            if np.sum(pp==0) > 0:
                return np.inf
            else:
                return -np.sum(np.log(pp))

        bounds = scipy.optimize.Bounds(
            [-np.inf, -np.inf, 1e-3, 1e-3],
            [np.inf, np.inf, 10., 10.])
        optres = scipy.optimize.minimize(f_mlnlike, [0., 0., 1., 1.],
                                         bounds=bounds, method='SLSQP',
                                         options={'disp': True, 'ftol': 1e-12})
        assert np.all(np.isclose(optres.x, [
            0.54321826, 1.27320101, 0.97319273, 2.3491366],
            rtol=1e-5, atol=1e-5))


    def test_15(self, obs_norm_obscorr):
        t0 = time.time()
        like = leopy.Likelihood(obs_norm_obscorr, p_true='norm', p_cond='norm')
        def f_mlnlike(x):
            loc_true = x[0:2]
            scale_true = x[2:4]
            rho = x[4]
            R = np.array([[1., rho], [rho, 1.]])
            pp = like.p(loc_true, scale_true, R_true=R, pool=self.pool)
            if np.sum(pp==0) > 0:
                return np.inf
            else:
                return -np.sum(np.log(pp))

        bounds = scipy.optimize.Bounds(
            [-np.inf, -np.inf, 1e-3, 1e-3,  -1+1e-3],
            [np.inf, np.inf, 10., 10., 1-1e-3])
        optres = scipy.optimize.minimize(f_mlnlike, [0., 0., 1., 1., 0.3],
                                         bounds=bounds, method='SLSQP',
                                         options={'disp': True, 'ftol': 1e-12})
        t1 = time.time()
        print('Needed {:.4f} s'.format(t1-t0))
        print(optres.x)
        assert np.all(np.isclose(optres.x, [
            -1.08265859,  2.14778872,  1.18368684,  2.74908927,  0.49219241],
            rtol=1e-5, atol=1e-5))


    # test independence of observations
    # p(x_1, y_1, ..., x_N, y_N | theta) = \prod_{i=1}^N p(x_i, y_i | theta)
    def test_16(self, obs_lognorm_obscorr):
        like = leopy.Likelihood(obs_lognorm_obscorr, p_true='lognorm',
                                p_cond='norm')
        loc_true = [-0.02, 1.95]
        scale_true = [1.2, 2.9]
        shape_true = np.array([0.5, 1.44]).reshape(2, 1)
        rho = 0.54
        R = np.array([[1., rho], [rho, 1.]])
        p_all = like.p(loc_true, scale_true, shape_true=shape_true, R_true=R,
                       pool=self.pool)

        N = obs_lognorm_obscorr.df.shape[0]
        p_all2 = np.zeros(N)
        for i in range(N):
            obs = leopy.Observation(obs_lognorm_obscorr.df.iloc[i:i+1],
                                    'test', verbosity=0)
            like = leopy.Likelihood(obs, p_true='lognorm', p_cond='norm')
            p_all2[i] = like.p(loc_true, scale_true, shape_true=shape_true,
                               R_true=R, pool=self.pool)

        assert np.all(np.isclose(p_all.reshape(N), p_all2))


    # test integrating out variables
    # p(x_i | theta, R) = int dy_i p(y_i | x_i, theta) p(x_i | theta, R)
    # = int dy_i p(x_i, y_i | theta, R)
    def test_17(self):

        v0 = [0.5, 2.0, 1.7]
        ev0 = [0.1, 0.2, 0.3]
        v1 = [3, 4, 5.2]
        ev1 = [0.1, 0.1, 0.15]
        rv0v1 = [0.2, 0.8, -0.8]

        d = {'v0': v0, 'e_v0': ev0, 'v1': v1, 'e_v1': ev1, 'r_v0_v1': rv0v1}
        obs = leopy.Observation(d, 'test', verbosity=0)
        like = leopy.Likelihood(obs, p_true='lognorm', p_cond='norm')

        loc_true = [-0.02, 1.95]
        scale_true = [0.7, 1.9]
        shape_true = np.array([0.5, 2.03]).reshape(2, 1)
        rho = 0.0
        R = np.array([[1., rho], [rho, 1.]])

        p_x = like.p(loc_true, scale_true, shape_true=shape_true, R_true=R,
                     vars=[0], pool=self.pool)
        p_y = like.p(loc_true, scale_true, shape_true=shape_true, R_true=R,
                     vars=[1], pool=self.pool)

        import scipy.integrate
        N = 2000

        xx = np.concatenate(
            [-np.logspace(1, -5, N//5)+loc_true[0], [loc_true[0]],
             np.logspace(-5, 4, N-N//5-1) + loc_true[0]])
        yy = np.concatenate(
            [-np.logspace(1, -5, N//5)+loc_true[1], [loc_true[1]],
             np.logspace(-5, 4, N-N//5-1) + loc_true[1]])

        d_x = {'v0': np.outer(v0, np.ones(N)).flatten(),
               'e_v0': np.outer(ev0, np.ones(N)).flatten(),
               'v1': np.outer(np.ones(3), yy).flatten(),
               'e_v1': np.outer(ev1, np.ones(N)).flatten(),
               'r_v0_v1': np.outer(rv0v1, np.ones(N)).flatten()}
        obs_x = leopy.Observation(d_x, 'test', verbosity=0)
        like_x = leopy.Likelihood(obs_x, p_true='lognorm', p_cond='norm')
        res = like_x.p(loc_true, scale_true, shape_true=shape_true, R_true=R,
                       pool=self.pool)
        res = res.reshape(3, N)
        p_x_2 = scipy.integrate.trapz(res, yy)

        assert np.all(np.isclose(p_x.reshape(3), p_x_2, atol=1e-4))

        d_y = {'v0': np.outer(np.ones(3), xx).flatten(),
               'e_v0': np.outer(ev0, np.ones(N)).flatten(),
               'v1': np.outer(v1, np.ones(N)).flatten(),
               'e_v1': np.outer(ev1, np.ones(N)).flatten(),
               'r_v0_v1': np.outer(rv0v1, np.ones(N)).flatten()}
        obs_y = leopy.Observation(d_y, 'test', verbosity=0)
        like_y = leopy.Likelihood(obs_y, p_true='lognorm', p_cond='norm')
        res = like_y.p(loc_true, scale_true, shape_true=shape_true, R_true=R,
                       pool=self.pool).reshape(3, N)
        p_y_2 = scipy.integrate.trapz(res, xx)

        assert np.all(np.isclose(p_y.reshape(3), p_y_2, atol=1e-4))

    # test product decomposition of uncorrelated, joint pdf
    def test_18(self):
        v0 = [0.5, 2.0, 1.7, 1.1]
        ev0 = [0.1, 0.2, 0.3, 0.15]
        v1 = [3, 4, 5.2, 2.2]
        ev1 = [0.1, 0.1, 0.15, 0.12]
        v2 = [-2, 3, 1.7, 1.]
        ev2 = [0.2, 0.1, 0.05, 0.15]

        d = {'v0': v0, 'e_v0': ev0, 'v1': v1, 'e_v1': ev1, 'v2': v2,
             'e_v2': ev2}
        obs = leopy.Observation(d, 'test', verbosity=0)
        like = leopy.Likelihood(obs, p_true=['lognorm','gamma','norm'],
                                p_cond='norm')
        loc_true = [-0.02, 1.95, 1]
        scale_true = [0.7, 1.9, 2.5]
        shape_true = [[0.5], [2.03], []]
        p_0 = like.p(loc_true, scale_true, shape_true=shape_true, vars=[0],
                     pool=self.pool)
        p_01 = like.p(loc_true, scale_true, shape_true=shape_true, vars=[0, 1],
                      pool=self.pool)
        p_02 = like.p(loc_true, scale_true, shape_true=shape_true, vars=[0, 2],
                      pool=self.pool)
        p_012 = like.p(loc_true, scale_true, shape_true=shape_true,
                       pool=self.pool)

        assert np.all(np.isclose(p_01/p_0 * p_02/p_0, p_012/p_0))
