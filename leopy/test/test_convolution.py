"""Simple test cases for the leopy.convolution module.

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
import scipy.stats
import scipy.integrate

import pytest

import leopy

@pytest.fixture
def conv_norm_norm_ana():
    return leopy.Convolution(scipy.stats.norm, scipy.stats.norm)

@pytest.fixture
def conv_norm_norm():
    c = leopy.Convolution(scipy.stats.norm, scipy.stats.norm, eps=1e-8)
    c.name = 'composite'
    return c

@pytest.fixture
def conv_norm_lognorm():
    return leopy.Convolution(scipy.stats.norm, scipy.stats.lognorm,
                             atol=1e-6, rtol=1e-6, eps=1e-8)

@pytest.fixture
def conv_lognorm_gamma():
    return leopy.Convolution(scipy.stats.lognorm, scipy.stats.gamma,
                             atol=1e-9, rtol=1e-9, eps=1e-8)

# test for norm-norm convolution
def test_pdf_1(conv_norm_norm_ana):
    def pdf(x):
        return 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2)
    x = np.array([-1, 0.5, 10.])
    scale_xy = 2
    loc_y = 0.25
    scale_y = 1.5
    scale_conv = np.sqrt(scale_xy**2 + scale_y**2)
    assert np.all(np.isclose(
        conv_norm_norm_ana.pdf(x, scale_xy, loc_y, scale_y),
        pdf((x - loc_y)/scale_conv)/scale_conv))

def test_pdf_2(conv_norm_norm_ana):
    def pdf(x):
        return 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2)
    x = np.array([-1, 0.5, 10.])
    scale_xy = 1e-4
    loc_y = 0.25
    scale_y = 2.
    scale_conv = np.sqrt(scale_xy**2 + scale_y**2)
    assert np.all(np.isclose(
        conv_norm_norm_ana.pdf(x, scale_xy, loc_y, scale_y),
        pdf((x - loc_y)/scale_conv)/scale_conv))

def test_pdf_3(conv_norm_norm):
    def pdf(x):
        return 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2)
    x = np.array([-1, 0.5, 10.])
    scale_xy = 2.
    loc_y = 0.25
    scale_y = 1.5
    scale_conv = np.sqrt(scale_xy**2 + scale_y**2)
    assert np.all(np.isclose(
        conv_norm_norm.pdf(x, scale_xy, loc_y, scale_y),
        pdf((x - loc_y)/scale_conv)/scale_conv))

def test_pdf_4(conv_norm_norm):
    def pdf(x):
        return 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2)
    x = np.array([-1, 0.5, 10.])
    scale_xy = 1e-4
    loc_y = 0.25
    scale_y = 2.
    scale_conv = np.sqrt(scale_xy**2 + scale_y**2)
    assert np.all(np.isclose(
        conv_norm_norm.pdf(x, scale_xy, loc_y, scale_y),
        pdf((x - loc_y)/scale_conv)/scale_conv))

# test for norm-lognorm convolution
def test_pdf_5(conv_norm_lognorm):
    x1 = -np.flipud(np.logspace(-5, np.log10(10.2), 300))
    x2 = np.logspace(-5, 5, 300)
    x = np.concatenate([x1, x2])
    integral = scipy.integrate.simps(
        conv_norm_lognorm.pdf(x, 1.7, 1., -0.2, 2.), x)
    assert np.isclose(integral, 1.0)

# test for lognorm_gamma convolution
def test_pdf_6(conv_lognorm_gamma):
    x = -1 + np.logspace(-4, 5, 200)
    integral = scipy.integrate.simps(
        conv_lognorm_gamma.pdf(x, 1.5, 1.7, 1.3, -1, 1.2), x)
    assert np.isclose(integral, 1.0, rtol=1e-4, atol=1e-4)

def test_cdf_1(conv_norm_norm):
    x = np.array([-1, 0.5, 10.])
    scale_xy = 2.
    loc_y = 0.25
    scale_y = 10.
    scale_conv = np.sqrt(scale_xy**2 + scale_y**2)
    conv_norm_norm.name = 'composite'
    assert np.all(np.isclose(
        conv_norm_norm.cdf(x, scale_xy, loc_y, scale_y),
        scipy.stats.norm.cdf((x - loc_y)/scale_conv)))

def test_cdf_2(conv_lognorm_gamma):
    x = np.concatenate([np.linspace(-8, 10, 300),np.logspace(1.01, 4, 300)])
    parameters = (1.5, 1.7, 0.1, -1, 2.0)
    limits = [0, 1, 10, 100, 1e4, np.max(x)]
    y = conv_lognorm_gamma.pdf(x, *parameters)
    cdf_from_pdf = []
    cdf_directly = []
    for limit in limits:
        nearest_limit = x[np.argmin(np.abs(x - limit))]
        sel = x <= nearest_limit
        cdf_from_pdf.append(scipy.integrate.simps(y[sel], x[sel]))
        cdf_directly.append(conv_lognorm_gamma.cdf(nearest_limit, *parameters))
    assert np.all(np.isclose(cdf_from_pdf, cdf_directly, rtol=1e-4, atol=1e-4))

def test_cdf_3(conv_lognorm_gamma):
    x = np.concatenate([np.linspace(-8, 10, 200),np.logspace(1.01, 4, 100)])
    parameters = (1.5, 1.7, 1.3, -1, 1.7)
    limits = [0, 1, 10, 100, 1e4, np.max(x)]
    y = conv_lognorm_gamma.pdf(x, *parameters)
    cdf_from_pdf = []
    cdf_directly = []
    for limit in limits:
        nearest_limit = x[np.argmin(np.abs(x - limit))]
        sel = x <= nearest_limit
        cdf_from_pdf.append(scipy.integrate.simps(y[sel], x[sel]))
        cdf_directly.append(conv_lognorm_gamma.cdf(nearest_limit, *parameters))
    assert np.all(np.isclose(cdf_from_pdf, cdf_directly, rtol=1e-4, atol=1e-4))

def test_cdf_4(conv_lognorm_gamma):
    x = np.concatenate([np.linspace(-8, 10, 200),np.logspace(1.01, 4, 100)])
    parameters = (1.5, 1.7, 2.0, -1, 0.3)
    limits = [0, 1, 10, 100, 1e4, np.max(x)]
    y = conv_lognorm_gamma.pdf(x, *parameters)
    cdf_from_pdf = []
    cdf_directly = []
    for limit in limits:
        nearest_limit = x[np.argmin(np.abs(x - limit))]
        sel = x <= nearest_limit
        cdf_from_pdf.append(scipy.integrate.simps(y[sel], x[sel]))
        cdf_directly.append(conv_lognorm_gamma.cdf(nearest_limit, *parameters))
    assert np.all(np.isclose(cdf_from_pdf, cdf_directly, rtol=1e-4, atol=1e-4))
    assert np.isclose(cdf_directly[-1], 1.0)
