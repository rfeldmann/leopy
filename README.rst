********
LEO-Py
********
.. inclusion-marker-do-not-remove

Data with uncertain, missing, censored, and correlated values are commonplace
in many research fields including astronomy. Unfortunately, such data are often
treated in an ad hoc way potentially resulting in inconsistent parameter
estimates. Furthermore, in a realistic setting, the variables of interest or
their errors may have non-normal distributions which complicates the modeling.
LEO-Py uses a novel technique to compute the likelihood function for such data
sets. This approach employs Gaussian copulas to decouple the correlation
structure of variables and their marginal distributions resulting in a flexible
method to compute likelihood functions of data in the presence of measurement
uncertainty, censoring, and missing data.

If you use any version of this code, please properly reference the code paper:
*Feldmann, R. (2019) "LEO-Py: Estimating likelihoods for correlated, censored,
and uncertain data with given marginal distributions", Astronomy & Computing,
29, 100331, DOI:10.1016/j.ascom.2019.100331*

Copyright 2019 University of Zurich, Robert Feldmann

----

LEO-Py requires a working python3.5 installation or later to run.

Before installing LEO-Py, you may want to set up a virtual environment
``python -mvenv /path/to/new/virtual/environment`` and activate it via
``source /path/to/new/virtual/environment/bin/activate``.

To install LEO-Py from a repository:

* Download the source code from <https://github.com/rfeldmann/leopy>.
* Go to the package directory.
* Run ``python setup.py install``.

To install LEO-Py via PyPI:

* Run ``pip install leopy-stat``.
* Note, example scripts and the documentation are not installed in this case.

To test the installation:

* Run ``python setup.py test`` from the package directory (if installed from
  source).
* Go to the 'site-packages' directory and run ``python -m pytest leopy``
  (if installed via PyPI).

To access the code documentation (if installed from source):

* Run ``python setup.py build_html`` from the package directory.
* Open ./build/sphinx/html/index.html to read the documentation.

Using leopy is very simple and consists of 4 steps:

* Load the module (``import leopy``).
* Create an observational data set (leopy.Observation).
* Create a likelihood instance (leopy.Likelihood).
* Call function p() of the likelihood instance.

For instance, a minimal example is::

    import pandas as pd
    from leopy import Observation, Likelihood
    d = {'v0': [1, 2], 'e_v0': [0.1, 0.2],
         'v1': [3, 4], 'e_v1': [0.1, 0.1]}
    obs = Observation(pd.DataFrame(d), 'testdata')

    l = Likelihood(obs, p_true='lognorm', p_cond='norm')
    l.p([0.5, 0.7], [1, 2], shape_true=[[1.4], [2.]])
    # array([[0.04415447],
    #        [0.01089338]])

Further examples are provided in the 'paper' and 'examples' sub-directories
(if installed from source).

----

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
