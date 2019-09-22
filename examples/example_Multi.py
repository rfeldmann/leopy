"""Example of data with lognormal distribution - this example demonstrates the
    importance of properly accounting for measurement uncertainty.

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
import leopy

from schwimmbad import MultiPool
pool = MultiPool()

d = {'v0': [1, 2], 'e_v0': [0.1, 0.2],
             'v1': [3, 4], 'e_v1': [0.1, 0.1]}
obs = leopy.Observation(d, 'testdata')
    
like = leopy.Likelihood(obs, 
    p_true='gamma', p_cond='norm')
    
print(like.p([0.5, 0.7], [1, 2], 
             shape_true=[1.4, 2], pool=pool))

pool.close()
