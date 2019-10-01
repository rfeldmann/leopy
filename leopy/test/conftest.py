"""Pytest fixtures for LEO-Py.

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
import pytest

@pytest.fixture(scope='class')
def pool(request):
    multimode = 'None'
    # multimode = 'Serial'
    # multimode = 'Multi'
    # multimode = 'MPI'

    # setup code
    pool = None
    if multimode == 'Serial':
        from schwimmbad import SerialPool
        pool = SerialPool()
    elif multimode == 'Multi':
        from schwimmbad import MultiPool
        pool = MultiPool()
    elif multimode == 'MPI':
        from schwimmbad import MPIPool
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            import sys
            sys.exit(0)

    # inject class variables
    request.cls.pool = pool
    yield

    # tear down
    if multimode == 'Multi' or multimode == 'MPI':
        pool.close()
