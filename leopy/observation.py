"""Handles observational data set.

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
import itertools


class Observation:
    """Container class handling observational data."""

    def __init__(self, df, reference, variables=None, auxiliary=None,
                 censored=None, remove_censored=None, remove_missing=None,
                 drop_missing=None, verbosity=1,
                 prefix_value='v', prefix_error='e_',
                 prefix_censored='c_', prefix_lower_limit='l_',
                 prefix_upper_limit='u_', prefix_correlation='r_'):
        r"""Create instance of the container class holding observational data.

        Parameters
        ----------
        df : Pandas dataframe or dict or numpy array
            Contains the observational data.
        reference : str
            Short identifier of the data source (i.e., 'Author et al. 2017').
        variables : list of str
            Column names of variables to be extracted from dataframe.
            Variables are assigned integer values starting from 0 in the
            order they are listed in `variables` (the default is None, which
            implies variables are extracted automatically; this requires that
            they are labelled 'v0', 'v1' etc if `prefix_value` is 'v').
        auxiliary : list of str or None
            Names of additional column kept for analysis (the default is None).
        censored : list of str or None
            List of variables with censoring information (the default is None,
            which implies censoring is detected automatically).
        remove_censored : list of str or None
            Remove observation if any of the listed variables is non-detected
            (the default is None, i.e., censored observations are not removed).
        remove_missing : list of str or None
            Remove observation if any of the listed variables is NaN
            (the default is None, i.e., missing observations are not removed).
        drop_missing : {None, 'any', 'all'}
            If 'any', remove all observations with one or more missing values.
            If 'all', remove all observations with all values missing
            (the default is None, which does not remove any data).
        verbosity : int
            Level of output verbosity, 0 is silent (default 1)
        prefix_value : str
            Prefix of column names for data values; the full column name is
            `prefix_value` followed by an integer (starting with 0).
            `prefix_value` is only used if `variables` == None (default 'v').
        prefix_error : str
            Prefixes for column names containing observational uncertainties
            (default 'e\_').
        prefix_censored, prefix_lower_limit, prefix_upper_limit : str
            Prefixes for column names providing information of
            whether a data value is censored (default 'c\_'),
            the lower detection limit of censored data (default 'l\_'),
            and the upper detection limit of censored data (default 'u\_').
            Variables that contain censored data require a lower or an uppper
            detection limit. It is possible to provide both lower and upper
            detection limits for variables. However, for any variable
            exactly one of the limits must be active for any given observation.
            A lower limit is active if it is larger than -np.infty, while an
            upper limit is active if it is smaller than np.infty.
        prefix_correlation : str
            Prefixes for column names containing the correlation between the
            observational errors of two variables. The full column name is
            `prefix_correlation` + var1 + '_' + var, where var1 and var2 are
            column names of variables in the data frame. If both var1+'_'+var2
            and var2+'_'+var1 correlation columns are present, their values are
            averaged. If no column are found for var1 and var2, then the
            correlation between the errors of these variables is assumed to be
            zero. (default 'r\_').

        Returns
        -------
        Observation
            Instance of class Observation

        Notes
        -----
        Values of variables of a given observation can be one of three types:
        'regular', 'missing', or 'censored'. Values are regular (are missing)
        if data values are provided (are NaN) AND either no censoring
        information is provided or the censoring column has a value of `False`.
        Values are censored if all necessary censoring information is provided
        and the censoring column has a value of `True`. A ValueError exception
        is raised if a value is encountered that does not belong to one of
        these three types.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import scipy.stats
        >>>
        >>> N = 1000
        >>> x_obs = scipy.stats.norm.rvs(loc=0, scale=1, size=N)
        >>> x_obs_err = scipy.stats.norm.rvs(loc=0, scale=0.1, size=N)
        >>>
        >>> y_lower_limit = 0.5
        >>> y_obs = scipy.stats.norm.rvs(loc=1, scale=2, size=N)
        >>> y_obs_err = scipy.stats.norm.rvs(loc=0, scale=0.5, size=N)
        >>> y_obs_censored = (y_obs < y_lower_limit)
        >>> y_obs_limit = -np.infty*np.ones_like(y_obs)
        >>> y_obs_limit[y_obs_censored] = y_lower_limit
        >>> rho = 2*(np.random.rand(N)-0.5)
        >>>
        >>> df = pd.DataFrame(
        ...     np.array([x_obs, x_obs_err, y_obs, y_obs_err,
        ...               y_obs_censored, y_obs_limit, rho]).T,
        ...     columns=['v0', 'e_v0', 'v1', 'e_v1', 'c_v1', 'l_v1',
        ...              'r_v0_v1'])
        >>>
        >>> obs = Observation(df, 'Example')
        Reading dataset 'Example' and extracting 2 variables (['v0', 'v1'])
        1 of the variables is censored (['v1'])
        Correlations between errors of observables found
        """
        import pandas as pd
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        elif isinstance(df, np.ndarray):
            columns = []
            for i in range(df.shape[1]):
                columns.append('{}{}'.format(prefix_value, i))
            df = pd.DataFrame(df, columns=columns)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                'Observations have to be given in form of a numpy array, '
                'dictionary, or pandas data frame.')

        self.df = df
        self.reference = reference
        self.verbosity = verbosity

        if variables is None:
            # extract variable names, order s.t. .., vX, .., vY, .. => X<Y
            import re
            var_indices = []
            for el in df.columns:
                pp = re.match('^{}([0-9]+)$'.format(prefix_value), el)
                if pp:
                    var_indices.append(int(pp.group(1)))
            variables = []
            for var_index in np.sort(np.unique(var_indices)):
                variables.append('{}{}'.format(prefix_value, var_index))
        elif isinstance(variables, str):
            variables = [variables]
        else:
            old_variables = list(variables)
            variables = []
            # want to keep the given ordering
            for variable in old_variables:
                if variable in df.columns:
                    variables.append(variable)

        self.num_var = len(variables)

        if censored is None:
            import re
            censored = []
            for el in set(df.columns):
                for variable in variables:
                    if (re.match('^{}{}$'.format(
                            prefix_censored, variable), el)):
                        censored.append(variable)
            censored.sort()

        if verbosity > 0:
            s_string = ''
            if len(variables) != 1:
                s_string = 's'
            print('Reading dataset \'{}\' and extracting '
                  '{} variable{} ({})'.format(
                    reference, self.num_var, s_string, variables))
            are_string = 'is'
            if len(censored) != 1:
                are_string = 'are'
            if len(censored) > 0:
                print('{} of the variables {} censored ({})'.format(
                    len(censored), are_string, censored))

        self.variables = variables
        self.censored = censored

        col_list = []
        if auxiliary is not None:
            for aux in auxiliary:
                if aux in self.df:
                    col_list.append(aux)

        if self.num_var > 0:
            for variable in variables:
                col_list.append('{}'.format(variable))
                se = '{}{}'.format(prefix_error, variable)
                if se in self.df:
                    col_list.append(se)
                if variable in censored:
                    sc = '{}{}'.format(prefix_censored, variable)
                    if sc in self.df:
                        col_list.append(sc)
                        sl = '{}{}'.format(prefix_lower_limit, variable)
                        if sl in self.df:
                            col_list.append(sl)
                        su = '{}{}'.format(prefix_upper_limit, variable)
                        if su in self.df:
                            col_list.append(su)
                        if sl not in self.df and su not in self.df:
                            raise ValueError(
                                'variable {} is censored but lower/upper '
                                'limits are missing'.format(variable))
                    else:
                        censored.remove(variable)

        # correlations between errors of observables
        if self.num_var > 0:
            for el in set(df.columns):
                for var1 in variables:
                    for var2 in variables:
                        if (re.match('^{}{}_{}$'.format(
                                prefix_correlation, var1, var2), el)):
                            col_list.append(el)

        self.df = self.df[col_list]

        # remove censored observations (default is to keep censored)
        if self.num_var > 0 and remove_censored is not None:
            for variable in set(censored).intersection(remove_censored):
                sel = (self.df['{}{}'.format(
                    prefix_censored, variable)] == False)
                self.df = self.df.loc[sel, ]
                if verbosity > 0:
                    print('Censored observations of variable \'{}\' '
                          'dropped'.format(variable))

        # remove missing observations (by variable)
        if self.num_var > 0 and remove_missing is not None:
            for variable in remove_missing:
                sel = (np.isnan(self.df[variable]) == False)
                self.df = self.df.loc[sel, ]
                if verbosity > 0:
                    print('Missing observations of variable \'{}\' '
                          'dropped'.format(variable))

        # drop missing observations (global)
        if drop_missing:
            self.df = self.df.dropna(how=drop_missing)
            if verbosity > 0:
                print('Dropping rows with {} values missing'.format(
                    drop_missing))

        # number of observations
        self.Nobs = self.df.shape[0]

        # prepare variables & errors as numpy array
        col_list = []
        if self.num_var > 0:
            for variable in variables:
                col_list.append('{}'.format(variable))
        self.v = self.df[col_list].to_numpy()

        col_list = []
        if self.num_var > 0:
            for variable in variables:
                se = '{}{}'.format(prefix_error, variable)
                if se not in self.df:
                    self.df.insert(0, se, np.nan)
                    # self.df[se] = np.nan
                col_list.append(se)
        self.ev = self.df[col_list].to_numpy()

        if auxiliary is not None:
            for aux in auxiliary:
                if aux in self.df:
                    exec('self.{} = self.df["{}"].to_numpy()'.format(
                         aux, aux))

        # potentially censored variables
        self.cv = np.zeros_like(self.v, dtype=bool)
        self.lv = -np.infty*np.ones_like(self.v)
        self.uv = np.infty*np.ones_like(self.v)
        if self.num_var > 0:
            for i, variable in enumerate(variables):
                if variable in censored:
                    # NaN in c_v? is thus translated to c_v? = False
                    sel = ~np.isnan(
                        self.df['{}{}'.format(prefix_censored, variable)])
                    self.cv[sel, i] = self.df.loc[sel, '{}{}'.format(
                        prefix_censored, variable)]
                    sl = '{}{}'.format(prefix_lower_limit, variable)
                    if sl in self.df:
                        self.lv[:, i] = self.df[sl]
                    su = '{}{}'.format(prefix_upper_limit, variable)
                    if su in self.df:
                        self.uv[:, i] = self.df[su]

                    sel = self.cv[:, i]
                    test1 = (np.sum(
                             np.logical_and(np.isneginf(self.lv[sel, i]),
                                            np.isposinf(self.uv[sel, i]))) > 0)
                    if test1:
                        raise ValueError(
                            'Variable \'{}\' contains censored observation(s) '
                            '*without* a proper lower or upper detection '
                            'limit.'.format(variable))
                    test2 = (np.sum(
                            np.logical_and(~np.isneginf(self.lv[sel, i]),
                                           ~np.isposinf(self.uv[sel, i]))) > 0)
                    if test2:
                        raise ValueError(
                            'Variable \'{}\' contains censored observation(s) '
                            'with proper lower *and* proper upper detection '
                            'limits. This functionality is currently not '
                            'supported.'.format(variable))

        # the below is useful because a given observational data point is only
        # allowed to have either an upper or a lower limit
        self.limt = np.zeros_like(self.v)
        self.lim = np.nan * np.ones_like(self.v)
        if self.num_var > 0:
            for i, variable in enumerate(variables):
                if variable in censored:
                    sel = ~np.isneginf(self.lv[:, i])
                    self.lim[sel, i] = self.lv[sel, i]
                    self.limt[sel, i] = 1
                    sel = ~np.isposinf(self.uv[:, i])
                    self.lim[sel, i] = self.uv[sel, i]
                    self.limt[sel, i] = -1

        # basic check that all censored data have a limit given
        assert np.sum(np.isnan(self.cv)) == 0
        for i, variable in enumerate(variables):
            sel = (self.cv[:, i] & np.isnan(self.lim[:, i]))
            if np.sum(sel) > 0:
                raise ValueError(
                    'Variable \'{}\' contains censored observation(s) '
                    'but the limit is "nan"'.format(variable))

        # correlation or correlation matrices
        self.correlated_errors = False
        if self.num_var > 1:
            self.rho = np.zeros((self.Nobs, self.num_var**2))
            for i in range(self.num_var):
                self.rho[:, i*self.num_var + i] = 1  # diagonal entries
            for var1, var2 in itertools.combinations(variables, 2):
                colname1 = '{}{}_{}'.format(prefix_correlation, var1, var2)
                colname2 = '{}{}_{}'.format(prefix_correlation, var2, var1)
                if colname1 in self.df or colname2 in self.df:
                    if colname2 not in self.df:
                        aux = self.df[colname1]
                    elif colname1 not in self.df:
                        aux = self.df[colname2]
                    else:
                        aux = 0.5*(self.df[colname1] + self.df[colname2])
                    i = variables.index(var1)
                    j = variables.index(var2)
                    if np.any(aux):
                        self.rho[:, i*self.num_var + j] = aux
                        self.rho[:, j*self.num_var + i] = aux
                        self.correlated_errors = True
            self.Rc = self.rho.reshape((self.Nobs, self.num_var, self.num_var))

        if verbosity > 0:
            if self.correlated_errors:
                print('Correlations between errors of observables found')
            else:
                print('Errors of different observables are assumed to be '
                      'uncorrelated')

if __name__ == '__main__':
    import doctest
    doctest.testmod()
