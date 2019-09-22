from setuptools import setup, find_packages

# read version (single place of truth)
name = 'leopy'
version = {}
with open("leopy/__about__.py") as fp:
    exec(fp.read(), version)
version = version['__version__']

# read the contents of README file
with open('README.rst', encoding='utf-8') as f:
        long_description = f.read()

cmdclass = {}
try:
    from sphinx.setup_command import BuildDoc
    cmdclass['build_html'] = BuildDoc
except ImportError:
    pass

setup(name=name,
      version=version,
      description='Likelihood Estimation for Observational data with Python',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
      ],
      url='http://github.com/rfeldmann/leopy',
      author='Robert Feldmann',
      author_email='RobertFeldmann@gmx.de',
      maintainer='Robert Feldmann',
      maintainer_email='RobertFeldmann@gmx.de',
      license='GNU GPLv3',
      keywords='statistics likelihood probability',
      packages=find_packages(),
      data_files=[('paper', ['paper/xGASS.csv'])],
      install_requires=[
        'pandas>=0.24.0',
        'scipy>=1.3.0',
        'numpy',
        'sphinx',
        'sphinx_rtd_theme'
      ],
      extras_require={
        'MPI':  ['schwimmbad', 'mpi4py'],
        'multiprocessing': ['schwimmbad']
      },
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False,
      cmdclass=cmdclass,
      command_options={
          'build_html': {
              'project': ('setup.py', name),
              'version': ('setup.py', version),
              'source_dir': ('setup.py', 'docs'),
          }
      })