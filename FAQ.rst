Frequently Asked Questions
**************************

Installation
------------

Q: The installation via ``python setup.py install`` fails with the error
``RuntimeError: Python version >= 3.5 required``.

A: LEO-Py requires python version 3.5 or higher (type ``python --version`` on
the command line to see which version you are using). On some systems, you call
python version 3 via ``python3``. If this is the case for your system, you
could try ``python3 setup.py install`` instead. There are a number of
convenient ways to download and install python version 3 if it is not already
set up on your system, such as,

- Anaconda <https://www.anaconda.com/distribution> (Windows / macOS / Linux)
- Homebrew <https://brew.sh> (macOS / Linux)

Q: ``python setup.py install`` fails to install one or more of the packages
LEO-Py depends on. What can I do?

A: This can happen if you are using an older version of setuptools. You may
want to try to install the dependencies directly via pip, i.e.,
type ``pip install scipy numpy pandas sphinx pytest`` in the command line. If
this has been successful, try ``python setup.py install`` again.

Miscellaneous
-------------

Q: Where can I find the code documentation?

A: Please run ``python setup.py build_html`` from the package directory. You
may then open the newly created file ./build/sphinx/html/index.html with your
browser to read the documentation.

Q: Can I participate in the further development of LEO-Py?

A: Yes, absolutely. Your contribution is very welcome. You can find the source
repository on Github <https://github.com/rfeldmann/leopy>.
