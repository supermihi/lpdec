lpdec: library for LP decoding and related things
=================================================
Overview
--------
*lpdec* is a scientific library dedicated to the *decoding* part of coding theory, in particular,
decoding based on methods of mathematical optimization, such as linear programming (LP) decoding.

Requirements
------------
The library is written in [Python](www.python.org). It was mainly developed using Python version 
2.7, but was recently ported to Python3. Note that there is no CPLEX interface for Python3, so you
can not use the CPLEX solvers with that version of Python.
 
To compile the library, you need [Cython](www.cython.org). Runtime requirements are the 
Python packages [numpy](www.numpy.org), [scipy](www.scipy.org),
[dateutil](https://labix.org/python-dateutil), [sqlalchemy](www.sqlalchemy.org), 
[sympy](http://sympy.org) and [jinja2](http://jinja.pocoo.org). The setup.py script described below
will pull these requirements automatically, but depending on your OS you might favor to install
them using your system's package manager.

Some of the decoder implementations require additional software, namely 
[GLPK](http://www.gnu.org/software/glpk/) (with C headers),
[IBM CPLEX](http://www.ibm.com/software/commerce/optimization/cplex-optimizer/), and
[Gurobi](http://gurobi.com). The first is open source, the others have free academic licenses
available.

We use the alternative gurobi python API called [gurobimh](https://github.com/supermihi/gurobimh).

Installation
------------

Download the package and type:

    python setup.py install --user
    
If you do not have GLPK installed, use:

    python setup.py install --no-glpk --user

to skip installation of GLPK-based decoders. Likewise, the switch `--no-gurobi` is available.
In both commands, replace ``python`` by an appropriate call to your Python interpreter.

Documentation
-------------
API documentation is provided [online](https://pythonhosted.org/lpdec).

You can also try to generate the API doc with [Sphinx](www.sphinx-doc.org). To build the 
documentation,
run the following command from within *lpdec*'s main directory:

    sphinx-build2 doc doc-html
    
This will generate HTML API documentation inside the *doc-html* folder.