lpdec: library for LP decoding and related things
=================================================
Overview
--------
*lpdec* is a scientific library dedicated to the *decoding* part of coding theory, inparticular,
decoding based on methods of mathematical optimization, such as linear programming (LP) decoding.

Requirements
------------
The library is written in [Python](www.python.org). It was mainly developed using Python version 
2.7, but was recently ported to Python3, although that port is not stable yet. Note that there is
 no CPLEX interface for Python3, so you can not use the CPLEX solvers with that version of Python.
 
To compile the library, you need [Cython](www.cython.org). Runtime requirements are the 
Python packages [numpy](www.numpy.org), [dateutil](https://labix.org/python-dateutil), 
[sqlalchemy](www.sqlalchemy.org), and [jinja2](http://jinja.pocoo.org) (only for displaying 
simulation 
results).

Some of the decoder implementations require additional software, namely 
[GLPK](http://www.gnu.org/software/glpk/) (with C headers),
[IBM CPLEX](http://www.ibm.com/software/commerce/optimization/cplex-optimizer/), and
 [Gurobi](http://gurobi.com). We use the alternative gurobi python API called
  [gurobimh](https://github.com/supermihi/gurobimh).

Installation
------------

Download the package and type:

    python setup.py install --user
    
If you do not have GLPK installed, use:

    python setup.py install --no-glpk --user

In both commands, replace ``python`` by an appropriate call to your Python interpreter. You will 
probably run into less problems when using ``python2``.

Documentation
-------------
API documentation is provided [online](https://pythonhosted.org/lpdec).

You can also try to generate the API doc with [Sphinx](www.sphinx-doc.org). To build the 
documentation,
run the following command from within *lpdec*'s main directory:

    sphinx-build2 doc doc-html
    
This will generate HTML API documentation inside the *doc-html* folder.