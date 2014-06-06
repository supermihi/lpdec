lpdec: library for LP decoding and related things
=================================================
Overview
--------
*lpdec* is a scientific library dedicated to the *decoding* part of coding theory, inparticular,
decoding based on methods of mathematical optimization, such as linear programming (LP) decoding.

Requirements
------------
The library is written in [Python](www.python.org), version 2.7. It requires
[Cython](www.cython.org). Optionally, also 
[GLPK](http://www.gnu.org/software/glpk/) (with C headers) and
[IBM CPLEX](http://www.ibm.com/software/commerce/optimization/cplex-optimizer/) are necessary. 

Installation
------------

Download the package and type:

    python2 setup.py install --user
    
If you do not have GLPK installed, use:

    python2 setup.py install --no-glpk --user

In both commands, replace ``python2`` by an appropriate call to a Python 2.7 interpreter.

Documentation
-------------
API documentation can be generated with [Sphinx](www.sphinx-doc.org). To build the documentation,
run the following command from within *lpdec*'s main directory:

    sphinx-build2 doc doc-html
    
This will generate HTML API documentation inside the *doc-html* folder.