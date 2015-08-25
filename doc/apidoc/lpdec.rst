The lpdec package: Structure and General Utilities
==================================================

The :mod:`lpdec` package is organized as follows:

* The main package directory contains several utility classes and functions used across the package
* The :mod:`lpdec.codes` subpackage contains implementations of various classes of codes
* The :mod:`lpdec.decoders` subpackage contains decoder implementations
* The :mod:`lpdec.database` packages is responsible for storing and retrieving simulation results
  in relational databases.
* The :mod:`lpdec.cli` package contains the command-line tools that provide e.g. database browsing or
  code analysis.

.. automodule:: lpdec

Channels
--------

.. automodule:: lpdec.channels

Matrix Input and Output
-----------------------
.. automodule:: lpdec.matrices

Handing of Binary (GF(2)) Matrices
----------------------------------
.. automodule:: lpdec.gfqla
   :members:

Various Utilities
-----------------
.. automodule:: lpdec.utils