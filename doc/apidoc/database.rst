====================
Database Integration
====================

The ability to persistenly store codes, decoders and computational results in a database is
deeply integrated into the :mod:`lpdec` package. To that end,
code and decoder objects implement an interface that allow them to be (de-)serialized using JSON.
The base class for that mechanism is contained in the :mod:`lpdec.persistence` module.

Persistent Object Storage
-------------------------
.. automodule:: lpdec.persistence

General Database Management
---------------------------
.. automodule:: lpdec.database

Storage of Simulation Results
-----------------------------
.. automodule:: lpdec.database.simulation