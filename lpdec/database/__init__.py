# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This package manages the database containing code definitions, simulation results, and other.

Before any of the functions are used, the :func:`init` method must be called in order to initialize
the database connection."""

from __future__ import division

from os.path import exists, join, expanduser
import os
import numbers
import atexit
from lpdec.codes import BinaryLinearBlockCode

import sqlalchemy as sqla


CONF_DIR = expanduser(join('~', '.config', 'lpdec'))
DB_LIST_FILE = join(CONF_DIR, 'databases')
_knownDBs = None


class DatabaseException(Exception):
    pass


def knownDatabases():
    """Return a list of known database connection strings.

    The returned list may be modified in order to add/remove database strings. It is
    automatically stored at program exit.
    """

    global _knownDBs
    if _knownDBs is None:
        # register exit handler on first call
        atexit.register(saveDatabases)
        _knownDBs = []
        if not exists(CONF_DIR):
            os.makedirs(CONF_DIR)
        if exists(DB_LIST_FILE):
            with open(DB_LIST_FILE, 'rt') as listFile:
                for line in listFile:
                    if line.strip() != "":
                        _knownDBs.append(line.strip())
    return _knownDBs


def saveDatabases():
    """Store the databases list to the config file."""
    with open(DB_LIST_FILE, 'wt') as listFile:
        listFile.write('\n'.join(_knownDBs))


def init(database=None, testMode=False):
    """Initialize the database package and connect to `database`, which is a connection string.

    Examples:
    - sqlite:///myresults.sqlite
    - mysql://username:password@localhost/port

    If `database` is `None`, an interactive console dialogs asks the user to input a string.
    """
    global codesTable, decodersTable, engine, metadata
    if not testMode:
        known = knownDatabases()
        if database is None:
            print('please choose a database for storing results')
            if len(known) > 0:
                print('\nThese databases have been used before:')
                for i, db in enumerate(known):
                    print('{0:>3d} {1}'.format(i, db))
                ans = raw_input('Enter a number or a new database connection string: ')
                try:
                    database = known[int(ans)]
                except (KeyError, ValueError):
                    database = ans
            else:
                database = raw_input('Please enter a database connection string: ')
    engine = sqla.create_engine(database, pool_recycle=3600)
    if not testMode and database not in known:
        known.append(database)
    metadata = sqla.MetaData()
    codesTable = sqla.Table('codes', metadata,
                            sqla.Column('id', sqla.Integer, primary_key=True),
                            sqla.Column('name', sqla.String(128), unique=True),
                            sqla.Column('classname', sqla.String(64)),
                            sqla.Column('json', sqla.Text),
                            sqla.Column('blocklength', sqla.Integer),
                            sqla.Column('infolength', sqla.Integer))
    decodersTable = sqla.Table('decoders', metadata,
                               sqla.Column('id', sqla.Integer, primary_key=True),
                               sqla.Column('name', sqla.String(128), unique=True),
                               sqla.Column('classname', sqla.String(64)),
                               sqla.Column('params', sqla.Text))
    metadata.create_all(engine)
    metadata.bind = engine


def teardown():
    global engine, metadata, decodersTable, codesTable
    if engine:
        engine.dispose()
    engine = metadata = decodersTable = codesTable = None


def checkCode(code, insert=True):
    """Tests if `code` is contained in the database. If there is a code with the same name that does
    not match the given code, a :class:`DatabaseException` is raised.

    The code will be inserted into the database if `insert` is `True` and the code not yet
    contained.

    :returns: The code's primary ID (if it exists or was inserted by this method), otherwise
    `None`.
    """
    s = sqla.select([codesTable], codesTable.c.name == code.name)
    row = engine.execute(s).fetchone()
    if row is not None:
        if row[codesTable.c.json] != code.toJSON():
            raise DatabaseException('A code named "{}" with different JSON representation '
                                    'already exists in the database')
        return row[codesTable.c.id]
    elif insert:
        result = codesTable.insert().execute(name=code.name,
                                             classname=type(code).__name__,
                                             json=code.toJSON(),
                                             blocklength=code.blocklength,
                                             infolength=code.infolength)
        return result.inserted_primary_key[0]


def getCode(code):
    """Retrieve a code from the database. The parameter `code` can be either the primary key or
    the name of the code.
    :returns: The :class:`BinaryLinearBlockCode` object corresponding to the input.
    :raises: :class:`DatabaseException` if the code was not found.
    """
    if isinstance(code, numbers.Integral):
        condition = codesTable.c.id == code
    elif isinstance(code, basestring):
        condition = codesTable.c.name == code
    elif isinstance(code, BinaryLinearBlockCode):
        condition = codesTable.c.name == code.name
    else:
        raise ValueError('{} is not a valid code identifier'.format(code))
    s = sqla.select([codesTable], condition)
    codeRow = engine.execute(s).fetchone()
    if codeRow is None:
        raise DatabaseException('Code "{}" not found'.format(code))
    if isinstance(code, BinaryLinearBlockCode):
        return code
    return BinaryLinearBlockCode.fromJSON(codeRow[codesTable.c.json])
