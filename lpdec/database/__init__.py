# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This package manages the database containing code definitions, simulation results, and other.

Before any of the functions are used, the :func:`init` method must be called in order to initialize
the database connection."""

from __future__ import division, unicode_literals

from os.path import exists, join, expanduser
import os, sys
import numbers
import atexit
import sqlalchemy.types as types
import sqlalchemy as sqla
from dateutil import tz
from lpdec.codes import BinaryLinearBlockCode
from lpdec.decoders import Decoder


CONF_DIR = expanduser(join('~', '.config', 'lpdec'))
DB_LIST_FILE = join(CONF_DIR, 'databases')
ONLY_DUMMY = False
_knownDBs = None
if sys.version_info.major == 2:
    input = raw_input


class UTCDateTime(types.TypeDecorator):
    """Subclasses :class:`sqla.DateTime` in order to add UTC timezone info to all objects.

    When used as bind parameter, any supplied datetime object will be converted to UTC (if a
    timezone exists). Retrieved datetimes will always have the UTC timezone set.
    """

    impl = types.DateTime

    def process_bind_param(self, value, dialect):
        if value.tzinfo and value.tzinfo != tz.tzutc():
            value = value.astimezone(tz.tzutc())
        return value.replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        return value.replace(tzinfo=tz.tzutc())


class DatabaseException(Exception):
    """Exception that indicates a problem in the database connection or state.
    """
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


codesTable = decodersTable = engine = metadata = None
initialized = False


def init(database=None, testMode=False):
    """Initialize the database package and connect to `database`, which is a connection string.

    Examples:
    - sqlite:///myresults.sqlite
    - mysql://username:password@localhost/port

    If `database` is `None`, an interactive console dialogs asks the user to input a string. The
    database string may as well be a single number; in that case, the connection will be made to
    the entry with that position in the list of known databases.
    """
    global codesTable, decodersTable, engine, metadata, initialized
    if initialized:
        return
    if not testMode:
        known = knownDatabases()
        if database is None:
            print('please choose a database for storing results')
            if len(known) > 0:
                print('\nThese databases have been used before:')
                for i, db in enumerate(known):
                    print('{0:>3d} {1}'.format(i, db))
                ans = input('Enter a number or a new database connection string: ')
                try:
                    database = known[int(ans)]
                except (KeyError, ValueError):
                    database = ans
            else:
                database = input('Please enter a database connection string: ')
        elif database.isdigit():
            database = known[int(database)]
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
                               sqla.Column('json', sqla.Text))
    metadata.create_all(engine)
    metadata.bind = engine
    initialized = True


def teardown():
    """Deallocate all database resources. This is mainly intended for testing."""
    global engine, metadata, decodersTable, codesTable, initialized
    if engine:
        engine.dispose()
    engine = metadata = decodersTable = codesTable = None
    initialized = False


def checkCode(code, insert=True):
    """Tests if `code` is contained in the database. If there is a code with the same name that does
    not match the given code, a :class:`DatabaseException` is raised.

    The code will be inserted into the database if `insert` is `True` and the code not yet
    contained.

    :returns: The code's primary ID (if it exists or was inserted by this method), otherwise
        `None`.
    :rtype: int
    """
    return _checkCodeOrDecoder('code', code, insert=insert)


def checkDecoder(decoder, insert=True):
    """Tests if `decoder` is contained in the database. If there is a decoder with the same name
    that does not match the given code, a :class:`DatabaseException` is raised.

    The decoder will be inserted into the database if `insert` is `True` and the decoder not yet
    contained.

    :returns: The decoder's primary ID (if it exists or was inserted by this method), otherwise
        `None`.
    :rtype: int
    """
    return _checkCodeOrDecoder('decoder', decoder, insert=insert)


def _checkCodeOrDecoder(which, obj, insert=True):
    assert which in ('code', 'decoder')
    table = codesTable if which == 'code' else decodersTable
    s = sqla.select([table], table.c.name == obj.name)
    row = engine.execute(s).fetchone()
    if row is not None:
        if row[table.c.json] != obj.toJSON():
            raise DatabaseException('A {} named "{}" with different JSON representation '
                                    .format(which, obj.name) +
                                    'already exists in the database:\n'
                                    '{}\n\n{}'.format(row[table.c.json], obj.toJSON()))
        return row[table.c.id]
    elif insert:
        args = dict(name=obj.name, classname=type(obj).__name__, json=obj.toJSON())
        if which == 'code':
            args['blocklength'] = obj.blocklength
            args['infolength'] = obj.infolength
        result = table.insert().execute(**args)
        return result.inserted_primary_key[0]


class DummyDecoder(Decoder):

    def __init__(self, code, name):
        Decoder.__init__(self, code, name)

    def solve(self, *args, **kwargs):
        raise RuntimeError('cannot solve using DummyDecoder')


def get(what, identifier, code=None):
    """Retrieve a code or decoder from the database. `what` is one of ("code", "decoder") and
    specifies what to retrieve. The `identifier` can be either the primary key or the name or
    the instance object of the code or decoder.
    :returns: The code or decoder object corresponding to the input identifier.
    :raises: :class:`DatabaseException` if it was not found.
    """
    if what == 'code':
        cls = BinaryLinearBlockCode
        table = codesTable
    elif what == 'decoder':
        cls = Decoder
        table = decodersTable
    else:
        raise ValueError('"what" has to be one of ("code", "decoder")')
    if isinstance(identifier, numbers.Integral):
        condition = table.c.id == identifier
    elif isinstance(identifier, cls):
        condition = table.c.name == identifier.name
    else:
        condition = table.c.name == identifier
    s = sqla.select([table], condition)
    row = engine.execute(s).fetchone()
    if row is None:
        raise DatabaseException('{} "{}" not found'.format(what, identifier))
    elif isinstance(identifier, cls):
        return identifier
    elif what == 'code':
        return cls.fromJSON(row[table.c.json])

    else:
        if ONLY_DUMMY:
            return DummyDecoder(code=code, name=row[table.c.name])
        try:
            return cls.fromJSON(row[table.c.json], code=code)
        except Exception as e:
            # CPLEX, Gurobi etc. might not be available
            print('Warning: creating dummy decoder "{}":\n{}'.format(row[table.c.name], e))
            return DummyDecoder(code=code, name=row[table.c.name])


def createCode(name, cls, **kwargs):
    """Convenience function that returns a code from database, if it exists, and otherwise creates
    it with the given parameters."""
    import lpdec.imports
    s = sqla.select([codesTable.c.name], codesTable.c.name == name)
    ans = engine.execute(s).fetchone()
    if ans is None:
        return cls(**kwargs)
    else:
        return get('code', name)


def names(what='codes'):
    """Return the names of all codes or decoders, depending on the parameter `what` which is one of
    ('decoders', 'codes').
    """
    table = codesTable if what == 'codes' else decodersTable

    return [row[0] for row in engine.execute(sqla.select([table.c.name]))]