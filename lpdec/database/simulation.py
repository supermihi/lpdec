# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals, print_function
import json
from collections import OrderedDict
import sqlalchemy as sqla
from sqlalchemy.sql import expression
import lpdec
from lpdec.persistence import JSONDecodable
from lpdec import simulation, utils, database as db


initialized = False
simTable = joinTable = None


def init():
    """Initialize the simulations database module. This needs to be called before any other
    function of this module can be used, but after :func:`db.init`.
    """
    global simTable, joinTable, initialized
    if initialized:
        return
    if not db.initialized:
        raise RuntimeError('database.init() needs to be called before simulation.init()')
    simTable = sqla.Table('simulations', db.metadata,
                          sqla.Column('id', sqla.Integer, primary_key=True),
                          sqla.Column('identifier', sqla.String(128)),
                          sqla.Column('code', None, sqla.ForeignKey('codes.id')),
                          sqla.Column('decoder', None, sqla.ForeignKey('decoders.id')),
                          sqla.Column('channel_class', sqla.String(16)),
                          sqla.Column('snr', sqla.Float),
                          sqla.Column('channel_json', sqla.Text),
                          sqla.Column('wordSeed', sqla.Integer),
                          sqla.Column('samples', sqla.Integer),
                          sqla.Column('errors', sqla.Integer),
                          sqla.Column('cputime', sqla.Float),
                          sqla.Column('stats', sqla.Text),
                          sqla.Column('date_start', db.UTCDateTime),
                          sqla.Column('date_end', db.UTCDateTime),
                          sqla.Column('machine', sqla.Text),
                          sqla.Column('program_name', sqla.String(64)),
                          sqla.Column('program_version', sqla.String(64)))
    db.metadata.create_all(db.engine)
    joinTable = simTable.join(db.codesTable).join(db.decodersTable)
    initialized = True


def teardown():
    global simTable, joinTable, initialized
    simTable = joinTable = None
    initialized = False


def existingIdentifiers():
    """Returns a list of all identifiers for which simulation results exist in the database."""
    s = sqla.select([simTable.c.identifier]).distinct()
    results = db.engine.execute(s).fetchall()
    return [r[0] for r in results]


def addDataPoint(point):
    """Add (or update) a data point to the database.
    :param simulation.DataPoint point: DataPoint instance.
    ."""
    point.checkResume()
    codeId = db.checkCode(point.code, insert=True)
    decoderId = db.checkDecoder(point.decoder, insert=True)
    channelJSON = point.channel.toJSON()
    channelClass = type(point.channel).__name__
    whereClause = (
        (simTable.c.identifier == point.identifier) &
        (simTable.c.code == codeId) &
        (simTable.c.decoder == decoderId) &
        (simTable.c.channel_json == channelJSON) &
        (simTable.c.wordSeed == point.wordSeed)
    )
    if utils.machineString() not in point.machine:
        point.machine = '{}/{}'.format(point.machine, utils.machineString())
    values = dict(code=codeId,
                  decoder=decoderId,
                  identifier=point.identifier,
                  channel_class=channelClass,
                  snr=point.channel.snr,
                  channel_json=channelJSON,
                  wordSeed=point.wordSeed,
                  samples=point.samples,
                  errors=point.errors,
                  cputime=point.cputime,
                  date_start=point.date_start,
                  date_end=point.date_end,
                  machine=point.machine,
                  program_name='lpdec',
                  program_version=lpdec.exactVersion(),
                  stats=json.dumps(point.stats, sort_keys=True))
    result = db.engine.execute(sqla.select([simTable.c.id], whereClause)).fetchall()
    if len(result) > 0:
        assert len(result) == 1
        simId = result[0][0]
        update = simTable.update().where(simTable.c.id == simId).values(**values)
        db.engine.execute(update)
    else:
        insert = simTable.insert().values(**values)
        db.engine.execute(insert)


def dataPoint(code, channel, wordSeed, decoder, identifier):
    """Return a :class:`simulation.DataPoint` object for the given parameters.

    If such one exists in the database, it is initialized with the data (samples, errors etc.) from
    there. Otherwise an empty point is created.
    """
    s = sqla.select([joinTable], (simTable.c.identifier == identifier) &
                                 (db.codesTable.c.name == code.name) &
                                 (db.decodersTable.c.name == decoder.name) &
                                 (simTable.c.channel_json == channel.toJSON()) &
                                 (simTable.c.wordSeed == wordSeed)
    )
    ans = db.engine.execute(s).fetchone()
    point = simulation.DataPoint(code, channel, wordSeed, decoder, identifier)
    if ans is not None:
        point.samples = point._dbSamples = ans[simTable.c.samples]
        point.errors = ans[simTable.c.errors]
        point.cputime = point._dbCputime = ans[simTable.c.cputime]
        point.date_start = ans[simTable.c.date_start]
        point.date_end = ans[simTable.c.date_end]
        point.stats = json.loads(ans[simTable.c.stats])
        point.version = ans[simTable.c.program_version]
    return point


def dataPointFromRow(row):
    code = db.get('code', row[db.codesTable.c.name])
    channel = JSONDecodable.fromJSON(row[simTable.c.channel_json])
    wordSeed = row[simTable.c.wordSeed]
    decoder = db.get('decoder', row[db.decodersTable.c.name], code=code)
    identifier = row[simTable.c.identifier]
    point = simulation.DataPoint(code, channel, wordSeed, decoder, identifier)
    point.samples = point._dbSamples = row[simTable.c.samples]
    point.errors = row[simTable.c.errors]
    point.cputime = row[simTable.c.cputime]
    point.date_start = row[simTable.c.date_start]
    point.date_end = row[simTable.c.date_end]
    point.stats = json.loads(row[simTable.c.stats], object_pairs_hook=OrderedDict)
    point.version = row[simTable.c.program_version]
    point.program = row[simTable.c.program_name]
    point.machine = row[simTable.c.machine]
    return point


def search(what, **conditions):
    if what == 'codename':
        columns = [db.codesTable.c.name]
    elif what == 'point':
        columns = [simTable.c.identifier, db.codesTable.c.name, db.decodersTable.c.name,
                   simTable.c.channel_json, simTable.c.wordSeed, simTable.c.samples,
                   simTable.c.errors, simTable.c.cputime, simTable.c.date_start,
                   simTable.c.date_end, simTable.c.machine, simTable.c.program_name,
                   simTable.c.program_version, simTable.c.stats]
    else:
        raise ValueError('unknown search: "{}"'.format(what))
    condition = expression.true()
    for key, val in conditions.items():
        if key == 'identifier':
            condition &= simTable.c.identifier.in_(val)
        elif key == 'code':
            condition &= db.codesTable.c.name.in_(val)
        else:
            raise ValueError()
    s = sqla.select(columns, whereclause=condition, from_obj=joinTable, distinct=True,
                    use_labels=True).order_by(db.codesTable.c.name)
    ans = db.engine.execute(s).fetchall()
    if what == 'point':
        return [dataPointFromRow(row) for row in ans]
    return db.engine.execute(s).fetchall()


def simulations(**conditions):
    points = search('point', **conditions)
    sims = {}
    for point in points:
        identifier = (point.code.name, point.decoder.name, point.channel.__class__,
                      point.identifier,
                      point.wordSeed, point.program)
        if identifier not in sims:
            sims[identifier] = simulation.Simulation()
        sims[identifier].add(point)
    return list(sims.values())
