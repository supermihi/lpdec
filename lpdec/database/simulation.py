# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals, print_function
import json
import sqlalchemy as sqla
import lpdec
from lpdec import database as db
from lpdec.persistence import JSONDecodable


initialized = False


def init():
    """Initialize the simulations database module. This needs to be called before any other
    function of this module can be used, but after :func:`db.init`.
    """
    global simTable, joinTable, initialized
    if initialized:
        return
    if not db.initialized:
        db.init()
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
                          sqla.Column('date_start', sqla.DateTime),
                          sqla.Column('date_end', sqla.DateTime),
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
                  machine=db.machineString(),
                  program_name='lpdec',
                  program_version=lpdec.__version__,
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
    from lpdec import simulation

    point = simulation.DataPoint(code, channel, wordSeed, decoder, identifier)
    if ans is not None:
        point.samples = point._dbSamples = ans[simTable.c.samples]
        point.errors = ans[simTable.c.errors]
        point.cputime = ans[simTable.c.cputime]
        point.date_start = ans[simTable.c.date_start]
        point.date_end = ans[simTable.c.date_end]
        point.stats = JSONDecodable.fromJSON(ans[simTable.c.stats])
        point.version = ans[simTable.c.program_version]
    return point
