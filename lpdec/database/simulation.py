# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals, print_function
import json
from collections import OrderedDict
import sqlalchemy as sqla
from sqlalchemy.sql import func
import lpdec
from lpdec import database as db
from lpdec import simulation
from lpdec.persistence import JSONDecodable


def init():
    """Initialize the simulations database module. This needs to be called before any other
    function of this module can be used, but after :func:`db.init`.
    """
    global simTable, joinTable
    simTable = sqla.Table('simulations', db.metadata,
                          sqla.Column('id', sqla.Integer, primary_key=True),
                          sqla.Column('identifier', sqla.String(128)),
                          sqla.Column('code', None, sqla.ForeignKey('codes.id')),
                          sqla.Column('decoder', None, sqla.ForeignKey('decoders.id')),
                          sqla.Column('channel_class', sqla.String(16)),
                          sqla.Column('snr', sqla.Float),
                          sqla.Column('channel_json', sqla.Text),
                          sqla.Column('total_frames', sqla.Integer),
                          sqla.Column('error_frames', sqla.Integer),
                          sqla.Column('cputime', sqla.Float),
                          sqla.Column('stats', sqla.Text),
                          sqla.Column('date_start', sqla.DateTime),
                          sqla.Column('date_end', sqla.DateTime),
                          sqla.Column('machine', sqla.Text),
                          sqla.Column('program_name', sqla.String(64)),
                          sqla.Column('program_version', sqla.String(64)))
    db.metadata.create_all(db.engine)
    joinTable = simTable.join(db.codesTable).join(db.decodersTable)
    module_initialized = True


def existingIdentifiers():
    """Returns a list of all identifiers for which simulation results exist in the database."""
    s = sqla.select([simTable.c.identifier]).distinct()
    results = db.engine.execute(s).fetchall()
    return [r[0] for r in results]


def addDataPoint(point):
    """Add (or update) a :class:`simulation.DataPoint` to the database."""
    codeId = db.checkCode(point.code, insert=True)
    decoderId = db.checkDecoder(point.decoder, insert=True)
    channelJSON = point.channel.toJSON()
    channelClass = type(point.channel).__name__
    whereClause = (
        (simTable.c.identifier == point.identifier) &
        (simTable.c.code == codeId) &
        (simTable.c.decoder == decoderId) &
        (simTable.c.channel_json == channelJSON)
    )
    values = dict(code=codeId,
                  decoder=decoderId,
                  identifier=point.identifier,
                  channel_class=channelClass,
                  snr=point.channel.snr,
                  channel_json=channelJSON,
                  samples=point.samples,
                  errors=point.errors,
                  cputime=point.cputime,
                  date_start=point.date_start,
                  date_end=point.date_end,
                  machine=db.machineString(),
                  program_name='lpdec',
                  program_version=lpdec.__version__,
                  stats=json.dumps(point.statistics, sort_keys=True))
    result = db.engine.execute(sqla.select([simTable.c.id], whereClause)).fetchall()
    if len(result) > 0:
        assert len(result) == 1
        simId = result[0][0]
        update = simTable.update().where(simTable.c.id == simId).values(**values)
        db.engine.execute(update)
    else:
        insert = simTable.insert().values(**values)
        db.engine.execute(insert)
    point._dbSamples = point.samples
    point._dbCputime = point.cputime


def dataPoint(code, channel, decoder, identifier):
    """Return a :class:`simulation.DataPoint` object for the given parameters.

    If such one exists in the database, it is initialized with the data (samples, errors etc.) from
    there. Otherwise an empty point is created.
    """
    s = sqla.select([joinTable], (simTable.c.identifier == identifier) &
                                 (db.codesTable.c.name == db.codeName(code)) &
                                 (db.decodersTable.c.name == db.decoderName(decoder)) &
                                 (simTable.c.channel_json == channel.toJSON()))
    ans = db.engine.execute(s).fetchone()
    point = simulation.DataPoint(code, channel, decoder, identifier)
    if ans is not None:
        point.samples = point._dbSamples = ans[simTable.c.samples]
        point.errors = ans[simTable.c.errors]
        point.cputime = ans[simTable.c.cputime]
        point.date_start = ans[simTable.c.date_start]
        point.date_end = ans[simTable.c.date_end]
        point.stats = JSONDecodable.fromJSON(ans[simTable.c.stats])
        point.version = ans[simTable.c.program_version]
    return point
