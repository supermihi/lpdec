# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals
import sys
from collections import OrderedDict
import jinja2
from dateutil import tz
from lpdec.database import simulation as dbsim


def initParser(parser):
    """Populate parser options for the "browse" action."""
    parser.add_argument('-i', '--identifier')
    parser.add_argument('-c', '--code')
    parser.add_argument('-a', '--all', action = 'store_true',
                        help='select all simulations for given identifier/code')
    parser.add_argument('-t', '--template', choices=tuple(TEMPLATES.keys()), default='cli',
                        help='template for the output format of simulation results')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')
    parser.set_defaults(func=browse)


def formatStats(point):
    """Jinja filter for nice output of decoder statistics"""
    ret = ''
    stats = OrderedDict()
    for key, val in point.stats.items():
        if isinstance(val, dict):
            for kkey, vval in val.items():
                stats["{}.{}".format(key, kkey)] = vval
        else:
            stats[key] = val
    maxLen = max(len(s) for s in stats)
    for stat, val in stats.items():
        import numbers
        if isinstance(val, numbers.Number):
            val = "{:<6f} avg".format(val / point.samples)
        ret += ("      {:"+str(maxLen) + "s} = {}\n").format(stat, val)
    return ret


def browse(args):
    import lpdec.imports
    lpdec.database.init(args.database)
    dbsim.init()
    identifiers = dbsim.existingIdentifiers()
    if args.identifier:
        identifiers = [ args.identifier ]
    else:
        print('Available identifiers:')
        for i, ident in enumerate(identifiers):
            print('{0:2d}: {1}'.format(i, ident))
        ans = raw_input('Select number(s): ')
        nums = [ int(s) for s in ans.split() ]
        identifiers = [identifiers[num] for num in nums]
    if args.code:
        selectedCodes = [args.code]
    else:
        codes = [row[0] for row in dbsim.search('codename', identifier=identifiers)]
        print('Available codes:')
        for i, code in enumerate(codes):
            print('{:>3d}: {}'.format(i, code))
        print('{:>3s}: select all'.format('A'))
        ans = raw_input('Select number(s): ')
        if ans in 'aA':
            nums = list(range(len(codes)))
        else:
            nums = [int(n) for n in ans.split()]
        selectedCodes = [codes[num] for num in nums]
    runs = dbsim.simulations(code=selectedCodes, identifier=identifiers)
    if not args.all:
        print('These simulation runs match your selection:')
        print('{:>3s}  {:30s} {:40s} {:16s} {:10s} {}\n'
              .format("i", "code", "decoder", "identifier", "snr-range", "date"))
        for i, run in enumerate(runs):
            print('{:>3d}: {:30s} {:40s} {:16s} {:10s} {}'
                  .format(i, run.code.name, run.decoder.name, run.identifier,
                          '{}–{}'.format(run.minSNR(), run.maxSNR()),
                          '{:%c} – {:%c}'.format(run.date_start.astimezone(tz.tzlocal()),
                                                 run.date_end.astimezone(tz.tzlocal()))))
        print('{0:>3s}: *select all*'.format('A'))
        ans = raw_input('Select number(s): ')
        if ans in 'aA':
            nums = list(range(len(runs)))
        else:
            nums = [int(n) for n in ans.split()]
        runs = [ runs[i] for i in range(len(runs)) if i in nums ]
    env = jinja2.Environment(autoescape=False)
    env.filters['formatStats'] = formatStats
    template = env.from_string(TEMPLATES[args.template])
    if args.outfile:
        out = open(args.outfile, 'wt')
    else:
        out = sys.stdout
    for run in runs:
        out.write(template.render(sim=run, verbose=args.verbose) + '\n')
    if args.outfile:
        out.close()


TEMPLATES = dict()
TEMPLATES['hp'] = \
"""{{sim.code.name}} ({{sim.code.blocklength}},{{sim.code.infolength}}) Code: ML Simulation Results
Channel: {{sim.channelClass.__name__}}
Modulation: BPSK

Eb/N0 (dB)   FER     error frames   total frames
{% for point in sim %}  {{"{:4.2f},{:12.3e},{:7d},{:16d}".format(point.snr, point.frameErrorRate,
point.errors,
  point.samples)}}
{% endfor %}

Stefan Scholl and Michael Helmling
Microelectronic Systems Design Research Group and Optimization Research Group
University of Kaiserslautern, Germany, {{"{:%Y}".format(sim.date_end)}}

This file was downloaded from: http://www.uni-kl.de/channel-codes/"""

TEMPLATES['cli'] = \
"""{{sim.identifier}}: {{sim.code.name}} // {{sim.decoder.name}}:
  snr   samples    errors     FER      av.cpu
{% for point in sim %}  {{"{:<4.2f}  {:<10d} {:<10d} {:<8.2e} {:<.6f}".format(point.snr,
point.samples, point.errors, point.frameErrorRate, point.cputime/point.samples)}}\
{% if verbose %}
  {{point|formatStats}}
{% endif %}
{% endfor %}"""

TEMPLATES['verb'] = \
"""{{sim.identifier}}: {{sim.code.name}} // {{sim.decoder.name}}:
  channel          samples    errors     FER      av.cpu
{% for point in sim %}  {{"{:16s}  {:<10d} {:<10d} {:<8.2e} {:<.6f}".format(point.channel,
point.samples, point.errors, point.frameErrorRate, point.cputime/point.samples)}}\
{% if verbose %}
  {{point|formatStats}}
{% endif %}
{% endfor %}"""