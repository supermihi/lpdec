# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals
import sys
import itertools
from collections import OrderedDict
import jinja2
from dateutil import tz
from lpdec.database import simulation as dbsim
from lpdec import utils

if sys.version_info.major == 2:
    input = raw_input


def initParser(parser):
    """Populate parser options for the "browse" action."""
    parser.add_argument('-i', '--identifier')
    parser.add_argument('-c', '--code')
    parser.add_argument('-a', '--all', action='store_true',
                        help='select all simulations for given identifier/code')
    parser.add_argument('-t', '--template', choices=tuple(TEMPLATES.keys()), default='cli',
                        help='template for the output format of simulation results')
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--min-errors', type=int, default=0,
                        help='minimum number of errors for plotting a value')
    parser.add_argument('--min-points', type=int, default=0,
                        help='minimum number of SNR values in a range for plotting')
    parser.add_argument('--time', action='store_true', help='plot decoding time instead of error rate')
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
    if len(stats) == 0:
        return ''
    maxLen = max(len(s) for s in stats)
    for stat, val in stats.items():
        import numbers
        if isinstance(val, numbers.Number):
            val = "{:<6f} avg".format(val / point.samples)
        ret += ("      {:"+str(maxLen) + "s} = {}\n").format(stat, val)
    return ret


def browse(args):
    """Interactive command-line browsing through simulation results."""
    import lpdec.imports  # ensure all decoder and code classes are loaded
    lpdec.database.init(args.database)
    lpdec.database.ONLY_DUMMY = True
    dbsim.init()

    # query simulation identifier, if not provided as CLI option
    if args.identifier:
        identifiers = [args.identifier]
    else:
        identifiers = dbsim.existingIdentifiers()
        print('Available identifiers:')
        for i, ident in enumerate(identifiers):
            print('{0:2d}: {1}'.format(i, ident))
        nums = utils.splitRanges(input('Select number(s): '))
        identifiers = [identifiers[num] for num in nums]

    # query code, if not provided as CLI option
    if args.code:
        selectedCodes = [args.code]
    else:
        codes = [row[0] for row in dbsim.search('codename', identifier=identifiers)]
        print('Available codes:')
        for i, code in enumerate(codes):
            print('{:>3d}: {}'.format(i, code))
        print('{:>3s}: select all'.format('A'))
        ans = input('Select number(s): ')
        if ans in 'aA':
            nums = list(range(len(codes)))
        else:
            nums = utils.splitRanges(ans)
        selectedCodes = [codes[num] for num in nums]

    # query simulation run
    runs = dbsim.simulations(code=selectedCodes, identifier=identifiers)
    runs.sort(key=lambda run: run.date_start)
    if not args.all:
        print('These simulation runs match your selection:')
        print('{:>3s}  {:30s} {:40s} {:16s} {:9s} {:8s} {} {}\n'
              .format("i", "code", "decoder", "identifier", "snr range", 'wordseed', "date",
                      "version"))
        for i, run in enumerate(runs):
            print('{:>3d}: {:30s} {:40s} {:16s} {:9s} {:<8d} {} {}'
                  .format(i, str(run.code), str(run.decoder), run.identifier,
                          '{}–{}'.format(run.minSNR(), run.maxSNR()),
                          run.wordSeed,
                          '{:%d.%m.%y %H:%M}/{:%d.%m.%y %H:%M}'
                          .format(run.date_start.astimezone(tz.tzlocal()),
                                  run.date_end.astimezone(tz.tzlocal())),
                          run.version))

        print('{0:>3s}: *select all*'.format('A'))
        ans = input('Select number(s): ')
        if ans in 'aA':
            nums = list(range(len(runs)))
        else:
            nums = utils.splitRanges(ans)
        runs = [ runs[i] for i in range(len(runs)) if i in nums ]
    env = jinja2.Environment(autoescape=False)
    env.filters['formatStats'] = formatStats
    template = env.from_string(TEMPLATES[args.template])
    for run in runs:
        if args.outfile:
            out = open(args.outfile.format(code=run.code.name), 'wt')
        else:
            out = sys.stdout
        out.write(template.render(sim=run, verbose=args.verbose) + '\n')
        if args.outfile:
            out.close()
    if args.plot:
        plotSimulation(args, runs)


def plotSimulation(args, runs):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    if not args.time:
        plt.yscale('log')
    codes = set(run.code.name for run in runs)
    decoders = set(run.decoder.name for run in runs)
    markers = itertools.cycle('os^p8>D+|')
    if len(decoders) > 1 and len(codes) > 1:
        fmt = '{c}/{d}'
    elif len(decoders) > 1:
        fmt = '{d}'
    else:
        fmt = '{c}'
    for run in runs:
        if not args.time and len(run) < args.min_points:
            continue
        if args.time:
            vals = [(point.snr, point.avgTime) for point in run]
        else:
            vals = [(point.snr, point.frameErrorRate)
                    for point in run if point.errors > args.min_errors]
        if len(vals) == 0:
            continue
        x, y = zip(*vals)
        label = fmt.format(c=run.code, d=run.decoder)
        plt.plot(x, y, marker=next(markers), label=label)
    plt.legend(loc='upper right')
    plt.grid(True, which='minor')
    plt.grid(True, which='major')
    plt.xlabel("SNR$_b$ [dB]")
    plt.ylabel('cpu time' if args.time else 'FER')
    plt.title("Decoding results")
    plt.show()



TEMPLATES = dict()
TEMPLATES['hp'] = \
"""{{sim.code}} ({{sim.code.blocklength}},{{sim.code.infolength}}) Code: ML Simulation Results
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
  samples    errors     FER      av.cpu   wordseed channel
{% for point in sim %}  {{"{:<10d} {:<10d} {:<8.2e} {:<.6f} {:8d} {:10s}".format(point.samples,
point.errors, point.frameErrorRate, point.cputime/point.samples, point.wordSeed,
point.channel.__repr__())}}\
{% if verbose %}
  {{point|formatStats}}
{% endif %}
{% endfor %}"""

TEMPLATES['pgf'] = \
"""{{sim.identifier}}: {{sim.code.name}} // {{sim.decoder.name}}:
{% for point in sim %}\
  {{"({:<4.2f}, {:<8.2e})   %{:d}".format(point.snr, point.frameErrorRate, point.errors)}}
{% endfor %}"""