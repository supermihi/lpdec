# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import print_function, division, unicode_literals
import argparse
from collections import OrderedDict
import lpdec.database
import lpdec.database.simulation as dbsim


def script():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--database', metavar='DB', help='database connection string')
    subparsers = parser.add_subparsers(title='Commands')
    parser_browse = subparsers.add_parser('browse', help='browse and plot results')
    parser_browse.add_argument('-i', '--identifier')
    parser_browse.add_argument('-t', '--template', choices=('cli', 'hp'), default='cli',
                               help='template for the output format of simulation results')
    parser_browse.set_defaults(func=browse)
    parser_codes = subparsers.add_parser('codes', help='code toolkit')
    parser_codes.set_defaults(func=codes)

    args = parser.parse_args()
    args.func(args)

def browse(args):
    import lpdec.jsonloads
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
    print('These simulation runs match your selection:')
    print('{:>3s}  {:30s} {:40s} {:16s} {:10s} {}\n'
          .format("i", "code", "decoder", "identifier", "snr-range", "date"))
    for i, run in enumerate(runs):
        print('{:>3d}: {:30s} {:40s} {:16s} {:10s} {}'
              .format(i, run.code.name, run.decoder.name, run.identifier,
                      '{}-{}'.format(run.minSNR(), run.maxSNR()),
                      '{}-{}'.format(run.date_start, run.date_end)))
    print("{0:>3s}: *select all*".format("A"))
    ans = raw_input("Select number(s): ")
    if ans in 'aA':
        nums = list(range(len(runs)))
    else:
        nums = [int(n) for n in ans.split()]
    runs = [ runs[i] for i in range(len(runs)) if i in nums ]
    from . import simTemplate
    template = simTemplate.getTemplate(args.template)
    for run in runs:
        print(template.render(sim=run))


def codes(args):
    pass

if __name__ == '__main__':
    script()