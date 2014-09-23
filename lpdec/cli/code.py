# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals, print_function
from lpdec.codes import BinaryLinearBlockCode
from lpdec import database as db, matrices


def initParser(parser):
    parser.add_argument('-f', '--file', help='input file for the code')
    parser.set_defaults(func=codeCommand)
    sub = parser.add_subparsers(title='actions', dest='action')
    printParser = sub.add_parser('print', help='print a code')
    printParser.add_argument('--alist', action='store_true', help='output in alist format')
    printParser.add_argument('-w', '--width', help='output width for matrix elements', default='2')


def printCode(args):
    """Print the code specified by the CLI options."""
    if args.verbose:
        print('({},{}) code (rate={})'.format(args.code.blocklength, args.code.infolength,
                                              args.code.rate))
    format = 'alist' if args.alist else 'plain'
    fname = args.outfile
    ans = matrices.formatMatrix(args.code.parityCheckMatrix, format, int(args.width), fname)
    if args.outfile is None:
        print(ans)


def codeCommand(args):
    if args.file:
        args.code = BinaryLinearBlockCode(parityCheckMatrix=args.file)
    else:
        db.init(args.database)
        codes = db.names('codes')
        print('Available codes:')
        for i, code in enumerate(codes):
            print('{:>3d}: {}'.format(i, code))
        ans = raw_input('Select number: ')
        import lpdec.imports  # ensures that all classes are loaded for JSON decoding
        args.code = db.get('code', codes[int(ans.strip())])
    if args.action == 'print':
        printCode(args)

