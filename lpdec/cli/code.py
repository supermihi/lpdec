# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals, print_function
import sys
import numpy as np
from lpdec.imports import *
from lpdec import database as db, matrices


if sys.version_info.major == 2:
    input = raw_input


def initParser(parser):
    parser.add_argument('-f', '--file', help='input file for the code')
    parser.add_argument('-e', '--eval', help='Python command for creating the code')
    parser.set_defaults(func=codeCommand)
    sub = parser.add_subparsers(title='actions', dest='action')
    printParser = sub.add_parser('print', help='print a code')
    printParser.add_argument('--alist', action='store_true', help='output in alist format')
    printParser.add_argument('-w', '--width', type=int, help='output width for matrix elements', default=2)
    compareParser = sub.add_parser('compare', help='compare with another code')
    compareParser.add_argument('other', help='input file for second code')


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


def compareCode(args):
    """Compare two codes by means of their parity-check matrix. Different matrices result in a
    diff-like output."""
    other = BinaryLinearBlockCode(parityCheckMatrix=args.other)
    if np.all(args.code.parityCheckMatrix == other.parityCheckMatrix):
        print('codes have the same parity-check matrix')
    else:
        print('codes do not have the same parity-check matrix')
        import difflib
        d = difflib.Differ()
        result = d.compare(matrices.formatMatrix(args.code.parityCheckMatrix),
                           matrices.formatMatrix(other.parityCheckMatrix))
        print('\n'.join(result))


def codeCommand(args):
    if args.file:
        args.code = BinaryLinearBlockCode(parityCheckMatrix=args.file)
    elif args.eval:
        args.code = eval(args.eval)
    else:
        db.init(args.database)
        codes = db.names('codes')
        print('Available codes:')
        for i, code in enumerate(codes):
            print('{:>3d}: {}'.format(i, code))
        ans = input('Select number: ')
        args.code = db.get('code', codes[int(ans.strip())])
    if args.action == 'print':
        printCode(args)
    elif args.action == 'compare':
        compareCode(args)

