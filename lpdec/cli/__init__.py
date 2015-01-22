# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import print_function, division, unicode_literals
import argparse
from lpdec.cli import code, browse


def script():
    import locale
    locale.resetlocale()
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--database', metavar='DB', help='database connection string')
    parser.add_argument('-v', '--verbose', action='store_true', help='be more verbose')
    parser.add_argument('-o', '--outfile', help='write output to given file instead of stdout')
    subparsers = parser.add_subparsers(title='Commands')
    parserBrowse = subparsers.add_parser('browse', help='browse and plot results')
    browse.initParser(parserBrowse)
    parserCodes = subparsers.add_parser('code', help='code toolkit')
    code.initParser(parserCodes)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    script()