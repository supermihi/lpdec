# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import unicode_literals, division

from unittest import TestCase

from lpdec import database
from lpdec.codes.classic import HammingCode


class TestCodePersistence(TestCase):

    def setUp(self):
        database.init('sqlite:///:memory:', testMode=True)
        self.code = HammingCode(4)
        database.checkCode(self.code, insert=True)

    def tearDown(self):
        database.teardown()

    def test_persistence(self):
        retrieved = database.get('code', self.code.name)
        self.assertEqual(retrieved.toJSON(), self.code.toJSON())

    def test_wrongName(self):
        code = HammingCode(3)
        code.name = self.code.name
        self.assertRaises(database.DatabaseException, database.checkCode, code)