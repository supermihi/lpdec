# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""Module supporting the creating of random linear codes."""
import numpy as np
from lpdec.codes.nonbinary import NonbinaryLinearBlockCode
from lpdec.codes import BinaryLinearBlockCode

def makeRandomCode(n, m, density, q=2, seed=1337):
    """
    Creates a random linear code over the prime field :math:`\mathbb F_q`. The code is created by randomly choosing a
    parity-check matrix.

    Args
    ----
    n : int
        Block length
    m : int
        Number of rows in the parity-check matrix. Design information length is k=n-m but can be larger if rows happen
        to be linearly dependent.
    density : float
        Probabilty of non-zero entry in the parity-check matrix.
    q : int
        Arity of the code. Defaults to 2 for binary codes.
    seed : int
        Random seed.
    """
    state = np.random.RandomState(seed)

    H = state.randint(1, q, (m, n))
    mask = state.random_sample((m, n)) >= density
    H[mask] = 0
    name = 'Random({}, {})Code[density={},q={},seed={}]'.format(n, n-m, density, q, seed)
    if q == 2:
        return BinaryLinearBlockCode(name=name, parityCheckMatrix=H)
    else:
        return NonbinaryLinearBlockCode(name=name, parityCheckMatrix=H, q=q)
