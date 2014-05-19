# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import jinja2


jinjaString = \
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


def getTemplate():
    env = jinja2.Environment(autoescape=False)
    template = env.from_string(jinjaString)
    return template