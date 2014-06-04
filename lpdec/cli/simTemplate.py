# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals
from collections import OrderedDict
import jinja2


homepageTemplate = \
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

consoleTemplate = \
"""{{sim.identifier}}: {{sim.code.name}} // {{sim.decoder.name}}:
  snr   samples    errors     FER      av.cpu
{% for point in sim %}  {{"{:<4.2f}  {:<10d} {:<10d} {:<8.2e} {:<.6f}".format(point.snr,
point.samples, point.errors, point.frameErrorRate, point.cputime/point.samples)}}\
{% if verbose %}
  {{point|formatStats}}
{% endif %}
{% endfor %}"""

verboseTemplate = \
"""{{sim.identifier}}: {{sim.code.name}} // {{sim.decoder.name}}:
  channel          samples    errors     FER      av.cpu
{% for point in sim %}  {{"{:16s}  {:<10d} {:<10d} {:<8.2e} {:<.6f}".format(point.channel,
point.samples, point.errors, point.frameErrorRate, point.cputime/point.samples)}}\
{% if verbose %}
  {{point|formatStats}}
{% endif %}
{% endfor %}"""


def formatStats(point):
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


def getTemplate(template):
    env = jinja2.Environment(autoescape=False)
    env.filters['formatStats'] = formatStats
    if template == 'cli':
        templateString = consoleTemplate
    elif template == 'hp':
        templateString = homepageTemplate
    elif template == 'verb':
        templateString = verboseTemplate
    template = env.from_string(templateString)
    return template