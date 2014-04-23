# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division

import json
from collections import OrderedDict

cimport numpy as np
import numpy as np

__version__ = '2014.1'


def subclasses(base):
    """Return all subclasses of `base` as dictionary mapping class names to class
    objects.
    """
    found = set([base])
    toCheck = list(base.__subclasses__())
    for cls in toCheck:
        found.add(cls)
        toCheck.extend(cls.__subclasses__())
    return dict( (cls.__name__, cls) for cls in found )


class JSONEncoder(json.JSONEncoder):
    """Custom JSON that encodes subclasses of :class:`JSONDecodable` by
    :func:`JSONDecodable.params` and the name of the class in the `className` attribute.
    """
    def default(self, obj):
        if issubclass(type(obj), JSONDecodable):
            dct = obj.params()
            dct['className'] = type(obj).__name__
            return dct
        return json.JSONEncoder.default(self, obj)


def jsonObjectHook(dct):
    """Specialized JSON object decoder can create :class:`JSONDecodable` objects."""
    classes = subclasses(JSONDecodable)
    if 'className' in dct and dct['className'] in classes:
        clsName = dct['className']
        del dct['className']
        return classes[clsName](**dct)
    return dct


cdef class JSONDecodable(object):
    """Base class for objects that can be serialized using JSON.
    """

    def params(self):
        """Return a JSON encodable dictionary of parameters which, when passed to the constructor,
        yield the same object."""
        raise NotImplementedError()

    @classmethod
    def fromParams(cls, paramString, classname=None):
        """Create object of this class or a subclass from the JSON string `paramString`.
        """
        decoded = json.loads(paramString, object_hook=jsonObjectHook)
        if not isinstance(decoded, JSONDecodable):
            if classname is None:
                raise ValueError('Classname must be given if paramString does not contain one.')
            classes = subclasses(cls)
            if classname not in classes:
                raise ValueError('Subclass {} of {} not found'.format(classname, type(cls)))
            return classes[classname](**decoded)
        return decoded

    def __repr__(self):
        paramString = ', '.join('{0}={1}'.format(k, repr(v)) for k, v in self.params().items())
        return '{c}({p})'.format(c=self.__class__.__name__, p=paramString)
