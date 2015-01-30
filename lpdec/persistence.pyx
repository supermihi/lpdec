# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import json

def classByName(name):
    from lpdec import subclasses
    classes = subclasses(JSONDecodable)
    return classes[name]

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


def makeObjectHook(**kwargs):
    def jsonObjectHook(dct):
        """Specialized JSON object decoder can create :class:`JSONDecodable` objects."""
        if 'className' in dct:
            clsName = dct['className']
            del dct['className']
            dct.update(kwargs)
            try:
                return classByName(clsName)(**dct)
            except KeyError:
                raise RuntimeError('Class "{}" not loaded'.format(clsName))
        return dct
    return jsonObjectHook

cdef class JSONDecodable(object):
    """Base class for objects that can be serialized using JSON.
    """

    def params(self):
        """Return a JSON encodable dictionary of parameters which, when passed to the constructor,
        yield the same object."""
        raise NotImplementedError()

    def toJSON(self):
        """Returns a JSON string representing this object."""
        return json.dumps(self, cls=JSONEncoder)

    @classmethod
    def fromJSON(cls, paramString, classname=None, **kwargs):
        """Create object of this class or a subclass from the JSON string `paramString`.
        """
        decoded = json.loads(paramString, object_hook=makeObjectHook(**kwargs))
        if not isinstance(decoded, JSONDecodable):
            if classname is None:
                raise ValueError('Classname must be given if paramString does not contain one.')
            try:
                return classByName(classname)(**decoded)
            except KeyError:
                raise ValueError('Subclass {} of {} not found'.format(classname, type(cls)))
        return decoded

    def __repr__(self):
        paramString = ', '.join('{0}={1}'.format(k, repr(v)) for k, v in self.params().items())
        return '{c}({p})'.format(c=self.__class__.__name__, p=paramString)

    def __richcmp__(self, other, int op):
        if op == 2: # '=='
            return type(self) == type(other) and self.params() == other.params()
        elif op == 3: # '!='
            return type(self) != type(other) or self.params() != other.params()
        raise TypeError()

    def __hash__(self):
        return object.__hash__(self)