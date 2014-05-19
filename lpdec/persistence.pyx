# -*- coding: utf-8 -*-
# cython: embedsignature=True
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import json

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
        from lpdec import subclasses
        classes = subclasses(JSONDecodable)
        if 'className' in dct:
            if dct['className'] not in classes:
                raise RuntimeError('Class "{}" not loaded'.format(dct['className']))
            clsName = dct['className']
            del dct['className']
            dct.update(kwargs)
            return classes[clsName](**dct)
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
            from lpdec import subclasses
            classes = subclasses(cls)
            if classname not in classes:
                raise ValueError('Subclass {} of {} not found'.format(classname, type(cls)))
            return classes[classname](**decoded)
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