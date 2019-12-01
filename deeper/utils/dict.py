
from itertools import chain
from itertools import islice


class NestedDict(dict):
    """   
    https://stackoverflow.com/questions/41472726/multi-nested-dictionary-from-tuples-in-python/41473151                                                                    
    Nested dictionary of arbitrary depth with autovivification.               

    Allows data access via extended slice notation.                           
    """
    def __getitem__(self, keys):
        # Let's assume *keys* is a list or tuple.                             
        if not isinstance(keys, str):
            try:
                node = self
                for key in keys:
                    node = dict.__getitem__(node, key)
                return node
            except TypeError:
            # *keys* is not a list or tuple.                              
                pass
        try:
            return dict.__getitem__(self, keys)
        except KeyError:
            raise KeyError(keys)
    def __setitem__(self, keys, value):
        # Let's assume *keys* is a list or tuple.                             
        if not isinstance(keys, str):
            try:
                node = self
                for key in keys[:-1]:
                    try:
                        node = dict.__getitem__(node, key)
                    except KeyError:
                        node[key] = type(self)()
                        node = node[key]
                return dict.__setitem__(node, keys[-1], value)
            except TypeError:
                # *keys* is not a list or tuple.                              
                pass
        dict.__setitem__(self, keys, value)


def iter_leafs(d, keys=[], types=[]):
    """Iterate over a nested dict and surface leaf values with paths
    for reconstruction"""
    for key, val in d.items():
        if isinstance(val, dict):
            yield from iter_leafs(val, keys + [key], types + [type(val)])
        else:
            yield keys + [key], types + [type(val)], val

