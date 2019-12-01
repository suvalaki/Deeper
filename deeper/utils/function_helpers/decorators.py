import inspect
import functools

def inits_args(func):
    """Initializes object attributes by the initializer signature"""
    argspec = inspect.getargspec(func)
    argnames = argspec.args[1:]
    defaults = dict(zip(argnames[-len(argspec.defaults):], argspec.defaults))
    @functools.wraps(func)
    def __init__(self, *args, **kwargs):
        args_it = iter(args)
        for key in argnames:
            if key in kwargs:
                value = kwargs[key]
            else:
                try:
                    value = next(args_it)
                except StopIteration:
                    value = defaults[key]
            setattr(self, key, value)
        func(self, *args, **kwargs)
    return __init__
