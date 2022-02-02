class Scope:
    def __init__(self, var_scope):
        """A helper class to feed variable scope to kera layers. Keras layers
        are broken within tensforflow 2.0 and do not propograte name-scope
        correctly.

        Usage: Set keras name to scope.self.vname
        """
        self.var_scope = var_scope
        pass

    def v_name(self, x):
        return str(self.var_scope) + "/" + str(x)
