import numpy as np
import tensorflow as tf


def linear_cooling(k, start, end, alpha):
    """Linear interpolation between two points over alpha time periods
    Parameters
    alpha:  interpolation steps
    """
    shifted_base = end - start
    return start + (shifted_base) * tf.math.minimum(k, alpha) / alpha


def exponential_multiplicative_cooling(i, start, end, alpha):
    """Proposed by Kirkpatrick, Gelatt and Vecchi (1983), and used as reference
    in the comparison among the different cooling criteria. The temperature decrease
    is made multiplying the initial temperature T0 by a factor that decreases
    exponentially with respect to temperature cycle k:"""
    shifted_base = start - end
    return end + (shifted_base) * ((alpha) ** i)


def logarithmical_multiplicative_cooling(k, start, end, alpha):
    """Based on the asymptotical convergence condition of simulated annealing (Aarts,
    E.H.L. & Korst, J., 1989), but incorporating a factor a of cooling speeding-up
    that makes possible its use in practice. The temperature decrease is made multiplying
    the initial temperature T0 by a factor that decreases in inverse proportion to the
    natural logarithm of temperature cycle k:"""
    cooling = 1 / (1 + alpha * np.log(1 + k))
    shifted_base = start - end
    return shifted_base * cooling + end


def linear_multiplicative_cooling(k, start, end, alpha):
    """The temperature decrease is made multiplying the initial temperature T0 by a factor
    that decreases in inverse proportion to the temperature cycle k"""
    cooling = 1 / (1 + alpha * k)
    shifted_base = start - end
    return shifted_base * cooling + end


def quadratic_multiplicative_cooling(k, start, end, alpha):
    """The temperature decrease is made multiplying the initial temperature T0 by a factor
    that decreases in inverse proportion to the square of temperature cycle k"""
    cooling = 1 / (1 + alpha * k ** 2)
    shifted_base = start - end
    return shifted_base * cooling + end


class CallableTempCls:
    def __init__(self, method, start, end, alpha):
        """Object to hold the cooling method for this

        method: name of the cooling regime to implement

        """
        self.method = method
        self.start = start
        self.end = end
        self.alpha = alpha
        self.iter = 0

    def __call__(self, x):
        self.iter = x
        value = self.method(self.iter, self.start, self.end, self.alpha)
        return value


class ConstantRegime:
    def __init__(self, const=1.0):
        self.const = const

    def increment_iter(self):
        pass

    def __call__(self):
        return self.const


class InfiniteCoolingRegime:
    def __init__(self, method, start, end, alpha):
        """Object to hold the cooling method for this

        method: name of the cooling regime to implement

        """
        self.method = method
        self.start = start
        self.end = end
        self.alpha = alpha
        self.iter = 0

    def increment_iter(self):
        self.iter += 1

    def __call__(self):
        value = self.method(self.iter, self.start, self.end, self.alpha)
        self.iter += 1
        return value


class CyclicCoolingRegime:
    def __init__(self, method, start, end, alpha, cycle):
        """Object to hold the cooling method for this

        method: name of the cooling regime to implement

        """
        self.method = method
        self.start = start
        self.end = end
        self.alpha = alpha
        self.cycle = cycle
        self.iter = 0

    def increment_iter(self):
        self.iter += 1

    def __call__(self):
        if self.iter > self.cycle:
            self.iter = 0
        value = self.method(self.iter, self.start, self.end, self.alpha)
        self.iter += 1
        return value


def get_method(identifier):
    """A dict of the form {class_name, class_args...}"""
    # class_args = {k:v for k,v in identifier.items() if k is not "class_name"}
    mappings = {
        "linear": linear_cooling,
        "exponential_multiplicative": exponential_multiplicative_cooling,
        "logarithmical_multiplicative": logarithmical_multiplicative_cooling,
        "linear_multiplicative": linear_multiplicative_cooling,
        "quadratic_multiplicative_cooling": quadratic_multiplicative_cooling,
    }
    # return (mappings[identifier], class_args)
    return mappings[identifier]


def get_regime(identifier: dict):
    """A dict of the form {class_name, class_args}"""
    class_args = {k: v for k, v in identifier.items() if k != "class_name"}
    mappings = {
        "static": CallableTempCls,
        "constant": ConstantRegime,
        "infinite": InfiniteCoolingRegime,
        "cyclic": CyclicCoolingRegime,
    }

    if "method" in class_args:
        # Get the appropriate method
        class_args["method"] = get_method(class_args["method"])

    regime = mappings[identifier["class_name"]](**class_args)
    return regime
