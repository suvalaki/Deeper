import numpy as np


def linear_cooling(k, start, end, alpha ):
    """Linear interpolation between two points over alpha time periods"""
    shifted_base = end - start
    return start + (shifted_base) * min(k, alpha) / alpha 


def exponential_multiplicative_cooling(i, start, end, alpha):
    """Proposed by Kirkpatrick, Gelatt and Vecchi (1983), and used as reference 
    in the comparison among the different cooling criteria. The temperature decrease 
    is made multiplying the initial temperature T0 by a factor that decreases 
    exponentially with respect to temperature cycle k:"""
    shifted_base = start - end
    return end + (shifted_base) *((alpha)**i)
   
def logarithmical_multiplicative_cooling(k, start, end, alpha):
    """Based on the asymptotical convergence condition of simulated annealing (Aarts, 
    E.H.L. & Korst, J., 1989), but incorporating a factor a of cooling speeding-up 
    that makes possible its use in practice. The temperature decrease is made multiplying 
    the initial temperature T0 by a factor that decreases in inverse proportion to the 
    natural logarithm of temperature cycle k:"""
    cooling = 1 / (1+alpha*np.log(1+k))
    shifted_base = start - end
    return shifted_base*cooling + end 

def  linear_multiplicative_cooling(k, start, end, alpha):
    """The temperature decrease is made multiplying the initial temperature T0 by a factor 
    that decreases in inverse proportion to the temperature cycle k"""
    cooling = 1 / (1+alpha*k)
    shifted_base = start - end
    return shifted_base*cooling + end 

def quadratic_multiplicative_cooling(k, start, end, alpha):
    """The temperature decrease is made multiplying the initial temperature T0 by a factor 
    that decreases in inverse proportion to the square of temperature cycle k"""
    cooling = 1 / (1+alpha*k**2)
    shifted_base = start - end
    return shifted_base*cooling + end 


class CyclicCoolingRegime():

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
