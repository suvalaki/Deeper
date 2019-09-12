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
