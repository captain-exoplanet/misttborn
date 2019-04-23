#implementation of a modified version of the quasi-periodic GP kernel used by Haywood et al. 2014 for use with MISTTBORN and celerite.
#based upon code from Daniel Foreman-Mackey from https://celerite.readthedocs.io/en/stable/python/kernel/

import autograd.numpy as anp
from celerite import terms

class CustomTerm(terms.Term):
    parameter_names = ("log_a", "log_b", "log_c", "log_P")

    def get_real_coefficients(self, params):
        log_a, log_b, log_c, log_P = params
        b = anp.exp(log_b)
        return (anp.exp(log_a) * (1.0 + b), anp.exp(log_c),)

    def get_complex_coefficients(self, params):
        log_a, log_b, log_c, log_P = params
        b = anp.exp(log_b)
        return (anp.exp(log_a), 0.0,anp.exp(log_c), 2*anp.pi*anp.exp(-log_P),)
