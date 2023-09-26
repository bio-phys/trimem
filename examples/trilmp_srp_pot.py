from __future__ import print_function
import numpy as np

class LAMMPSPairPotential(object):
    def __init__(self):
        self.pmap=dict()
        self.units='lj'
    def map_coeff(self,name,ltype):
        self.pmap[ltype]=name
    def check_units(self,units):
        if (units != self.units):
           raise Exception("Conflicting units: %s vs. %s" % (self.units,units))


class SRPTrimem(LAMMPSPairPotential):
    def __init__(self):
        super(SRPTrimem,self).__init__()
        # set coeffs: kappa_r, cutoff, r (power)
        #              4*eps*sig**12,  4*eps*sig**6
        self.units = 'lj'
        self.coeff = {'C'  : {'C'  : (1.0,1000.0,2)  } }

    def compute_energy(self, rsq, itype, jtype):
        coeff = self.coeff[self.pmap[itype]][self.pmap[jtype]]

        srp1 = coeff[0]
        srp2 = coeff[1]
        srp3 = coeff[2]
        r = np.sqrt(rsq)
        rl=r-srp1

        e=0.0
        e+=np.exp(r/rl)
        e/=r**srp3
        e*=srp2

        return e

    def compute_force(self, rsq, itype, jtype):
        coeff = self.coeff[self.pmap[itype]][self.pmap[jtype]]
        srp1 = coeff[0]
        srp2 = coeff[1]
        srp3 = coeff[2]

        r = np.sqrt(rsq)
        f=0.0

        rp = r ** (srp3 + 1)
        rl=r-srp1
        f=srp1/(rl*rl)+srp3/r
        f/=rp
        f*=np.exp(r/rl)
        f*=srp2

        return f    
