#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  a template for a simple galaxy formation model a la Krumholz & Dekel (2012); 
#                    see also Feldmann 2013
#  used as part of the 2016 A304 "Galaxies" class
#
#   Andrey Kravtsov, May 2016
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from colossus.cosmology import cosmology

def fg_in(Mh,z):
    cosmo = cosmology.setCosmology('WMAP9')
    Mc = 6.e9*np.exp(-0.63*z)
    if Mh > Mc:
        fg = 1.0
    else:
        fg = 0.0
    print fg
    return fg

def R_loss(dt):
    """
    fraction of mass formed in stars that is returned back to the ISM
    """
    return 0.0

class model_galaxy(object):

    def __init__(self,  t = None, Mh = None, Mg = None, Ms = None, MZ = None, Z_IGM = 1.e-4, sfrmodel = None, cosmo = None, verbose = False):

        self.Zsun = 0.02

        if cosmo is not None: 
            self.cosmo = cosmo
            self.fbuni = cosmo.Ob0/cosmo.Om0
        else:
            errmsg = 'to initialize gal object it is mandatory to supply the collossus cosmo(logy) object!'
            raise Exception(errmsg)
            return
            
        if Mh is not None: 
            self.Mh = Mh
        else:
            errmsg = 'to initialize gal object it is mandatory to supply Mh!'
            raise Exception(errmsg)
            return
            
        if t is not None: 
            self.t = t # in Gyrs
            self.z = self.cosmo.age(t, inverse=True)        
            self.gr = self.cosmo.growthFactor(self.z)
            self.dDdz = self.cosmo.growthFactor(self.z,derivative=1)
            self.thubble = self.cosmo.hubbleTime(self.z)
            self.dDdt = self.cosmo.growthFactor(self.z, derivative=1) * self.cosmo.age(t, derivative=1, inverse=True)

        else:
            errmsg = 'to initialize gal object it is mandatory to supply t!'
            raise Exception(errmsg)
            return
            
        # metallicity yield of a stellar population - this is a constant derived from SN and AGB simulations
        self.yZ = 0.069; 
        # assumed metallicity of freshly accreting gas (i.e. metallicity of intergalactic medium)
        self.Z_IGM = Z_IGM; 
        
        if Ms is not None:
            self.Ms = Ms
        else: 
            self.Ms = 0.0
        if Mg is not None:
            self.Mg = Mg
        else: 
            self.Mg = self.fbuni*Mh
            
        if MZ is not None:
            self.MZ = MZ
        else: 
            self.MZ = self.Z_IGM*self.Mg
        if MZ is not None and Mg is not None:
            # model for molecular hydrogen content is to be implemented here
            self.MH2 = 0.0
        else:
            self.MH2 = 0.0
        
        # only one model based on total gas density for starters, a better model is to be implemented
        self.sfr_models = {'gaslinear': self.SFRgaslinear}
    
        if sfrmodel is not None: 
            try: 
                self.sfr_models[sfrmodel]
            except KeyError:
                print "unrecognized sfrmodel in model_galaxy.__init__:", sfrmodel
                print "available models:", self.sfr_models
                return
            self.sfrmodel = sfrmodel
        else:
            errmsg = 'to initialize gal object it is mandatory to supply sfrmodel!'
            raise Exception(errmsg)
            return
            
        if verbose is not None:
            self.verbose = verbose

        self.epsout = self.eps_out(); self.Mgin = self.Mg_in(t); 
        self.Msin = self.Ms_in(t)
        self.sfr = self.SFR(t)
        self.Rloss = R_loss(0.); self.Rloss1 = 1.0-self.Rloss

        return
        
    
    def dMhdt(self, Mcurrent, t):
        """
        halo mass accretion rate using approximation of eqs 3-4 of Krumholz & Dekel 2012
        output: total mass accretion rate in Msun/h /Gyr
        """
        self.Mh = Mcurrent
        dummy = 1.06e12*self.cosmo.h*(Mcurrent/self.cosmo.h/1.e12)**1.14 *self.dDdt/(self.gr*self.gr)

        # approximation in Krumholz & Dekel (2012) for testing 
        #dummy = 5.02e10*(Mcurrent/1.e12)**1.14*(1.+self.z+0.093/(1.+self.z)**1.22)**2.5    

        return dummy
        
    def Mhot(self, z):
        mnl = self.cosmo.nonLinearMass(z)
        mhot = 1.e12 * self.cosmo.h * np.maximum(2., 1.e12/(3.*mnl/self.cosmo.h))
        return mhot
        
    def eps_in(self, t, z, Mh):
        """
        fraction of universal baryon fraction that makes it into galaxy 
        along with da
        """
        Mps = self.cosmo.massFromPeakHeight(1.0,z) #returns in solar mass/h
        Mps = Mps*self.cosmo.h
        M12max = max(2, 1.e12/(3*Mps))
	if Mh*self.cosmo.h/1.e12 < M12max:
            epsin = 1.0
        else:
            epsin = 0.0
        return epsin

    def Mg_in(self, t):
        dummy = self.fbuni*self.eps_in(t, self.z, self.Mh)*fg_in(self.Mh, self.z)*self.dMhdt(self.Mh,t)
        return dummy
    
    def Ms_in(self, t):
        dummy = self.fbuni*(1.0-fg_in(self.Mh, self.z))*self.dMhdt(self.Mh,t)
        return dummy

    def tau_sf(self):
        """
        gas consumption time in Gyrs 
        """
        return 2.5

    def SFRgaslinear(self, t):
        return self.Mg/self.tau_sf()
         
    def SFR(self, t):
        """
        master routine for SFR - 
        eventually can realize more star formation models
        """  
        return self.sfr_models[self.sfrmodel](t)
        
    def dMsdt(self, Mcurrent, t):
        dummy = self.Msin + self.Rloss1*self.sfr
        return dummy

    def eps_out(self):
        return 1.0
        
    def dMgdt(self, Mcurrent, t):
        dummy = self.Mgin - (self.Rloss1 + self.epsout)*self.sfr
        return dummy

    def zeta(self):
        """
        output: fraction of newly produced metals removed by SNe in outflows
        """
        return 0.0

    def dMZdt(self, Mcurrent, t):
        dummy = self.Z_IGM*self.Mgin + (self.yZ*self.Rloss1*(1.-self.zeta()) - (self.Rloss1+self.epsout)*self.MZ/(self.Mg))*self.sfr
        return dummy
        
    def evolve(self, Mcurrent, t):
        # first set auxiliary quantities and current masses
        self.z = self.cosmo.age(t, inverse=True)        
        self.gr = self.cosmo.growthFactor(self.z)
        self.dDdz = self.cosmo.growthFactor(self.z,derivative=1)
        self.thubble = self.cosmo.hubbleTime(self.z)
        self.dDdt = self.cosmo.growthFactor(self.z, derivative=1) * self.cosmo.age(t, derivative=1, inverse=True)

        self.Mh = Mcurrent[0]; self.Mg = Mcurrent[1]; 
        self.Ms = Mcurrent[2]; self.MZ = Mcurrent[3]
        self.epsout = self.eps_out(); self.Mgin = self.Mg_in(t); 
        self.Msin = self.Ms_in(t)
        self.Rloss = R_loss(0.); self.Rloss1 = 1.0-self.Rloss
        self.sfr = self.SFR(t)*self.cosmo.h
        
        # calculate rates for halo mass, gas mass, stellar mass, and mass of metals
        dMhdtd = self.dMhdt(Mcurrent[0], t)
        dMgdtd = self.dMgdt(Mcurrent[1], t)
        dMsdtd = self.dMsdt(Mcurrent[2], t)
        dMZdtd = self.dMZdt(Mcurrent[3], t)
        if self.verbose:
            print "evolution: t=%2.3f Mh=%.2e, Mg=%.2e, Ms=%.2e, Z/Zsun=%2.2f,SFR=%4.1f"%(t,self.Mh,self.Mg,self.Ms,self.MZ/self.Mg/0.02,self.SFR(t)*1.e-9)

        return [dMhdtd, dMgdtd, dMsdtd, dMZdtd]

def plot_pretty():
    plt.rc('text', usetex=True)
    plt.rc('font',size=20)
    plt.rc('xtick.major',pad=5); plt.rc('xtick.minor',pad=5)
    plt.rc('ytick.major',pad=5); plt.rc('ytick.minor',pad=5)

def test_galaxy_evolution(Minit, sfrmodel, cosmo, verbose, figsize=(4,4)):
    tu = 13.65
    zg = np.linspace(20., 0., 40)
    t_output = cosmo.age(zg)

    g = model_galaxy(t = t_output[0], Mh = Minit, Mg = None, Ms = None, MZ = None, sfrmodel = sfrmodel, cosmo = cosmo, verbose = verbose)
    
    y0 = np.array([g.Mh, g.Mg, g.Ms, g.MZ])
    
    Mout = odeint(g.evolve, y0, t_output, rtol = 1.e-7, mxstep = 4000)
    
    # let's split the output into arrays with intuitive names
    Mhout = np.clip(Mout[:,0],0.0,1.e100)
    Mgout = np.clip(Mout[:,1],0.0,1.e100)
    Msout = np.clip(Mout[:,2],0.0,1.e100)
    MZout = np.clip(Mout[:,3],0.0,1.e100)

    #
    # plot
    #
    fig = plt.figure(figsize=figsize)
    plt.xlabel(r'$t\ \rm (Gyr)$')
    plt.ylabel(r'$M_{\rm h},\ M_{\rm g},\ M_{*},\ M_{\rm Z}\ (\rm M_\odot)$')
    plt.xlim(0.,tu); #plt.ylim(1.e7,3.e12)
    plt.yscale('log')
    plt.plot(t_output, Mhout, lw = 3.0, label=r'$M_{\rm vir}$')
    plt.plot(t_output, Mgout, lw = 3.0, label=r'$M_{\rm gas}$')
    plt.plot(t_output, Msout, lw = 3.0, label=r'$M_{\rm *}$')
    plt.plot(t_output, MZout, lw = 3.0, label=r'$M_{\rm Z}$')

    plt.legend(frameon=False,loc='lower right', ncol=2, fontsize=9)
    plt.grid()
    plt.show()
    
    #
    # plot SFR and Z(t)
    #
    plt.figure(figsize=figsize)
    plt.xlabel(r'$t\ \rm (Gyr)$')
    plt.ylabel(r'$Z= M_{\rm Z}/M_{\rm g}\ (Z_\odot),\rm\ SFR\ (M_\odot/yr)$')
    plt.xlim(0.,tu); plt.ylim(1.e-5,15.)
    plt.yscale('log')
    plt.plot(t_output, MZout/Mgout/0.02, lw = 3.0, label=r'$Z$')
    SFR = np.zeros_like(t_output); Rdisk = np.zeros_like(t_output)
    for i, td in enumerate(t_output):
        ge = model_galaxy(t = td, Mh = Mhout[i], Mg = Mgout[i], Ms = Msout[i], MZ = MZout[i], sfrmodel = sfrmodel, cosmo = cosmo)
        SFR[i] = ge.SFR(td)*1.e-9;
    plt.plot(t_output,SFR, lw = 3.0, label=r'$\rm SFR$')

    plt.legend(frameon=False,loc='lower right', ncol=2, fontsize=9)
    plt.grid()
    plt.show()
