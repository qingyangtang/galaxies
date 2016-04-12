import matplotlib.pylab as plt
import numpy as np
import scipy.optimize as opt
from matplotlib.colors import LogNorm

from astroML.datasets import fetch_sdss_spectrum
from scipy.interpolate import interp1d
from scipy.integrate import simps

from setup import image_home_dir
from fetch_sdss_image import fetch_sdss_image
#from code import * 
from PIL import Image

from setup import setup 
import os

def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    import matplotlib.pyplot as plt

    plt.rc("savefig", dpi=dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', size=fontsize)
    plt.rc('xtick.major', pad=5); plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5); plt.rc('ytick.minor', pad=5)
    return

def plot_image_spec_sdss_galaxy(sdss_obj, save_figure=False):
    #plot image and spectrum of a specified SDSS galaxy
    # input sdss_obj = individual SDSS main galaxy data base entry
    # save_figure specifies whether PDF of the figure will be saved in fig/ subdirectory for the record
    #
    # define plot with 2 horizonthal panels of appropriate size
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 2.))
    # set an appropriate font size for labels
    plt.rc('font',size=8)

    #get RA and DEC of the galaxy and form the filename to save SDSS image
    RA = sdss_obj['ra']; DEC = sdss_obj['dec']; scale=0.25
    outfile = image_home_dir()+str(sdss_obj['objID'])+'.jpg'
    fetch_sdss_image(outfile, RA, DEC, scale)
    img = Image.open(outfile)
    # do not plot pixel axis labels
    ax0.axis('off')
    ax0.imshow(img)

    # fetch SDSS spectrum using plate number, epoch, and fiber ID
    plate = sdss_obj['plate']; mjd = sdss_obj['mjd']; fiber = sdss_obj['fiberID']
    spec = fetch_sdss_spectrum(plate, mjd, fiber)

    # normalize spectrum for plotting
    spectrum = 0.5 * spec.spectrum  / spec.spectrum.max()
    lam = spec.wavelength()
    text_kwargs = dict(ha='center', va='center', alpha=0.5, fontsize=10)

    # set axis limits for spectrum plot
    ax1.set_xlim(3000, 10000)
    ax1.set_ylim(0, 0.6)

    color = np.zeros(5)
    for i, f, c, loc in zip([0,1,2,3,4],'ugriz', 'bgrmk', [3500, 4600, 6100, 7500, 8800]):
        data_home = setup.sdss_filter_dir()
        archive_file = os.path.join(data_home, '%s.dat' % f)
        if not os.path.exists(archive_file):
            raise ValueError("Error in plot_img_spec_sdss_galaxy: filter file '%s' does not exist!" % archive_file )
        F = open(archive_file)
        filt = np.loadtxt(F, unpack=True)
        ax1.fill(filt[0], filt[2], ec=c, fc=c, alpha=0.4)
        fsp = interp1d(filt[0],filt[2], bounds_error=False, fill_value=0.0)
        ax1.text(loc, 0.03, f, color=c, **text_kwargs)
        # compute magnitude in each band using simple Simpson integration of spectrum and filter using eq 1.2 in the notes
        fspn = lam*fsp(lam)/simps(fsp(lam)/lam,lam)
        lamf = fspn; #lamf = lamf.clip(0.0)
        specf = spec.spectrum * lamf
        color[i] = -2.5 * np.log10(simps(specf,lam)) 
        
    # print estimated g-r color for the object
    grcatcol = sdss_obj['modelMag_g']-sdss_obj['modelMag_r'] 
    print "computed g-r = ", color[1]-color[2]
    print "catalog g-r=", grcatcol

    ax1.plot(lam, spectrum, '-k', lw=0.5, label=r'$(g-r)=%.2f$'%grcatcol )
    ax1.set_xlabel(r'$\lambda {(\rm \AA)}$')
    ax1.set_ylabel(r'$\mathrm{norm.\ specific\ flux}\ \ \ f_{\lambda}$')
    #ax.set_title('Plate = %(plate)i, MJD = %(mjd)i, Fiber = %(fiber)i' % locals())
    #ax.set_title('%s' % objects[obj_]['name'])
    ax1.legend(frameon=False)

    if save_figure: 
        plt.savefig('fig/gal_img_spec'+'_'+str(sdss_obj['objID'])+'.pdf',bbox_inches='tight')

    plt.subplots_adjust(wspace = 0.2)
    plt.show()
    return
    
from itertools import product
import scipy.optimize as opt
from matplotlib.colors import LogNorm
from numpy import random as rnd
import os


    
def sdss_img_collage(objs, ras, decs, nrow, ncol, npix, scale, savefig=None):
    from PIL import Image
    from code.fetch_sdss_image import fetch_sdss_image
    from code.setup import image_home_dir
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol, nrow))

    # Check that PIL is installed for jpg support
    if 'jpg' not in fig.canvas.get_supported_filetypes():
        raise ValueError("PIL required to load SDSS jpeg images")

    for _obj, ra, dec, ax in zip(objs, ras, decs, axs.flatten()):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        outfile = image_home_dir()+str(_obj)+'.jpg'
        fetch_sdss_image(outfile, ra, dec, scale=scale, width=npix, height=npix)
        I = Image.open(outfile)
        ax.imshow(I,origin='lower')

    plt.tight_layout(pad=0.0)
    plt.show()
    if savefig != None:
        plt.savefig(savefig,bbox_inches='tight')
        
    
def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level

def plot_2d_dist(x,y, xlim,ylim,nxbins,nybins, weights=None, xlabel='x',ylabel='y', clevs=None, fig_setup=None, savefig=None):
    if fig_setup == None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        #ax = plt.add_subplot(1,1,1)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    else:
        ax = fig_setup
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    #
    if xlim[1] < 0.: ax.invert_xaxis()

    if weights == None: weights = np.ones_like(x)
    H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(xlim[0], xlim[1], nxbins),np.linspace(ylim[0], ylim[1], nybins)))
    
    H = np.rot90(H); H = np.flipud(H); H = H/np.sum(H)        
    Hmask = np.ma.masked_where(H==0,H)
             
    X,Y = np.meshgrid(xbins,ybins) 
    pcol = ax.pcolormesh(X,Y,(Hmask), cmap=plt.cm.BuPu, norm = LogNorm(), linewidth=0., rasterized=True)
    pcol.set_edgecolor('face')

    if clevs != None:
        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
                   
        ax.contour(H, linewidths=(1.0,0.75, 0.5, 0.25), colors='black', levels = lvls, 
                    norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    if fig_setup == None:
        plt.show()
    return


    
def fetch_image(objid, ra, dec, scale, npix):
    from PIL import Image
    from code.fetch_sdss_image import fetch_sdss_image
    from code.setup import image_home_dir

    outfile = image_home_dir()+str(objid)+str(np.round(scale,2))+'.jpg'
    if not os.path.isfile(outfile):
        fetch_sdss_image(outfile, ra, dec, scale=scale, width=npix, height=npix)
    return Image.open(outfile)

def plot_sdss_collage_with_2d_dist(objs=None, ras=None, decs=None, xs=None, ys=None, xlab='x', ylab='y', 
                                   xlims=None, ylims=None, nrows=3, ncols=3, npix = 150, show_axis=False, facecolor='white',
                                   clevs = None, ncont_bins = None, 
                                   rnd_seed=None, dA = None, kpc_per_npix = 25, outfile=None):
                                   
    """
    Plot a collage of SDSS images (downloading them if needed) for a list of SDSS objects
    with declinations decs and right ascensions ras ordered on a grid by properties xs and ys (x and y-axis)
    + plot contours of object distribution on top of images if needed
    
    Parameters
    --------------------------------------------------------------------------------------------------------
    objs: array_like
           list of SDSS objIDs 
    ras:  array_like
           list of R.A.s of objects in objs of the same size as objs
    decs: array_like 
          list of DECs of objects in objs of the same size as objs
    show_axis: bool
          show axis with labels if True
    xs: array_like
        property of objects in objs to order along x
    ys: array_like
        property of objects in objs to order along x
    """

    arcsec_to_rad = np.pi/180./3600.
    samp_dist = 0.2
    #axes ranges and number of images along each axis
    if xlims == None:
        xmin = np.min(xs); xmax = np.max(xs)
        xlims = np.array([xmin,xmax])
    if ylims == None:
        ymin = 0.95*np.min(ys); ymax = 1.05*np.max(ys)
        ylims = np.array([ymin, ymax])
        
    dxh = 0.5*np.abs(xlims[1] - xlims[0])/ncols; dyh = 0.5*np.abs(ylims[1] - ylims[0])/nrows
    
    xgrid = np.linspace(xlims[0]+dxh, xlims[1]-dxh, ncols)
    ygrid = np.linspace(ylims[0]+dyh, ylims[1]-dyh, nrows)

    fig, ax = plt.subplots(1,1,figsize=(5, 5))    
    #fig.patch.set_facecolor('white')
    ax.patch.set_facecolor(facecolor)
    if facecolor == 'black' and show_axis == True:
        ecol = 'whitesmoke'
        ax.tick_params(color=ecol, labelcolor='black')
        for spine in ax.spines.values():
            spine.set_edgecolor(ecol)

    ax.set_xlim(xlims[0], xlims[1]); ax.set_ylim(ylims[0], ylims[1])
    if xlims[1] < 0.: ax.invert_xaxis()
    #if ylims[1] < ylims[0]: ax.invert_yaxis()

    if not show_axis:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    
    from itertools import product
    # Check that PIL is installed for jpg support
    if 'jpg' not in fig.canvas.get_supported_filetypes():
        raise ValueError("PIL required to load SDSS jpeg images")
    
    np.random.seed(rnd_seed)
    for xi, yi in product(xgrid, ygrid):
        inds = ((xs > xi-samp_dist*dxh) & (xs < xi+samp_dist*dxh) &
                (ys > yi-samp_dist*dyh) & (ys < yi+samp_dist*dyh))
        _objs = objs[inds]; _ras = ras[inds]; _decs = decs[inds]
        lobjs = len(_objs)
        if lobjs < 3 : continue
        if lobjs == 1: 
            iran = 0
        else:    
            iran = np.random.randint(0,lobjs-1,1)
        if dA[0] != None: 
            _dA = dA[inds]
            dAi = _dA[iran]
            img_scale = kpc_per_npix/(dAi*1.e3*npix*arcsec_to_rad)
        else:
            img_scale = 0.2
        I = fetch_image(_objs[iran],_ras[iran],_decs[iran],img_scale, npix)
        ax.imshow(I, extent=[xi-dxh, xi+dxh, yi-dyh, yi+dyh])

    ax.set_aspect(dxh/dyh)
    
    # add contours if ncont_bins is specified on input
    if ncont_bins != None:
        if clevs == None:
            raise Exception('ncont_bin is specified but contour levels clevs is not!')
            
        contours_bins = np.linspace(xlims[0], xlims[1], ncont_bins), np.linspace(ylims[0], ylims[1], ncont_bins)

        H, xbins, ybins = np.histogram2d(xs, ys, bins=contours_bins)
        H = np.rot90(H); H = np.flipud(H); Hmask = np.ma.masked_where(H==0,H)
        H = H/np.sum(H)        

        X,Y = np.meshgrid(xbins,ybins) 

        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
        
        ax.contour(H, linewidths=np.linspace(1,2,len(lvls))[::-1], 
                    colors='whitesmoke', alpha=0.4, levels = lvls, norm = LogNorm(), 
                    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]], interpolation='bicubic')

    # save plot if file is specified 
    if outfile != None:
            plt.savefig(outfile, bbox_inches='tight')
            
    plt.show()

def plot_sdss_collage_with_2d_dist_old(objs=None, ras=None, decs=None, show_axis=False, xs=None, ys=None, xlab='x', ylab='y', 
                                   xlims=None, ylims=None, nrows=3, ncols=3, npix = 150, clevs = None, ncont_bins = None, 
                                   rnd_seed=None, dA = None, kpc_per_npix = 25, outfile=None):
    """
    Plot a collage of SDSS images (downloading them if needed) for a list of SDSS objects
    with declinations decs and right ascensions ras ordered on a grid by properties xs and ys (x and y-axis)
    + plot contours of object distribution on top of images if needed
    
    Parameters
    --------------------------------------------------------------------------------------------------------
    objs: array_like
           list of SDSS objIDs 
    ras:  array_like
           list of R.A.s of objects in objs of the same size as objs
    decs: array_like 
          list of DECs of objects in objs of the same size as objs
    show_axis: bool
          show axis with labels if True
    xs: array_like
        property of objects in objs to order along x
    ys: array_like
        property of objects in objs to order along x
    """
    arcsec_to_rad = np.pi/180./3600.
    samp_dist = 0.2
    #axes ranges and number of images along each axis
    if xlims == None:
        xmin = np.min(xs); xmax = np.max(xs)
        xlims = np.array([xmin,xmax])
    if ylims == None:
        ymin = 0.95*np.min(ys); ymax = 1.05*np.max(ys)
        ylims = np.array([ymin, ymax])
        
    dxh = 0.5*(xlims[1] - xlims[0])/ncols; dyh = 0.5*(ylims[1] - ylims[0])/nrows
    
    xgrid = np.linspace(xlims[0]+dxh, xlims[1]-dxh, ncols)
    ygrid = np.linspace(ylims[0]+dyh, ylims[1]-dyh, nrows)

    fig, ax = plt.subplots(1,1,figsize=(5, 5))    
    ax.set_xlim(xlims[0], xlims[1]); ax.set_ylim(ylims[0], ylims[1])
    if xlims[1] < 0.: ax.invert_xaxis()
    if ylims[1] < 0.: ax.invert_yaxis()

    if not show_axis:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    
    from itertools import product
    # Check that PIL is installed for jpg support
    if 'jpg' not in fig.canvas.get_supported_filetypes():
        raise ValueError("PIL required to load SDSS jpeg images")
    
    np.random.seed(rnd_seed)
    for xi, yi in product(xgrid, ygrid):
        inds = ((xs > xi-samp_dist*dxh) & (xs < xi+samp_dist*dxh) &
                (ys > yi-samp_dist*dyh) & (ys < yi+samp_dist*dyh))
        _objs = objs[inds]; _ras = ras[inds]; _decs = decs[inds]
        lobjs = len(_objs)
        if lobjs < 3 : continue
        if lobjs == 1: 
            iran = 0
        else:    
            iran = np.random.randint(0,lobjs-1,1)
        if dA[0] != None: 
            _dA = dA[inds]
            dAi = _dA[iran]
            img_scale = kpc_per_npix/(dAi*1.e3*npix*arcsec_to_rad)
        else:
            img_scale = 0.2
        I = fetch_image(_objs[iran],_ras[iran],_decs[iran],img_scale, npix)
        ax.imshow(I, extent=[xi-dxh, xi+dxh, yi-dyh, yi+dyh])

    ax.set_aspect(dxh/dyh)
    
    # add contours if ncont_bins is specified on input
    if ncont_bins != None:
        if clevs == None:
            raise Exception('ncont_bin is specified but contour levels clevs is not!')
            
        contours_bins = np.linspace(xlims[0], xlims[1], ncont_bins), np.linspace(ylims[0], ylims[1], ncont_bins)

        H, xbins, ybins = np.histogram2d(xs, ys, bins=contours_bins)
        H = np.rot90(H); H = np.flipud(H); Hmask = np.ma.masked_where(H==0,H)
        H = H/np.sum(H)        

        X,Y = np.meshgrid(xbins,ybins) 

        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
        
        ax.contour(H, linewidths=np.linspace(1,2,len(lvls))[::-1], 
                    colors='whitesmoke', alpha=0.4, levels = lvls, norm = LogNorm(), 
                    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]], interpolation='bicubic')

    # save plot if file is specified 
    if outfile != None:
            plt.savefig(outfile, bbox_inches='tight')
            
    plt.show()


def plot_sdss_collage_with_2d_dist_old(objs=None, ras=None, decs=None, xs=None, ys=None, xlab='x', ylab='y', 
                                   xlims=None, ylims=None, nrows=3, ncols=3, clevs = None, 
                                   npix = 150, rnd_seed=None, ncont_bins = 51, dA = None, outfile=None):
    arcsec_to_rad = np.pi/180./3600.
    kpc_per_npix = 25
    samp_dist = 0.2
    #axes ranges and number of images along each axis
    if xlims == None:
        xmin = np.min(xs); xmax = np.max(xs)
        xlims = np.array([xmin,xmax])
    if ylims == None:
        ymin = 0.95*np.min(ys); ymax = 1.05*np.max(ys)
        ylims = np.array([ymin, ymax])
        
    dxh = 0.5*(xlims[1] - xlims[0])/ncols; dyh = 0.5*(ylims[1] - ylims[0])/nrows
    
    xgrid = np.linspace(xlims[0]+dxh, xlims[1]-dxh, ncols)
    ygrid = np.linspace(ylims[0]+dyh, ylims[1]-dyh, nrows)

    fig, ax = plt.subplots(1,1,figsize=(5, 5))    
    ax.set_xlim(xlims[0], xlims[1]); ax.set_ylim(ylims[0], ylims[1])
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    if xlims[1] < 0.: ax.invert_xaxis()
    if ylims[1] < 0.: ax.invert_yaxis()
    
    from itertools import product
    # Check that PIL is installed for jpg support
    if 'jpg' not in fig.canvas.get_supported_filetypes():
        raise ValueError("PIL required to load SDSS jpeg images")
    
    np.random.seed(rnd_seed)
    for xi, yi in product(xgrid, ygrid):
        inds = ((xs > xi-samp_dist*dxh) & (xs < xi+samp_dist*dxh) &
                (ys > yi-samp_dist*dyh) & (ys < yi+samp_dist*dyh))
        _objs = objs[inds]; _ras = ras[inds]; _decs = decs[inds]
        lobjs = len(_objs)
        if lobjs < 3 : continue
        if lobjs == 1: 
            iran = 0
        else:    
            iran = np.random.randint(0,lobjs-1,1)
        if dA[0] != None: 
            _dA = dA[inds]
            dAi = _dA[iran]
            img_scale = kpc_per_npix/(dAi*1.e3*npix*arcsec_to_rad)
        else:
            img_scale = 0.2
        I = fetch_image(_objs[iran],_ras[iran],_decs[iran],img_scale, npix)
        ax.imshow(I, extent=[xi-dxh, xi+dxh, yi-dyh, yi+dyh])

    ax.set_aspect(dxh/dyh)
    
    contours_bins = np.linspace(xlims[0], xlims[1], ncont_bins), np.linspace(ylims[0], ylims[1], ncont_bins)

    # add contours:
    H, xbins, ybins = np.histogram2d(xs, ys, bins=contours_bins)

    H = np.rot90(H); H = np.flipud(H); Hmask = np.ma.masked_where(H==0,H)
    H = H/np.sum(H)        

    X,Y = np.meshgrid(xbins,ybins) 

    lvls = []
    for cld in clevs:  
        sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
        lvls.append(sig)
        
    ax.contour(H, linewidths=np.linspace(1,2,len(lvls))[::-1], 
                colors='whitesmoke', alpha=0.4, levels = lvls, norm = LogNorm(), 
                extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]], interpolation='bicubic')

    if outfile != None:
        plt.savefig(outfile)
    plt.show()

    
if __name__ == '__main__':
    from read_sdss_fits import read_sdss_fits
    from setup import data_home_dir

    # read fits file with the SDSS DR8 main spectroscopic sample
    data = read_sdss_fits(data_home_dir()+'SDSSspecgalsDR8.fit')

    z_max = 0.04

    # redshift cut
    sdata = data[data['z'] < z_max]
    mr = sdata['modelMag_r']
    gr = sdata['modelMag_g'] - sdata['modelMag_r']
    r50 = sdata['petroR50_r']
    sb = mr - 2.5*np.log10(0.5) + 2.5*np.log10(np.pi*(r50)**2)

    from colossus.cosmology import cosmology

    # set cosmology to the best values from 9-year WMAP data
    cosmo = cosmology.setCosmology('WMAP9')

    # compute luminosity and angular distances
    d_L = cosmo.luminosityDistance(sdata['z'])
    d_A = d_L/(1.+sdata['z'])**2
    # absolute magnitude in the r-band; for such small redshifts K-correction is negligible, so we ignore it
    M_r = mr - 5.0*np.log10(d_L/1e-5) 

    # select a volume complete sample of SDSS galaxies down to a given luminosity ()
    Mlim = -17.0; Dmax = 70.
    sdata = sdata[(M_r < Mlim) & (d_A < Dmax)]
    M_rs  = M_r[(M_r < Mlim) & (d_A < Dmax)]
    
    from random import randint
    import numpy as np

    # select a random galaxy from sdata
    iran = randint(0,np.size(sdata)-1)
    randobj = sdata[iran]

    # plot its image and spectrum
    plot_image_spec_sdss_galaxy(randobj)

    # print some key properties from the SDSS catalog
    print "SDSS objID=", randobj['objID']
    print "r-band absolute magnitude, M_r=", M_rs[iran]
    r50s = randobj['petroR50_r']
    print "light concentration in the r-band, c_r=", randobj['petroR90_r']/r50s
    print "fraction of light profile fit by the de Vaucouleurs (spheroidal) component in the r-band, fracdeV_r=",randobj['fracdeV_r']
    sbs = randobj['modelMag_r'] - 2.5*np.log10(0.5) + 2.5*np.log10(np.pi*(r50s)**2)
    print "r-band surface brightness, mu_r=", sbs, " mag/arcsec^2"
    print "r-band axis ratio q=b/a of ellipsoidal fit to light distribution=", randobj['expAB_r']

    import py_compile
    from setup import setup
    py_compile.compile(os.path.join(setup.code_home_dir(),'fetch_sdss_image.py'))
    
    