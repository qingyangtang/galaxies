import os
import urllib, cStringIO
from PIL import Image
import matplotlib.pyplot as plt
from setup import setup

def fetch_sdss_image(outfile, RA, DEC, scale=0.2, width=400, height=400):
    """Fetch the image at the given RA, DEC from the SDSS server"""
    url = ("http://skyservice.pha.jhu.edu/DR8/ImgCutout/"
           "getjpeg.aspx?ra=%.8f&dec=%.8f&scale=%.2f&width=%i&height=%i"
           % (RA, DEC, scale, width, height))
    # uncomment for debugging if needed
    #print "downloading %s" % url
    #print " -> %s" % outfile
    fd = cStringIO.StringIO(urllib.urlopen(url).read())
    im = Image.open(fd)
    im.save(outfile)

if __name__ == '__main__':
    fig = plt.figure(figsize=(6, 6))

    # Check that PIL is installed for jpg support
    if 'jpg' not in fig.canvas.get_supported_filetypes():
        raise ValueError("PIL required to load SDSS jpeg images")
        
    outfile='img/ngc4449.jpg'
    RA=187.04626325; DEC=44.09362883; ObjId = 588017603611262984
    scale = 1.0
    fetch_sdss_image(outfile, RA, DEC, scale)
    img = Image.open(outfile)
    plt.imshow(img)
    plt.show()
    
    import py_compile
    py_compile.compile(os.path.join(setup.code_home_dir(),'fetch_sdss_image.py'))

