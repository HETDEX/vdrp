""" Utility functions for virus reductions

.. moduleauthor:: Maximilian Fabricius <mxhf@mpe.mpg.de>
"""
import os
from collections import OrderedDict

def createDir(directory):
    """ Creates a directory.
    Does not raise an excpetion if the directory already exists.

    Args:
        directory (string): Name for directory to create.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        logging.error('Creating directory. ' +  directory)



def read_radec(filename):
    """ Reads radec.dat file and returns ra,dec,pa.

    Args:
        filename (str): Filename, typically radec.dat or radec2.dat.

    Returns:
        float,float,float: 3 element list with RA, DEC and PA
    """
    with open(filename,"r") as f:
        ll = f.readlines()
    ra, dec, pa = ll[0].split()
    return float(ra), float(dec), float(pa)


def write_radec(ra,dec,pa,filename):
    """ Creates radec.dat-type  file and returns ra,dec,pa.

    Args:
        ra (float): Right ascension
        dec (float): declination
        pa (float): position angle
        filename (str): Filename, typically radec.dat or radec2.dat.

    """
    with open(filename,"w") as f:
        s = "{:.6f} {:.6f} {:.6f}\n".format(ra, dec, pa)
        f.write(s)


def read_all_mch(all_mch):
    """ Reads all.mch and returns dither information.

    Args:
        all_mch (str): Filename, typically all.mch.

    Returns:
        (OrdereDict): Dictionary of float tuples, with dither offsets, 
            e.g. {1 : (0.,0.), 2 : (1.27,-0.73), 3 : (1.27,0.73)}
    """
    dither_offsets = OrderedDict()
    with open(all_mch,"r") as f:
        ll = f.readlines()
        for i,l in enumerate(ll):
            tt = l.split()
            dx,dy = float(tt[2]), float(tt[3])
        dither_offsets[i+1] = (dx,dy)
    return dither_offsets


def rm(ff):
    """ Takes a list of files names and deletes them.
    Does not raise an Exception if a specific file was not in place.

    Args:
        ff (list): List of file names to delete.

    """
    for f in ff:
        try:
            os.remove(f)
        except:
            pass
