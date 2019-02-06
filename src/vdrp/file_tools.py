# VDRP File access tools.
#
# These serve as wrappers around access to different pre-reduced files,
# allowing easy switching to HDF5


import os


def get_mulitfits_file(basedir, night, shot, expname, fname):

    fpath = os.path.join(basedir, night, 'virus', 'virus%07d' % shot,
                         expname, fname+'.fits')
    return fpath


def get_dithall_file(basedir, night, shot):

    dithall_file = os.path.join(basedir, night+'v'+shot, 'dithall.use')

    return dithall_file


def get_throughput_file(path, night, shot):
    """
    Equivalent of rtp0 script.

    Checks if a night/shot specific throughput file exists.

    If true, return the filename, otherise the filename
    for an average throughput file.

    Parameters
    ----------
    path : str
        Path to the throughput files
    shotname : str
        Name of the shot
    """

    tpfile = os.path.join(path, night+'v'+shot+"sedtp_f.dat")
    if os.path.exists(tpfile):
        return tpfile
    else:
        return os.path.join(path, "tpavg.dat")


def get_norm_file(path, fname):

    norm_file = os.path.join(path, fname+".norm")
    return norm_file
