#!/usr/bin/env python
""" Spectrum extraction routines

Contains python translation of Karl Gebhardt

.. moduleauthor:: Jan Snigula <snigula@mpe.mpg.de>
"""

from __future__ import print_function

import os
import logging
import logging.config
from astropy.io import fits
import numpy as np

import vdrp.programs as vp

import vdrp.containers as vcont
import vdrp.file_tools as vft


_baseDir = os.getcwd()

_logger = logging.getLogger()


def getDefaults():

    defaults = {}

    defaults['dithall_dir'] = '/work/00115/gebhardt/maverick/detect/'
    defaults['multifits_dir'] = '/work/03946/hetdex/maverick/red1/reductions/'
    defaults['tp_dir'] = '/work/00115/gebhardt/maverick/detect/tp/'
    defaults['norm_dir'] = '/work/00115/gebhardt/maverick/getampnorm/all/'

    defaults['radec_file'] = '/work/00115/gebhardt/maverick/getfib/radec.all'

    defaults['ifu_search_radius'] = 4.
    defaults['shot_search_radius'] = 600.

    return defaults


def get_arguments(parser):
    '''
    Add command line arguments for the photometry routines, this function
    can be called from another tool.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    '''

    parser.add_argument("--dithall_dir", type=str, help="Base directory "
                        "used to find the dithall.use files")
    parser.add_argument("--multifits_dir", type=str, help="Directory "
                        "with the multi extension fits files")
    parser.add_argument("--tp_dir", type=str, help="Directory "
                        "with the throughput files")
    parser.add_argument("--norm_dir", type=str, help="Directory "
                        "with the amplifier normalization files")

    parser.add_argument("--radec_file", type=str, help="Filename of file with "
                        "RA DEC PA positions for all shots")

    parser.add_argument("--ifu_search_radius", type=float, help="Radius for "
                        "search for fibers near a given star.")
    parser.add_argument("--shot_search_radius", type=float, help="Radius for "
                        "search for shots near a given star.")

    return parser


def get_star_spectrum_data(ra, dec, args, nightshot, multi_shot=False,
                           dithall=None):
    """
    This extracts the data about the different observations of the same star
    on different ifus.

    This is essentially the information stored in the l1 file.

    Parameters
    ----------
    ra : float
        Right Ascension of the star.
    dec : float
        Declination of the star.
    args : struct
        The arguments structure
    nightshot : tuple
        Tuple of strings with night and shot to search. if None, use all shots
        containing the given RA /DEC
    """

    if multi_shot:
        # First find matching shots
        _logger.info('Reading radec file %s' % args.radec_file)

        night, shot = np.loadtxt(args.radec_file, unpack=True, dtype='U50',
                                 usecols=[0, 1])
        ra_shot, dec_shot = np.loadtxt(args.radec_file, unpack=True,
                                       usecols=[2, 3])

        _logger.info('Searching for shots within %f arcseconds of %f %f'
                     % (args.shot_search_radius, ra, dec))
        # First find shots overlapping with the RA/DEC coordinates
        w_s = np.where(((np.sqrt((np.cos(dec/57.3)*(ra_shot-ra))**2
                                 + (dec_shot-dec)**2)*3600.)
                        < args.shot_search_radius))[0]

        if not len(np.where(w_s)[0]):
            raise vcont.NoShotsException('No shots found!')

        night = night[w_s]
        shot = shot[w_s]

    else:
        night, shot = map(list, zip(nightshot))

    night_shots = []
    starobs = []
    c = 0

    _logger.info('Found %d shots' % len(shot))

    for n, s in zip(night, shot):
        if multi_shot or dithall is None:
            dithall_file = vft.get_dithall_file(args.dithall_dir, n, s)
            _logger.info('Reading dithall file %s' % dithall_file)
            try:
                dithall = vcont.DithAllFile(dithall_file)

            except Exception as e:
                _logger.warn('Failed to read %s' % dithall_file)
                _logger.exception(e)
                continue

        _logger.info('Filtering dithall file')
        filtered = dithall.where(((np.sqrt((np.cos(dec/57.3)
                                            * (dithall.ra-ra))**2
                                           + (dithall.dec-dec)**2) * 3600.)
                                  < args.ifu_search_radius))

        _logger.info('Found %d fibers' % len(filtered))

        for d in filtered:

            so = vcont.StarObservation()

            so.num = c+101
            so.night = n
            so.shot = s
            so.ra = d.ra
            so.dec = d.dec
            so.x = d.x
            so.y = d.y
            so.set_fname(d.filename)
            so.shotname = d.timestamp
            so.expname = d.expname

            so.dist = 3600.*np.sqrt((np.cos(dec/57.3)*(so.ra-ra))**2
                                    + (so.dec-dec)**2)

            # This is written to loffset
            so.offsets_ra = 3600.*(d.ra-ra)
            so.offsets_dec = 3600.*(d.dec-dec)

            # Make sure we actually have data for this shot
            fpath = vft.get_mulitfits_file(args.multifits_dir, so.night,
                                           int(so.shot), so.expname, so.fname)

            if not os.path.exists(fpath):
                print(fpath)
                _logger.warn('No fits data found for ifuslot %s in  %sv%s'
                             % (so.ifuslot, so.night, so.shot))
                continue

            starobs.append(so)
            night_shots.append('%s %s' % (n, s))

            c += 1

    return starobs, np.unique(night_shots)


def extract_star_spectrum(starobs, args, wl, wlr, wdir, prefix=''):
    """
    Equivalent of the rextsp[1] and parts of the rsp1b scripts.

    Extract stellar spectra, creating the tmp*.dat files. If prefix
    is set, it is prefixed to the tmp*dat file names.

    Parameters
    ----------
    starobs : list
        List with StarObservation objects.
    args : struct
        The arguments structure
    wdir : str
        Name of the work directory
    prefix : str (optional)
        Optional prefix for the output filenames.

    Returns
    -------
    list
        List of tmp*dat filenames created.
    """

    specfiles = []

    _logger.info('Extracting star spectrum')

    for s in starobs:
        fpath = vft.get_mulitfits_file(args.multifits_dir, s.night,
                                       int(s.shot), s.expname, s.fname)
        vp.call_imextsp(fpath, s.ifuslot, wl, wlr,
                        vft.get_throughput_file(args.tp_dir, s.night, s.shot),
                        vft.get_norm_file(args.norm_dir, s.fname),
                        prefix+'tmp%d.dat' % s.num, wdir)
        specfiles.append(prefix+'tmp%d.dat' % s.num)
    return specfiles


def get_structaz(starobs, path):
    """
    Equivalent of the rgetadc script
    Read the STRUCTAZ parameter from the multi extension fits files and fill
    in the StarObservation entries.

    Parameters:
    -----------
    starobs : list
        List with StarObservation objects.
    path : string
        Path to the directory where the multi extension fits are stored.
    """
    missingobs = False
    az_vals = []
    m_obs = []

    for obs in starobs:
        fpath = '%s/%s/virus/virus%07d/%s/virus/%s' \
            % (path, obs.night, int(obs.shot),
               obs.expname, obs.fname) + '.fits'
        if not os.path.exists(fpath):
            missingobs = True
            m_obs.append(obs)
        else:
            with fits.open(fpath, 'readonly') as hdu:
                obs.structaz = hdu[0].header['STRUCTAZ']
                az_vals.append(obs.structaz)

    if missingobs and len(m_obs):  # Replace AZ values for missing fits images
        az_avg = np.average(az_vals)
        for obs in m_obs:
            obs.structaz = az_avg
