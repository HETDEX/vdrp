#!/usr/bin/env python
""" Fluxlimit routine

Contains python translation of Karl Gebhardt

.. moduleauthor:: Jan Snigula <snigula@mpe.mpg.de>
"""

# import matplotlib

# from matplotlib import pyplot as plt

from argparse import RawDescriptionHelpFormatter as ap_RDHF
from argparse import ArgumentParser as AP

import os
import configParser
import logging
import logging.config
from astropy.io import fits
import shutil
import tempfile
import numpy as np
from astropy.stats import biweight_location

import vdrp.mplog as mplog
import vdrp.programs as vp
import vdrp.extraction as vext
# import vdrp.star_extraction as vstar

from vdrp.mphelpers import mp_run
# from vdrp.vdrp_helpers import VdrpInfo, save_data, read_data, run_command
from vdrp.containers import DithAllFile


_baseDir = os.getcwd()

_logger = logging.getLogger()


def getDefaults():

    defaults = {}

    defaults['tmp_dir'] = '/tmp/'

    defaults['dithall_dir'] = '/work/00115/gebhardt/maverick/detect/'
    defaults['multifits_dir'] = '/work/03946/hetdex/maverick/red1/reductions/'
    defaults['tp_dir'] = '/work/00115/gebhardt/maverick/detect/tp/'
    defaults['norm_dir'] = '/work/00115/gebhardt/maverick/getampnorm/all/'
    defaults['rel_norm_dir'] = '/work/00115/gebhardt/maverick/detect/'
    defaults['fwhm_dir'] = '/work/00115/gebhardt/maverick/detect/'

    defaults['radec_file'] = '/work/00115/gebhardt/maverick/getfib/radec.all'

    defaults['ra_range'] = 70
    defaults['dec_range'] = 70

    defaults['extraction_wl'] = 4505.
    defaults['extraction_wlrange'] = 1035.

    defaults['ifu_search_radius'] = 3.
    defaults['shot_search_radius'] = 600.

    defaults['fitradec_step'] = 0
    defaults['fitradec_nsteps'] = 1
    defaults['fitradec_w_center'] = 4505.
    defaults['fitradec_w_range'] = 3.
    defaults['fitradec_ifit1'] = 1

    defaults['fill'] = 1.
    defaults['sn'] = 6.

    defaults['apcorlim'] = 10000

    defaults['pixsize'] = 2.0

    return defaults


def get_arguments(parser):
    '''
    Add command line arguments for the photometry routines, this function
    can be called from another tool.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    '''

    parser.add_argument("--tmp_dir", type=str, help="Base directory "
                        "used to create the temporary work directory")

    parser.add_argument("--dithall_dir", type=str, help="Base directory "
                        "used to find the dithall.use files")
    parser.add_argument("--multifits_dir", type=str, help="Directory "
                        "with the multi extension fits files")
    parser.add_argument("--tp_dir", type=str, help="Directory "
                        "with the throughput files")
    parser.add_argument("--norm_dir", type=str, help="Directory "
                        "with the amplifier normalization files")

    # Parameters for getnormexp
    parser.add_argument("--rel_norm_dir", type=str, help="Base directory with"
                        " the norm.dat files. These are expected in nightvshot"
                        " directories under this directory.")
    parser.add_argument("--fwhm_dir", type=str, help="Base directory with the"
                        " fwhm.out files. These are expected in nightvshot "
                        " directories under this directory.")

    parser.add_argument("--ra_range", type=int, help="Width in RA"
                        " direction for search grid in asec")
    parser.add_argument("--dec_range", type=int, help="Width in DEC"
                        " direction for search grid in asec")

    parser.add_argument("--extraction_wl", type=float, help="Central "
                        "wavelength for the extraction")
    parser.add_argument("--extraction_wlrange", type=float, help="Wavelength "
                        "range for the extraction")
    parser.add_argument("--radec_file", type=str, help="Filename of file with "
                        "RA DEC PA positions for all shots")

    parser.add_argument("--ifu_search_radius", type=float, help="Radius for "
                        "search for fibers near a given star.")
    parser.add_argument("--shot_search_radius", type=float, help="Radius for "
                        "search for shots near a given star.")

    parser.add_argument("--fitradec_step", type=float, help="Starting step"
                        " for fitradecsp call")
    parser.add_argument("--fitradec_nsteps", type=float, help="Number of steps"
                        " for fitradecsp call")
    parser.add_argument("--fitradec_w_center", type=float, help="Center wl"
                        " for fitradecsp call")
    parser.add_argument("--fitradec_w_range", type=float, help="Wavelength"
                        " range for fitradecsp call")
    parser.add_argument("--fitradec_ifit1", type=float, help="fit flag"
                        " for fitradecsp call")
    parser.add_argument("--fill", type=float, help="Fill value")
    parser.add_argument("--sn", type=float, help="SNR value")

    parser.add_argument("--pixsize", type=float, help="Size of cube pixels"
                        " (arcsec). should result in integer grid size")

    parser.add_argument("--apcorlim", type=float, help="Minimum limit for "
                        "values to be included in average aperture "
                        "correction.")

    return parser


def parseArgs(argv):
    """ Parses configuration file and command line arguments.
    Command line arguments overwrite configuration file settiongs which
    in turn overwrite default values.

    Args:
        args (argparse.Namespace): Return the populated namespace.
    """

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = AP(description=__doc__,  # printed with -h/--help
                     # Don't mess with format of description
                     formatter_class=ap_RDHF,
                     # Turn off help, so we print all options in response to -h
                     add_help=False)
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args(argv)

    defaults = getDefaults()

    config_source = "Default"
    if args.conf_file:
        config_source = args.conf_file
        config = configParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("FluxLim")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h

    # Inherit options from config_paarser
    parser = AP(parents=[conf_parser])

    parser.set_defaults(**defaults)

    parser.add_argument("--logfile", type=str,
                        help="Filename for log file.")

    parser = get_arguments(parser)

    # Boolean paramters
    parser.add_argument("--debug", action='store_true',
                        help="Keep temporary directories")

    # positional arguments
    parser.add_argument('ra', metavar='ra', type=float,
                        help='Right Ascension.')
    parser.add_argument('dec', metavar='dec', type=float,
                        help='Declination.')
    parser.add_argument('night', metavar='night', type=str,
                        help='Night of observation (e.g. 20180611).')
    parser.add_argument('shotid', metavar='shotid', type=str,
                        help='Shotname of observation (e.g. 021).')
    parser.add_argument('fname', metavar='fname', type=str,
                        help='Basename of the multifits file.')

    args = parser.parse_args(remaining_argv)

    args.config_source = config_source
    # should in principle be able to do this with accumulate???
    # args.use_tmp = args.use_tmp == "True"
    # args.remove_tmp = args.remove_tmp == "True"

    # args.fplane_txt = utils.mangle_config_pathname(args.fplane_txt)
    # args.shuffle_cfg = utils.mangle_config_pathname(args.shuffle_cfg)

    args.nightshot = '%sv%s' % (args.night, args.shotid)

    return args


def compute_apcor(apcor_all, apcorlim):
    """
    Filter out edge and other bad regions by selecting
    the apcorlim greatest aperture correction values.
    Then compute the biweight value and return the
    resultant average aperture correction.

    Parameters
    ----------
    apcor_all : numpy:ndarray
        the aperture corrections

    apcorlim : int
        the number of largest
        values to consider
    """

    flattened = sorted(apcor_all.flatten())
    top_vals = np.flip(flattened, 0)[:apcorlim]

    # Check if all elements are identical
    # as this breaks biweight
    if any(((top_vals - top_vals[0])/top_vals[0]) > 1e-10):
        return biweight_location(top_vals)
    else:
        _logger.warning("All aperture correction measurements the same!")
        return top_vals[0]


def calc_fluxlim(args, workdir):
    """
    Equivalent of the rflim0 script and of mklistfl and the rspfl3f scripts.

    Calculate the flux limit for a given night and shot, looping over a
    (by default) 70 x 70 arcsecond grid

    Parameters
    ----------
    args : struct
        The arguments structure
    """

    curdir = tempfile.mkdtemp(prefix='flimtmp', dir=args.tmp_dir)
    cosd = np.cos(args.dec / 57.3)
    rstart = args.ra - args.ra_range/2./3600./cosd
    dstart = args.dec - args.dec_range/2./3600.

    wave_min = args.extraction_wl - args.extraction_wlrange
    wave_max = args.extraction_wl + args.extraction_wlrange
    n_wave = int((wave_max - wave_min) / 2.) + 1

    # Compute gird dimensions from pixel size and range
    nx = int(args.ra_range/args.pixsize)
    ny = int(args.dec_range/args.pixsize)

    _logger.info("Computed required grid size: {:d} by {:d}".format(nx, ny))

    allspec = np.full((n_wave, nx*ny,
                       4), -9999.)
    apcor_all = np.full((n_wave, nx*ny), -9999.)

    dithall_file = args.dithall_dir+'/'+args.night + 'v' \
        + args.shotid+'/dithall.use'

    _logger.info('Reading dithall file %s' % dithall_file)
    try:
        dithall = DithAllFile(dithall_file)

    except Exception as e:
        _logger.warn('Failed to read %s' % dithall_file)
        _logger.exception(e)
        return

    counter = 0
    # This counter tracks the number of extracted spectra
    speccounter = 0

    cosdec = np.cos(args.dec/57.3)

    for r_off in np.arange(0, args.ra_range, args.pixsize):
        ra = rstart + r_off/3600./cosd
        for d_off in np.arange(0, args.dec_range, args.pixsize):
            dec = dstart + d_off/3600.
            counter += 1

            _logger.info('Working at #%d %f %f' % (counter, ra, dec))

            wdir = curdir + '/%s_%d' % (args.nightshot, counter)
            _logger.info('Creating workdir %s' % wdir)
            if not os.path.exists(wdir):
                os.mkdir(wdir)
            # os.chdir(wdir)

            try:
                starobs, _ = vext.get_star_spectrum_data(ra, dec, args,
                                                         (args.night,
                                                          args.shotid),
                                                         False, dithall)

                if not len(starobs):
                    raise Exception('No shots found, skipping!')

                # Call rspstar
                # Get fwhm and relative normalizations
                vp.call_getnormexp(args.nightshot, args.rel_norm_dir,
                                   args.fwhm_dir, wdir)

                specfiles = \
                    vext.extract_star_spectrum(starobs, args,
                                               args.extraction_wl,
                                               args.extraction_wlrange,
                                               wdir)

                vext.get_structaz(starobs, args.multifits_dir)

                vp.run_fitradecsp(ra, dec, args.fitradec_step,
                                  args.fitradec_nsteps, args.fitradec_w_center,
                                  args.fitradec_w_range, args.fitradec_ifit1,
                                  starobs, specfiles, wdir)

                # Now produce the final output

                if not os.path.exists(os.path.join(wdir, 'spec.out')):
                    raise Exception('fitradecsp failed!')

            except Exception as e:
                _logger.error(e.message)
                if not args.debug:
                    _logger.info('Removing workdir %s' % wdir)
                    shutil.rmtree(wdir, ignore_errors=True)
                continue

            specdata = np.loadtxt(os.path.join(wdir, 'spec.out'))

            w = np.where(specdata[:, 8] > 0)[0]

            allspec[:, speccounter, 0] = np.full_like(allspec[:, 0, 0],
                                                      3600. * (ra-args.ra)
                                                      * cosdec)
            allspec[:, speccounter, 1] = np.full_like(allspec[:, 0, 0],
                                                      3600. * (dec-args.dec))
            allspec[:, speccounter, 2] = np.full_like(allspec[:, 0, 0], -9999.)
            allspec[:, speccounter, 3] = np.full_like(allspec[:, 0, 0], -9999.)
            allspec[w, speccounter, 2] = specdata[w, 2]*specdata[w, 7] \
                / (specdata[w, 8] * args.fill) * args.sn
            allspec[w, speccounter, 3] = specdata[w, 4]*specdata[w, 7] \
                / (specdata[w, 8] * args.fill) * args.sn

            apcor_all[:, speccounter] = specdata[:, 8]

            speccounter += 1

            # del starobs
            # del specdata

            if not args.debug:
                _logger.info('Removing workdir %s' % wdir)
                shutil.rmtree(wdir, ignore_errors=True)

    # Now write all the spec files and the list.

    with open(os.path.join(workdir, 'list'), 'w') as f:
        for i in range(0, n_wave):
            wl = int(wave_min + i*2.)

            w = np.where(allspec[i, :, 2] > -9000.)
            if len(w[0]):
                np.savetxt(os.path.join(workdir, 'w%d.j4' % wl),
                           allspec[i, w[0], :], fmt="%.5f %.5f %.3f %.3f")
            else:
                with open(os.path.join(workdir, 'w%d.j4' % wl), 'w') as ff:
                    ff.write('')

            f.write('%s' % 'w%d.j4\n' % wl)

    vp.call_mkimage3d(workdir)

    apcor = compute_apcor(apcor_all, args.apcorlim)

    # wcor = np.where(apcor_all > args.apcorlim)
    # apcor = np.median(apcor_all[wcor])

    update_im3d_header(args, nx, ny, apcor, workdir)

    outname = os.path.join(os.getcwd(),
                           args.nightshot + '_'
                           + args.fname + '.fits')
    if os.path.exists(outname):
        os.remove(outname)
    shutil.move(os.path.join(workdir, 'image3d.fits'), outname)


def update_im3d_header(args, nx, ny, apcor, wdir):
    """
    Add header keywords to the image3d.fits
    """
    with fits.open(os.path.join(wdir, 'image3d.fits'), 'update') as hdu:

        hdu[0].header['OBJECT'] = 'CAT'
        hdu[0].header['CRVAL1'] = args.ra
        hdu[0].header['CRVAL2'] = args.dec
        hdu[0].header['CRVAL3'] = 3470.0
        hdu[0].header['CDELT3'] = 2.0
        hdu[0].header['CTYPE1'] = 'RA---TAN'
        hdu[0].header['CTYPE2'] = 'DEC--TAN'
        hdu[0].header['CTYPE3'] = 'Wave'
        # Image scale in pixels
        hdu[0].header['CD1_1'] = 0.0002777*args.pixsize
        hdu[0].header['CD1_2'] = 0
        hdu[0].header['CD2_2'] = 0.0002777*args.pixsize
        hdu[0].header['CD2_1'] = 0
        hdu[0].header["CD3_3"] = hdu[0].header["CDELT3"]
        hdu[0].header["CD3_1"] = 0.0
        hdu[0].header["CD3_2"] = 0.0
        hdu[0].header["CD2_3"] = 0.0
        hdu[0].header["CD1_3"] = 0.0
        hdu[0].header['CRPIX1'] = nx/2.0
        hdu[0].header['CRPIX2'] = ny/2.0
        hdu[0].header['CRPIX3'] = 1
        hdu[0].header['CUNIT1'] = 'deg'
        hdu[0].header['CUNIT2'] = 'deg'
        hdu[0].header['EQUINOX'] = 2000
        hdu[0].header['APCOR'] = apcor
        hdu[0].header['SNRCUT'] = args.sn

# vdrp_info = None


def main(jobnum, args):
    """
    Main function.
    """
    # global vdrp_info

    # _logger.info("Executing task : {}".format(task))

    # default is to work in results_dir
    # Create a temporary directory
    tmp_dir = os.path.join(os.getcwd(), args.nightshot + '_' + args.fname)
    _logger.info("Tempdir is {}".format(tmp_dir))
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    _logger.info("Copying over prior data (if any)...")
    # dir_util.copy_tree(results_dir, tmp_dir)
    # set working directory to tmp_dir
    wdir = tmp_dir

    _logger.info("Configuration {}.".format(args.config_source))

    # vdrp_info = VdrpInfo.read(wdir)
    # vdrp_info.night = args.night
    # vdrp_info.shotid = args.shotid

    args.wdir = wdir
    args.jobnum = jobnum

    try:
        # os.chdir(wdir)

        _logger.info('Starting flux limit calculation')
        calc_fluxlim(args, wdir)
        _logger.info('Finished flux limit calculation')
    except Exception as e:
        _logger.exception(e)

    finally:
        # vdrp_info.save(wdir)
        if not args.debug:
            _logger.info('Removing workdir %s' % wdir)
            shutil.rmtree(wdir, ignore_errors=True)
        _logger.info("Done.")


def calc_fluxlim_entrypoint():

    # Here we create another external argument parser, this checks if we
    # are supposed to run in multi-threaded mode.

    # First check if we should loop over an input file
    parser = AP(description='Test', formatter_class=ap_RDHF, add_help=False)
    # parser.add_argument('args', nargs=ap_remainder)
    parser.add_argument('-M', '--multi', help='Input filename to loop over.')
    parser.add_argument('--mcores', type=int, default=1,
                        help='Number of paralles process to execute.')
    parser.add_argument('-l', '--logfile', type=str, default='vdrp.log',
                        help='Logfile to write to.')
    parser.add_argument('-L', '--loglevel', type=str, default='INFO',
                        help='Loglevel to use.')

    args, remaining_argv = parser.parse_known_args()

    mplog.setup_mp_logging(args.logfile, args.loglevel)

    # Run (if requested) in threaded mode, this function will call sys.exit
    mp_run(main, args, remaining_argv, parseArgs)


if __name__ == "__main__":
    calc_fluxlim_entrypoint()
