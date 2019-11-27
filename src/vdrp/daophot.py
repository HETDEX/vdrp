""" Interface to daophot routines.

Module provides interface to daophot.

.. moduleauthor:: Maximilian Fabricius <mxhf@mpe.mpg.de>
"""
import os
import subprocess
# import sys
from .utils import rm
from hetdex_vdrp_support.tools import bindir

DAOPHOT_FIND_CMD = \
"""att {}
find
1 1

n
{}


y
"""

DAOPHOT_PHOT_CMD =\
"""
att {}
phot


{}.coo
{}.ap
"""

ALLSTAR_CMD = """

{}
{}
{}.ap


"""

DAOMASTER_CMD = \
"""all
2 .1 2
30
2
3
3
3
3
3
3
3
3
3
3
3
3
3
3
2
2
2
2
2
2
2
2
2
2
1
1
1
1
1
1
0
n
n
n
y

y


n
n
n
"""


class DaophotException(Exception):
    pass


from astropy.table import Table


class DAOPHOT_ALS(object):
    """
    Reads DAOPHOT ALS files.
    """
    def __init__(self, NL, NX, NY, LOWBAD, HIGHBAD, THRESH, AP1, PH_ADU,
                 RNOISE, FRAD, data):
        self.NL = NL
        self.NX = NX
        self.NY = NY
        self.LOWBAD = LOWBAD
        self.HIGHBAD = HIGHBAD
        self.THRESH = THRESH
        self.AP1 = AP1
        self.PH_ADU = PH_ADU
        self.RNOISE = RNOISE
        self.FRAD = FRAD
        self.data = data

    @staticmethod
    def read(als_file):
        """Reads daophot als file.

        Notes:
            Creates file `imrot.fits`.

        Args:
            als_file (str): Input file name.

        Returns:
            (DAOPHOT_ALS): Object containing all the infromation
                           in the als file.
        """
        with open(als_file) as f:
            ll = f.readlines()
        tt = ll[1].split()
        NL, NX, NY, LOWBAD, HIGHBAD, THRESH, AP1, PH_ADU, RNOISE, FRAD = \
            [int(t) for t in tt[:3]] + [float(t) for t in tt[3:]]
        names = ["ID", "X", "Y", "MAG", "MAG_ERR", "SKY",
                 "NITER", "CHI", "SHARP"]
        dtype = [int, float, float, float, float, float,
                 float, float, float]
        t = Table.read(ll[3:], format='ascii')
        for i, n, d in zip(list(range(len(t.columns))), names, dtype):
            t.columns[i].name = n
            t.columns[i].dtype = d
        # , format='ascii', names=names, dtype=dtype)
        return DAOPHOT_ALS(NL, NX, NY, LOWBAD, HIGHBAD, THRESH, AP1,
                           PH_ADU, RNOISE, FRAD, t)


def test_input_files_exist(input_files):
    """Takes a list of files names and check if they are in place.
    Raises DaophotException if not.

    Args:
        input_files (list): List of file names to check.
    Raises:
        DaophotException: If not all given files are present.
    """
    for f in input_files:
        if not os.path.exists(f):
            raise DaophotException("Input file {} not in place.".format(f))


def daophot_find(prefix, sigma, logging=None):
    """Interface to daophot find.

    Notes:
        Replaces second part of rdcoo.
        Requires daophot.opt to be in place.

    Args:
        prefix (str): File name prefix to call daophot phot for.
        sigma (float): Daophot phot sigma parameter.
        logging (module): Pass logging module if.
    Otherwise output is passed to the screen.
    """
    global DAOPHOT_FIND_CMD
    input_files = ["daophot.opt", prefix + ".fits"]
    test_input_files_exist(input_files)

    rm([prefix + ".coo", prefix + ".lst", prefix + "jnk.fits"])
    proc = subprocess.Popen(bindir()+"/daophot", stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    s = DAOPHOT_FIND_CMD.format(prefix, sigma)
    so, se = proc.communicate(input=s.encode())
    for l in so.split(b"\n"):
        if logging is not None:
            logging.info(l.decode())
        else:
            print(l.decode())
    # p_status = proc.wait()
    proc.wait()
    rm([prefix + "jnk.fits"])


def daophot_phot(prefix, logging=None):
    """Interface to daophot phot.

    Notes:
        Replaces first part of rdsub.
        Requires photo.opt to be in place.

    Args:
        prefix (str): File name prefix to call daophot phot for.
        logging (module): Pass logging module if.
        Otherwise output is passed to the screen.
    """
    global DAOPHOT_PHOT_CMD

    input_files = ["daophot.opt", "photo.opt", prefix + ".fits",
                   prefix + ".coo"]
    test_input_files_exist(input_files)

    rm([prefix + ".ap", prefix + "1s.fits",
        prefix + ".als", prefix + "jnk.fits"])
    proc = subprocess.Popen(bindir()+"/daophot", stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    s = DAOPHOT_PHOT_CMD.format(prefix, prefix, prefix)
    so, se = proc.communicate(input=s.encode())
    for l in so.split(b"\n"):
        if logging is not None:
            logging.info(l.decode())
        else:
            print(l.decode())
    # p_status = proc.wait()
    proc.wait()
    rm([prefix + "jnk.fits"])


def allstar(prefix, psf="use.psf", logging=None):
    """Interface to allstar.

    Notes:
        Replaces second part of rdsub.
        Requires allstar.opt and use.psf, PREFIX.ap to be in place.

    Args:
        prefix (str): File name prefix to call daophot phot for.
        psf (str): File name for PSF model.
        logging (module): Pass logging module if.
        Otherwise output is passed to the screen.
    """
    global ALLSTAR_CMD

    input_files = [prefix + ".fits", "allstar.opt", psf, prefix + ".ap"]
    test_input_files_exist(input_files)

    rm([prefix + "s.fits", prefix + ".als", prefix + "jnk.fits"])
    proc = subprocess.Popen(bindir()+"/allstar", stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    s = ALLSTAR_CMD.format(prefix, psf, prefix)
    so, se = proc.communicate(input=s.encode())
    for l in so.split(b"\n"):
        if logging is not None:
            logging.info(l.decode())
        else:
            print(l.decode())
    # p_status = proc.wait()
    proc.wait()


def daomaster(logging=None):
    """Interface to daomaster.

    Notes:
        replaces "rmaster0".
        Requires 20180611T054545tot.als
        and all.mch to be in place.

    Args:
        logging (module): Pass logging module if.
        Otherwise output is passed to the screen.
    """
    global DAOMASTER_CMD

    # TODO: Shoull check for existence of als input files listed in
    # all.mch, e.g. "20180611T054545tot.als"
    input_files = ["all.mch"]
    test_input_files_exist(input_files)

    rm(["all.raw"])
    proc = subprocess.Popen(bindir()+"/daomaster", stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    s = DAOMASTER_CMD
    so, se = proc.communicate(input=s.encode())
    for l in so.split(b"\n"):
        if logging is not None:
            logging.info(l.decode())
        else:
            print(l.decode())
    # p_status = proc.wait()
    proc.wait()


def mk_daophot_opt(args):
    s = ""
    s += "VAR = {}\n".format(args.daophot_opt_VAR)
    s += "READ = {}\n".format(args.daophot_opt_READ)
    s += "LOW = {}\n".format(args.daophot_opt_LOW)
    s += "FWHM = {}\n".format(args.daophot_opt_FWHM)
    s += "WATCH = {}\n".format(args.daophot_opt_WATCH)
    s += "PSF = {}\n".format(args.daophot_opt_PSF)
    s += "GAIN = {}\n".format(args.daophot_opt_GAIN)
    s += "HIGH = {}\n".format(args.daophot_opt_HIGH)
    s += "THRESHOLD = {}\n".format(args.daophot_opt_THRESHOLD)
    s += "FIT = {}\n".format(args.daophot_opt_FIT)
    s += "EX = {}\n".format(args.daophot_opt_EX)
    s += "AN = {}\n".format(args.daophot_opt_AN)

    with open("daophot.opt", 'w') as f:
        f.write(s)


def filter_daophot_out(file_in, file_out, xmin, xmax, ymin, ymax):
    """ Filter daophot coo ouput files. For close-to-edge detections.

    Read the daophot *.coo output file and rejects detections
    that fall outside xmin - xmax and ymin - ymax.
    Translated from
    awk '{s+=1; if (s<=3||($2>4&&$2<45&&$3>4&&$3<45)) print $0}' $1.coo > $1.lst

    Notes:
    Args:
        file_in (str): Input file name.
        file_out (str): Output file name.
        xmin (float): Detection x coordinate must be larger than this vaule.
        xmax (float): Detection x coordinate must be smaller than this vaule.
        ymin (float): Detection y coordinate must be larger than this vaule.
        ymax (float): Detection y coordinate must be smaller than this vaule.
    """
    with open(file_in, 'r') as fin:
        ll = fin.readlines()
    with open(file_out, 'w') as fout:
        for i in range(3):
            fout.write(ll[i])
        for l in ll[3:]:
            tt = l.split()
            x, y = float(tt[1]), float(tt[2])
            if x > xmin and x < xmax and y > ymin and y < ymax:
                fout.write(l)
