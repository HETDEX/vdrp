import os
import subprocess
import sys

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

ALLSTAR_CMD ="""

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
    def __init__(self, NL, NX, NY, LOWBAD, HIGHBAD, THRESH, AP1, PH_ADU, RNOISE, FRAD, data):
        self.NL      = NL
        self.NX      = NX
        self.NY      = NY
        self.LOWBAD  = LOWBAD
        self.HIGHBAD = HIGHBAD
        self.THRESH  = THRESH
        self.AP1     = AP1
        self.PH_ADU  = PH_ADU
        self.RNOISE  = RNOISE
        self.FRAD    = FRAD
        self.data    = data

    @staticmethod
    def read(als_file):
        with open(als_file) as f:
            ll = f.readlines()
        tt = ll[1].split()
        NL, NX, NY, LOWBAD, HIGHBAD, THRESH, AP1, PH_ADU, RNOISE, FRAD = \
                [int(t) for t in tt[:3]] + [float(t) for t in tt[3:]]
        names = ["ID", "X", "Y", "MAG", "MAG_ERR", "SKY", "NITER", "CHI", "SHARP"]
        dtype = [int,float,float,float,float,float,float,float,float]
        t = Table.read(ll[3:], format='ascii')
        for i,n,d in zip(range(len(t.columns)),names,dtype):
            t.columns[i].name = n
            t.columns[i].dtype = d
        #, format='ascii', names=names, dtype=dtype)
        return DAOPHOT_ALS(NL, NX, NY, LOWBAD, HIGHBAD, THRESH, AP1, PH_ADU, RNOISE, FRAD, t)


def test_input_files_exist(input_files):
    """
    Takes a list of files names and check if they are in place.
    Raises DaophotException if not.
    """
    for f in input_files:
        if not os.path.exists(f):
            raise DaophotException("Input file {} not in place.".format(f))


def rm(ff):
    """
    Takes a list of files names and deletes them.
    """
    for f in ff:
        try:
            os.remove(f)
        except:
            pass


def daophot_find(prefix, sigma):
    """
    Interface to daophot find.
    Replaces second part of rdcoo.
    Requires daophot.opt to be in place.
    """
    global DAOPHOT_FIND_CMD
    input_files = ["daophot.opt", prefix + ".fits"]
    test_input_files_exist(input_files)

    rm([prefix + ".coo",prefix + ".lst",prefix + "jnk.fits"])
    #proc = subprocess.Popen("daophot", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc = subprocess.Popen("daophot", stdin=subprocess.PIPE)
    s = DAOPHOT_FIND_CMD.format(prefix, sigma)
    so,se = proc.communicate(input=s)
    print so
    p_status = proc.wait()
    rm([prefix + "jnk.fits"])

def daophot_phot(prefix):
    """
    Interface to daophot phot.
    Replaces first part of rdsub.
    Requires photo.opt to be in place.
    """
    global DAOPHOT_PHOT_CMD

    input_files = ["daophot.opt", "photo.opt", prefix + ".fits", prefix + ".coo"]
    test_input_files_exist(input_files)

    rm([prefix + ".ap",prefix + "1s.fits",prefix + ".als", prefix + "jnk.fits"])
    #proc = subprocess.Popen("daophot", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc = subprocess.Popen("daophot", stdin=subprocess.PIPE)
    s = DAOPHOT_PHOT_CMD.format(prefix, prefix, prefix)
    so,se = proc.communicate(input=s)
    print so
    p_status = proc.wait()
    rm([prefix + "jnk.fits"])


def allstar(prefix, psf="use.psf"):
    """
    Interface to allstar.
    Replaces second part of rdsub.
    Requires allstar.opt and use.psf, PREFIX.ap to be in place.
    """
    global ALLSTAR_CMD

    input_files = [prefix + ".fits", "allstar.opt", psf, prefix + ".ap"]
    test_input_files_exist(input_files)

    rm([prefix + "s.fits",prefix + ".als", prefix + "jnk.fits"])
    #proc = subprocess.Popen("allstar", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc = subprocess.Popen("allstar", stdin=subprocess.PIPE)
    s = ALLSTAR_CMD.format(prefix, psf, prefix)
    so,se = proc.communicate(input=s)
    print so
    p_status = proc.wait()


def daomaster():
    """
    Interface to daomaster
    replaces "rmaster0".
    Requires 20180611T054545tot.als
    and all.mch to be in place.
    """
    global DAOMASTER_CMD

    # TODO: Shoudl check for existence of e.g. "20180611T054545tot.als"
    input_files = ["all.mch"]
    test_input_files_exist(input_files)

    #rm([prefix + ".raw"])
    #proc = subprocess.Popen("daomaster", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc = subprocess.Popen("daomaster", stdin=subprocess.PIPE)
    s = DAOMASTER_CMD
    so,se = proc.communicate(input=s)
    print so
    p_status = proc.wait()


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
    """
    Read the daophot *.coo output file and rejects detections
    that fall outside xmin - xmax and ymin - ymax.
    Translated from
    awk '{s+=1; if (s<=3||($2>4&&$2<45&&$3>4&&$3<45)) print $0}' $1.coo > $1.lst
    """
    with open(file_in, 'r') as fin:
        ll = fin.readlines()
    with open(file_out, 'w') as fout:
        for i in range(3):
            fout.write(ll[i])
        for l in ll[3:]:
            tt = l.split()
            x,y = float(tt[1]), float(tt[2])
            if x > xmin and y < xmax and y > ymin and y < ymax:
                fout.write(l)

def test():
    prefix = "20180611T054545_034"
    sigma = 2

    daophot(prefix, sigma)
