
from astropy.table import Table
import  subprocess
from pyhetdex.het import fplane
import os

def getoff2(fnradec, fnshuffle_ifustars, radius, ra_offset, dec_offset, logging=None):
    """
    Interface to getoff2.
    """
    GETOFF_CMD = "{}\n"

    # reformat shout.ifustars to shout.ifu as required by "getoff2"
    with open(fnshuffle_ifustars, 'r') as fin:
        ll = fin.readlines()
    with open("shout.ifu", 'w') as fout:
        # legacy code was:
        # rdo_shuffle: grep -v "#" shout.ifustars > $8.ifu
        # runall1: grep 000001 $1v$2.ifu | awk '{print $3,$4,$5,$6,$2}' > shout.ifu # not sure what the grep 000001 does, all lines start with 000001
        for l in ll:
            if not l.startswith("000001"):
                continue
            tt = l.split()
            s = "{} {} {} {} {}\n".format(tt[2], tt[3], tt[4], tt[5], tt[1])
            fout.write(s)

    # reformat add_ra_dec outout to j3 fileformat that getoff2 expects
    t = Table.read(fnradec, format="ascii")
    tmatch_input = Table([t['ra']+ra_offset, t['dec']+dec_offset, t['ifuslot']])
    tmatch_input.write('j3', format='ascii.fast_no_header', overwrite=True)

    # run getoff2
    proc = subprocess.Popen("getoff2", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    s = GETOFF_CMD.format(radius)
    so,se = proc.communicate(input=s)
    for l in so.split("\n"):
        if logging != None:
            logging.info(l)
        else:
            print(l)
    #print so
    p_status = proc.wait()
    with open("getoff2.out") as f:
        l = f.readline()
    tt = l.split()
    new_ra_offset, new_dec_offset = float(tt[0]), float(tt[1])
    return new_ra_offset+ra_offset, new_dec_offset+dec_offset



def immosaicv(prefixes, fplane_file = "fplane.txt", logging=None):
    """
    Interface to immosaicv which creates
    a mosaic give a set of fits files and x y coordinates.
    Requires infp.
    This function will prepare the necessary infp file that is read by immosaicv
    command line tool.
    Format of the latter is 
    20180611T054545_015.fits 015 -450.0 -50.0
    20180611T054545_022.fits 022 -349.743 250.336
    20180611T054545_023.fits 023 -349.798 150.243
    20180611T054545_024.fits 024 -350.0 50.0
    ...
    """
    fp = fplane.FPlane(fplane_file)

    with open('infp', 'w') as infp:
        for f in prefixes:
            ifuslot = f[-3:]
            if not ifuslot in fp.ifuslots:
                msg = "IFU slot {} for file {}.fits not found int {}.".format(ifuslot, f, fplane_file)
                if logging != None:
                    logging.warning("IFU slot {} for file {}.fits not found int {}.".format(ifuslot, f, fplane_file))
                else:
                    print(msg)
                continue

            if not os.path.exists(f + ".fits"):
                msg = "File {}.fits not found.".format(f)
                if logging != None:
                    logging.warning(msg)
                else:
                    print(msg)
                continue

            ifu = fp.by_ifuslot(ifuslot)
            s = "{}.fits {} {} {}\n".format(f, ifuslot, ifu.x, ifu.y)
            infp.write(s)

    # run immosaicv
    proc = subprocess.Popen("immosaicv", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    so,se = proc.communicate()
    for l in so.split("\n"):
        if logging != None:
            logging.info(l)
        else:
            print(l)
    p_status = proc.wait()


def imrot(fitsfile, angle, logging=None):
    """
    Interface to getoff2.
    Rotates fits image by given angle.
    """
    CMD_IMROT="{}\n1\n{}"

    # run imrot
    proc = subprocess.Popen("imrot", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    s = CMD_IMROT.format(fitsfile, angle)
    so,se = proc.communicate(input=s)
    for l in so.split("\n"):
        if logging != None:
            logging.info(l)
        else:
            print(l)

