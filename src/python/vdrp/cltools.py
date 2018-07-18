
from astropy.table import Table
import  subprocess

def getoff2(fnradec, fnshuffle_ifustars, radius, fnout, ra_offset, dec_offset, logging=None):
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
