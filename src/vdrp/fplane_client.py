import argparse
import os
import glob
import logging
import shutil
import numpy as np
import re

try:
    # Python 3
    from urllib.request import urlopen
    from urllib.error import HTTPError
except ImportError:
    # Python 2
    from urllib2 import urlopen, HTTPError


def get_fplane(filename, datestr='', actpos=True, full=False):
    """ Obtains fplane file from fplane server at MPE.

    Args:
        filename (str) : Filename that the fplane file should be saved to.
        datestr (str): Datestring of format YYYYMMDD (e.g. 20180611).
    """
    url = 'https://luna.mpe.mpg.de/fplane/' + datestr

    if actpos:
        url += '?actual_pos=1'
    else:
        url += '?actual_pos=0'

    if full:
        url += '&full_fplane=1'
    else:
        url += '&full_fplane=0'

    try:
        resp = urlopen(url)
    except HTTPError as e:
        raise Exception(' Failed to retrieve fplane file, server '
                        'responded with %d %s' % (e.getcode(), e.reason))

    with open(filename, 'w') as f:
        f.write(resp.read().decode())


def retrieve_fplane(night, fplane_txt, wdir):
    """ Saves the fplane file to the target directory
    and names it fplane.txt.

    Args:
    fplane_txt (str) : Either a specific fplane file is specified here, 'DATABASE' is passed,
                       or a file pattern is provided e.g. fplane_YYYYMMDD.txt.
                       In case of the latter a substring of format YYYYMMDD is
                       expected.  The routine will then search
                       for an fplane file of the current date or pick the next
                       older one. E.g. if shot 2080611v017 is to be analysed
                       and fplane files fplane_2080610.txt and
                       fplane_2080615.txt exist, then fplane_2080610.txt will
                       be picked.  In the case of DATABASE the fplane file is
                       retrieved from https://luna.mpe.mpg.de/fplane/.

    """
    global vdrp_info
    target_filename =  os.path.join(wdir, "fplane.txt")
    if fplane_txt == "DATABASE":
        # fplane is retrieved from MPE server
        logging.info("Retrieving fplane file from MPE server. ")
#        get_fplane(target_filename, datestr=night, actpos=False, full=True)
        get_fplane(target_filename, datestr=night, actpos=True, full=False)
    else:
        if not "YYYYMMDD" in fplane_txt:
            # a specific fplane is specified in the config file.
            logging.info("Using {}.".format(fplane_txt))
            shutil.copy2(fplane_txt, os.path.join(wdir, "fplane.txt"))
            return
        else:
            # a fplane file pattern is specified in the config file.
            # find all files that math the pattern
            ff = glob.glob(fplane_txt.replace("YYYYMMDD", "????????"))
            ff = np.array(ff)
            # parse dates
            dd = []
            for f in ff:
                m = re.match("(.*)(20\d{2}\d{2}\d{2})", f)
                d = m.group(2)
                dd.append(int(d))
            dd = np.array(dd)
            ii = np.argsort(dd)
            dd = dd[ii]
            ff = ff[ii]
            jj = dd <= int(night)
            source_filename = ff[jj][-1]
            logging.info("Using {}.".format(source_filename))
            shutil.copy2(source_filename, os.path.join(wdir, "fplane.txt"))


def main():
    parser = argparse.ArgumentParser(description='Tool to retrieve fplane file for a given date.')
    parser.add_argument('date', metavar='date', type=str, 
                        help='Datestring of format YYYYMMDD (e.g. 20180611).')

    args = parser.parse_args()

    datestr = args.date
    filename = "fplane_{}.txt".format(datestr)
    get_fplane(filename, datestr='', actpos=True, full=False)
    print("Wrote {}".format(filename))

if __name__ == "__main__":
    main()
