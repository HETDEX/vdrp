import argparse


try:
    # Python 3
    from urllib.request import urlopen
    from urllib.error import HTTPError
except ImportError:
    # Python 2
    from urllib2 import urlopen, HTTPError


def get_fplane(filename, datestr='', actpos=False, full=True):
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


def main():
    parser = argparse.ArgumentParser(description='Tool to retrieve fplane file for a given date.')
    parser.add_argument('date', metavar='date', type=str, 
                        help='Datestring of format YYYYMMDD (e.g. 20180611).')

    args = parser.parse_args()

    datestr = args.date
    filename = "fplane_{}.txt".format(datestr)
    get_fplane(filename, datestr='', actpos=False, full=True)
    print("Wrote {}".format(filename))

if __name__ == "__main__":
    main()
