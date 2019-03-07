import logging
import numpy as np

_logger = logging.getLogger()


class NoShotsException(Exception):
    pass


class DithAll():

    def __init__(self, ra=0., dec=0., ifuslot='', x=0., y=0.,
                 x_fp=0., y_fp=0., filename='', timestamp='',
                 expname=''):

        self.ra = ra
        self.dec = dec
        self.ifuslot = ifuslot
        self.x = x
        self.y = y
        self.x_fp = x_fp
        self.y_fp = y_fp
        self.filename = filename
        self.timestamp = timestamp
        self.expname = expname


class DithAllFile():

    def __init__(self, filename=None):

        self.ra = np.array([], dtype=float)
        self.dec = np.array([], dtype=float)
        self.ifuslot = np.array([], dtype='U50')
        self.x = np.array([], dtype=float)
        self.y = np.array([], dtype=float)
        self.x_fp = np.array([], dtype=float)
        self.y_fp = np.array([], dtype=float)

        self.filename = np.array([], dtype='U50')
        self.timestamp = np.array([], dtype='U50')
        self.expname = np.array([], dtype='U50')

        if filename is not None:
            self.ra, self.dec, self.x, self.y, self.x_fp, self.y_fp = \
                np.loadtxt(filename, unpack=True, usecols=[0, 1, 3, 4, 5, 6])
            self.ifuslot, self.filename, self.timestamp, self.expname = \
                np.loadtxt(filename, unpack=True,
                           dtype='U50', usecols=[2, 7, 8, 9])

    def where(self, cond):

        w = np.where(cond)
        res = DithAllFile()
        res.ra = self.ra[w]
        res.dec = self.dec[w]
        res.ifuslot = self.ifuslot[w]
        res.x = self.x[w]
        res.y = self.y[w]
        res.x_fp = self.x_fp[w]
        res.y_fp = self.y_fp[w]
        res.filename = self.filename[w]
        res.timestamp = self.timestamp[w]
        res.expname = self.expname[w]

        return res

    def __getitem__(self, idx):
        return DithAll(self.ra[idx], self.dec[idx], self.ifuslot[idx],
                       self.x[idx], self.y[idx], self.x_fp[idx],
                       self.y_fp[idx], self.filename[idx],
                       self.timestamp[idx], self.expname[idx])

    def __len__(self):
        return len(self.ra)


class StarObservation():
    """
    Data for one spectrum covering a star observation. This corresponds to the
    data stored in the l1 file with additions from other files

    Attributes
    ----------
    num : int
        Star number
    night : int
        Night of the observation
    shot : int
        Shot of the observation
    ra : float
        Right Ascension of the fiber center
    dec : float
        Declination of the fiber center
    x : float
        Offset of fiber relative to IFU center in x direction
    y : float
        Offset of fiber relative to IFU center in y direction
    full_fname : str
        Filename of the multi extension fits file.
    shotname : str
        NightvShot shot name
    expname : str
        Name of the exposure.
    dist : float
        Distance of the fiber from the star position
    offset_ra : float
        Offset in ra of the fiber from the star position
    offset_dec : float
        Offset in dec of the fiber from the star position
    fname : str
        Basename of the fits filenname
    ifuslot : str
        IFU slot ID
    avg : float
        Average of the spectrum
    avg_norm : float

    avg_error : float
        Error of the average of the spectrum
    structaz : float
        Azimuth of the telescope structure, read from the image header
    """
    def __init__(self, num=0., night=-1, shot=-1, ra=-1, dec=-1, x=-1, y=-1,
                 fname='', shotname='', expname='', offset_ra=-1,
                 offset_dec=1):

        self.num = 0
        self.night = -1.  # l1 - 10
        self.shot = -1.  # l1 - 11
        self.ra = -1.  # l1 - 1
        self.dec = -1.  # l1 - 2
        self.x = -1.  # l1 - 3
        self.y = -1.  # l1 - 4
        self.full_fname = ''  # l1 - 5
        self.shotname = ''  # l1 - 9
        self.expname = ''  # l1 - 6
        self.dist = -1.  # l1 - 7
        self.offset_ra = -1.
        self.offset_dec = -1.
        self.fname = ''
        self.ifuslot = ''

        # l1 - 8 is args.extraction_wl

        self.avg = 0.
        self.avg_norm = 0.
        self.avg_error = 0.

        self.structaz = -1.

    def set_fname(self, fname):
        """
        Split the full filename into the base name and the ifuslot
        """
        self.full_fname = fname
        self.fname, self.ifuslot = self.full_fname.split('.')[0].rsplit('_', 1)


class Spectrum():
    """
    This class encapsulates the content of a tmp*.dat spectrum file

    Attributes
    ----------

    wl : float
        Wavelength
    cnts : float
        Counts of the spectrum
    flx : float
        Flux of the spectrum
    amp_norm : float
        Ampliflier normalization
    tp_norm : float
        Throughput normalization
    ftf_norm : float
        Fiber to fiber normalization
    err_cts : float

    err_cts_local : float

    err_max_flux : float

    """
    def __init__(self):
        self.wl = None
        self.cnts = None
        self.flux = None
        self.amp_norm = None
        self.tp_norm = None
        self.ftf_norm = None
        self.err_cts = None
        self.err_cts_local = None
        self.err_max_flux = None

    def read(self, fname):
        indata = np.loadtxt(fname).transpose()

        self.wl = indata[0]
        self.cnts = indata[1]
        self.flux = indata[2]
        self.amp_norm = indata[3]
        self.tp_norm = indata[4]
        self.ftf_norm = indata[5]
        self.err_cts = indata[6]
        self.err_cts_local = indata[7]
        self.err_max_flux = indata[8]
