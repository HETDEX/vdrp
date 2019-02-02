import logging
import numpy as np

_logger = logging.getLogger()


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
