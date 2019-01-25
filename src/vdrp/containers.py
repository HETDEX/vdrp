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
        self.shotname = timestamp
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
            self.ifuslot, self.fname, self.timestamp, self.expname = \
                np.loadtxt(filename, unpack=True,
                           dtype='U50', usecols=[2, 7, 8, 9])

    def where(self, cond):

        res = DithAllFile()
        res.ra = self.ra(np.where(cond))
        res.dec = self.dec(np.where(cond))
        res.ifuslot = self.ifuslot(np.where(cond))
        res.x = self.x(np.where(cond))
        res.y = self.y(np.where(cond))
        res.x_fp = self.x_fp(np.where(cond))
        res.y_fp = self.y_fp(np.where(cond))
        res.filename = self.filename(np.where(cond))
        res.shotname = self.shotname(np.where(cond))
        res.expname = self.expname(np.where(cond))

        return res

    def __getitem__(self, idx):
        return DithAll(self.ra[idx], self.dec[idx], self.x[idx],
                       self.y[idx], self.filename[idx], self.shotname[idx],
                       self.expname[idx])

    def __len__(self):
        return len(self.ra)
