from matplotlib import pyplot as plt
% matplotlib inline
import numpy as np
from astropy.io import ascii
from scipy.optimize import least_squares

ifupos = "ifupos.dat"
exaggeration = 100.


t = ascii.read(ifupos)

# plot fit
f = plt.figure(figsize=[14,7])
ax = plt.subplot(121)
xx,yy = t['X_orig'], t['Y_orig']
dxx,dyy = t['X_orig'] - t['X_measured'], t['Y_orig'] - t['Y_measured']
plt.plot(xx,yy,'.')

for x,y,dx,dy in zip( xx,yy,dxx,dyy):
    plt.arrow(x, y, dx*exaggeration,dy*exaggeration, head_width=0.05, head_length=0.1, fc='k', ec='k', width=1., )

    
plt.xlabel("x [\"]")
plt.ylabel("y [\"]")
plt.axis('equal')
plt.title("Offset x {}".format(exaggeration))


xy_measured = np.vstack( [t['X_orig'].data,t['Y_orig'].data] )
xy_nominal  = np.vstack( [ (t['X_measured']).data,(t['Y_measured']).data] )

    
def peval(a, xy):
    alpha = np.radians(a)
    c, s = np.cos(alpha), np.sin(alpha)
    R = np.array(((c,-s), (s, c)))
    return R.dot(xy)

def resid(a, xy_nominal, xy):
    xy_rot = peval(a, xy_nominal)
    d = ( xy_rot -  xy).T
    res = np.sqrt( d[:,0]**2. + d[:,1]**2. )
    return res

aa = np.arange(-1.,1.,.001)

rr = []
for a in aa:
    res = resid(a, xy_nominal, xy_measured)
    r = sum(res)/len(res)
    rr.append(r)
  
# plot fit
ax = plt.subplot(122)

plt.plot(aa,rr)
plt.ylabel("res")
plt.xlabel("offset angle")

i = np.argmin(rr)
amin = aa[i]
plt.axvline(amin, ls=':')

plt.text(.2,.8,"offset = {:.4} Deg".format(amin), transform=ax.transAxes)

plt.savefig("offset.pdf")
