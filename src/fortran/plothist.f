
      parameter(nmax=15000)
      real x(nmax)
      character file1*80

 1    write(*,"('Data file : '$)")
      read *,file1
      open(unit=1,file=file1,status='old',err=1)
      
      scale=0.22
      scale=1.
      n=0
      xmin=1e30
      xmax=-1e30
      do i=1,nmax
c         read(1,*,end=666) x1,x2,x3
         read(1,*,end=666) x2
         n=n+1
         x(n)=x2*scale
c         print *,n,x(n)
         xmin=min(xmin,x(n))
         xmax=max(xmax,x(n))
      enddo
 666  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(2)

      write(*,"('Limits : '$)")
      read *,xmin,xmax
c      call pgenv(xmin,xmax,0.,250.,0,0)
c      call pgenv(xmin,xmax,0.,1100.,0,0)
c      xmin=0.
c      xmax=4.
c      call pghist(n,x,xmin,xmax,30,0)
c      call pghist(n,x,xmin,xmax,100,0)
c      call pghist(n,x,xmin,xmax,19,1)
      call pghist(n,x,xmin,xmax,19,0)
c      call pglabel('Radial Velocity Uncertainty (km s\\U-1\\D)','','')
c      call pglabel('','Number','')
c      call pglabel('DARKTIME-EXPTIME (sec)','Number','')
      call pglabel('Focal Plane Center Accuracy (arcsec)','Number','')
c      call pglabel('FWHM (arcsec)','Number','')
c      call pglabel('Normalized response at 4940\(2078)','Number','')
c      call pglabel('Sigma in \(2078) for candidate LAEs','Number','')
c      call pglabel('Response at 4940 AA','Number','')
c      call pglabel('sqrt(x\U2\D+y\U2\D)','Number','')

      call pgend

      end
