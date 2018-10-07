
      parameter(nmax=10000)
      real x(nmax),y(nmax),yl(nmax),yh(nmax)
      real xb(nmax),yb(nmax)
      character file1*80,file2*80,c1*18

      ibin=1
      ib1=(ibin-1)/2
      xib=float(ibin)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      xmin=3550.
      xmax=5300.
c      xmin=1.
c      xmax=112.
c      ymin=-0.03
c      ymax=0.03
      ymin=0.
      ymax=0.18
      ymin=600.
      ymax=1150.
      ymin=4.2
      ymax=6.5
      call pgsls(1)
      call pgslw(2)
      call pgsci(1)
      call pgsch(1.2)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
c      call pglabel('Wavelength','Residual from mastersky (600 cts)','')
c      call pglabel('Fiber','FtF correction','')
      call pgsch(1.4)
c      call pglabel('Wavelength','Throughput for 50 sq-m','')
      call pglabel('Wavelength','FWHM in AA','')
      call pgslw(3)
      call pgsch(1.)

      open(unit=2,file='j3',status='old')

      n=0
      do i=1,nmax
         read(2,*,end=667) x1,x2,x3
         n=n+1
         x(n)=x1
         y(n)=x2
         yl(n)=x2-x3
         yh(n)=x2+x3
      enddo
 667  continue
      close(2)
      call pgline(n,x,y)
      call pgsls(4)
      call pgline(n,x,yl)
      call pgline(n,x,yh)

      call pgend

      end
