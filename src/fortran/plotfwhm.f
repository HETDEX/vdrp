
      parameter(nmax=10000)
      real x(nmax),y(nmax),y2(nmax)

      ibin=1
      ib1=(ibin-1)/2
      xib=float(ibin)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=-110
      xmax=250.
      ymin=1.4
      ymax=2.5
      call pgsls(1)
      call pgsci(1)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel("Days (01/01/2018=0)","FWHM\Dguider","")

      open(unit=2,file='smline.out',status='old')
      n=0
      do i=1,nmax
         read(2,*,end=667) x1,x2,x3
         n=n+1
         x(n)=x1
         y(n)=x2
         y2(n)=x3
      enddo
 667  continue
      close(2)
      call pgslw(3)
      call pgline(n,x,y2)
      call pgslw(4)
      call pgsci(2)
      call pgline(n,x,y)

      call pgend

      end
