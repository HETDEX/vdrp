
      parameter(nmax=200000)
      real xp(nmax),xn(nmax)
      character file1*80

      read *,xmin,xmax
      read *,nbin,ymax
      open(unit=1,file="in",status='old')

      offset=0.00
      np=0
      nn=0
      do i=1,nmax
         read(1,*,end=666) x1
         if(x1.gt.0) then
            np=np+1
            xp(np)=x1-offset
         endif
         if(x1.lt.0) then
            nn=nn+1
            xn(nn)=-(x1-offset)
         endif
 667     continue
      enddo
 666  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(2)

c      xmin=3.5
c      xmax=6.5
c      nbin=15
c      ymax=80.
      call pgenv(xmin,xmax,0.,ymax,0,0)
      call pgsci(1)
      call pglabel('Significance','Number','')
      call pgslw(5)
      call pgsci(4)
      call pghist(np,xp,xmin,xmax,nbin,5)
      call pgsci(2)
      call pghist(nn,xn,xmin,xmax,nbin,5)

      call pgend

      end
