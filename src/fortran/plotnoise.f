
      parameter(nmax=100000000)
      real xp(nmax),xn(nmax)
      character file1*80

      open(unit=1,file="signif.dat",status='old')

      wave1=5460
      wave2=5430
      waves=1.3

      w1lo=wave1-waves
      w1hi=wave1+waves
      w2lo=wave2-waves
      w2hi=wave2+waves

      offset=0.0
      np=0
      nn=0
      xmin=1e30
      xmax=-1e30
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,x5,x6,x7
         if(x1.gt.w1lo.and.x1.lt.w1hi) then
c         if(x3.gt.0) then
            np=np+1
            xp(np)=x1-offset
         endif
         if(x1.gt.w2lo.and.x1.lt.w2hi) then
c         if(x3.lt.0) then
            nn=nn+1
            xn(nn)=-(x1-offset)
         endif
 667     continue
      enddo
 666  continue
      close(1)
      print *,nn,np

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(2)

      xmin=2.5
      xmax=6.
      nbin=15
      ymax=100.
      call pgenv(xmin,xmax,0.,ymax,0,0)
      call pgsci(1)
      call pglabel('Noise','Number','')
      call pgslw(5)
      call pgsci(4)
      call pghist(np,xp,xmin,xmax,nbin,5)
      call pgsci(2)
      call pghist(nn,xn,xmin,xmax,nbin,5)

      call pgend

      end
