
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      character file1*80,file2*80,c1*18

      call pgbegin(0,'?',4,4)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      xmin=0.
      xmax=113.
      ymin=-0.04
      ymax=0.04

      open(unit=1,file='list',status='old')

      nl=0
      ic=0
      do il=1,10000
         read(1,*,end=666) file1
         open(unit=2,file=file1,status='old')
         n=0
         do i=1,nmax
            read(2,*,end=667) x1,x2,x3
            n=n+1
            x(n)=x1
            y(n)=x2
            yn(n)=x3
         enddo
 667     continue
         close(2)
         c1=file1(1:15)
         ic=ic+1
         icp=int(float(ic)/4.)+1
         if(icp.eq.14) ic=1
c         call pgsci(icp)
         call pgsci(1)
         call pgsls(1)
         call pgslw(2)
         call pgsch(1.5)
         call pgenv(xmin,xmax,ymin,ymax,0,0)
         call pglabel('Fiber','Normalization Difference','')
         call pgsch(2.5)
         call pgmtxt('T',-1.3,0.5,0.5,c1)
c         call pgsch(1.4)
         call pgslw(3)
         call pgsci(1)
         call pgline(n,x,yn)
         call pgsci(2)
         call pgline(n,x,y)
 866     continue
      enddo
 666  continue
      close(1)

      call pgend

      end
