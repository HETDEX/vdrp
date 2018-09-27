
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      real xb(nmax),yb(nmax),xp(10),yp(10)
      character file1*80,file2*80,c1*18

      xp(1)=30.
      yp(1)=35.
      xp(2)=175.
      yp(2)=35.
      xp(3)=220.
      yp(3)=42.

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      xmin=0.
      xmax=225.
      ymin=10.
      ymax=45.
      call pgsls(1)
      call pgslw(2)
      call pgsci(1)
      call pgsch(1.5)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Days since 20171001','Units on line','')
      call pgslw(4)
      call pgsch(1.)

      open(unit=1,file='list',status='old')

      nl=0
      ic=0
      do il=1,10000
         read(1,*,end=666) file1
         open(unit=2,file=file1,status='old')
         n=0
         do i=1,nmax
            read(2,*,end=667) x1,x2
            if(x2.ge.-1e10.or.x2.le.1e10) then
               n=n+1
               x(n)=x1
               y(n)=x2
            else
               goto 866
            endif
         enddo
 667     continue
         close(2)
         call pgline(n,x,y)
 866     continue
         close(2)
      enddo
 666  continue
      close(1)

      call pgsci(2)
      call pgsch(2.)
      call pgpt(3,xp,yp,17)
      call pgsch(1.3)
      call pgsci(4)
      call pgptxt(190.,11.,90.,0.,'MUX re-design')

      call pgend

      end
