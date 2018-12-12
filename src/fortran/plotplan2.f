
      parameter(nmax=20000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax)
      real xl(2),yl(2)
      character file1*80,file2*80,c1*3

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

c - diff from Jan 1 to Sept 1 is 122 days
c   let's center on Feb 1, 2018, which is mjd=58150.3
      x0=58130.3
      xt1=58211.3-x0
      xt2=58331.2-x0
      xt3=58452.2-x0

      xmin=0.
      xmax=365.
      ymin=0.
      ymax=1200.
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Days since 1 Jan 2018','Cumulative Number','')

      open(unit=2,file='inlist',status='old')
      do ia=1,100
         read(2,*,end=777) file1
         open(unit=1,file=file1,status='old')
         n=0
         do i=1,100000
            read(1,*,end=667) x1
            x1=x1-7
            if(x1.gt.0) then
               n=n+1
               x(n)=x1
               xcount=float(n)
               y(n)=xcount
            endif
         enddo
 667     continue
         close(1)
         call pgslw(4)
         call pgline(n,x,y)
      enddo
 777  continue
      close(2)

      open(unit=1,file='dex.dat',status='old')
      n=0
      do i=1,100000
         read(1,*,end=666) x1,x2
         n=n+1
         x(n)=x1-x0
         if(x1.le.x0) xn0=float(n)
         y(n)=float(n)
      enddo
 666  continue
      close(1)
      do i=1,n
         y(i)=y(i)-xn0
      enddo
      call pgsci(2)
c      call pgline(n,x,y)
      do i=1,n
         y(i)=y(i)+90.
      enddo
c      call pgsls(4)
      call pgline(n,x,y)

      call pgsci(1)
      call pgsls(4)
      yl(1)=ymin
      yl(2)=ymax
      xl(1)=xt1
      xl(2)=xl(1)
      call pgline(2,xl,yl)
      xl(1)=xt2
      xl(2)=xl(1)
      call pgline(2,xl,yl)
      xl(1)=xt3
      xl(2)=xl(1)
      call pgline(2,xl,yl)

      call pgend

      end
