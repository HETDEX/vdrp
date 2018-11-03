
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax)
      character file1*80,file2*80,c1*3

c      call pgbegin(0,'?',3,3)
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

c - diff from Jan 1 to Sept 1 is 122 days
      x0=122. 
c      x0=0

      xmin=0.
      xmax=420.-x0
      ymin=0.
      ymax=900.
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Days since Jan 1','Cumulative Number','')

c      open(unit=1,file='in',status='old')
      open(unit=1,file='inall',status='old')
      n=0
      do i=1,100000
         read(1,*,end=667) x1
         n=n+1
         x(n)=x1-x0
c         xcount=float(n)
         xcount=float(n)/9.
         if(x1.le.x0) xn0=xcount
         y(n)=xcount
      enddo
 667  continue
      close(1)
      do i=1,n
         y(i)=y(i)-xn0
      enddo
      call pgslw(4)
      call pgline(n,x,y)

      open(unit=1,file='dex.dat',status='old')
      x0=58130.1421154849
c      x0=58120.1421154849
c      x0=57997.1453929939
      n=0
      do i=1,10000
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
         y(i)=y(i)+80.
      enddo
c      call pgsls(4)
      call pgline(n,x,y)

      call pgend

      end
