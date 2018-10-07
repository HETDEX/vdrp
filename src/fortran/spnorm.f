
      parameter(nmax=10000)
      real xn(nmax),wn(nmax)

c      n=2
c      wn(1)=3500.
c      wn(2)=5500.
c      xn(1)=1.
c      xn(2)=1.
c      open(unit=1,file='norm.dat',status='old',err=666)
      open(unit=1,file='norm.dat',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         wn(n)=x1
         xn(n)=x2
      enddo
 666  continue
      close(1)

      open(unit=1,file='in',status='old')
      open(unit=11,file='out',status='unknown')
      do i=1,nmax
         read(1,*,end=667) x1,x2
         call xlinint(x1,n,wn,xn,xp)
         write(11,*) x1,x2/xp
      enddo
 667  continue
      close(1)
      close(11)

      end
      
      subroutine xlinint(xp,n,x,y,yp)
      real x(n),y(n)
      do j=1,n-1
         if(xp.ge.x(j).and.xp.lt.x(j+1)) then
            yp=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
            return
         endif
      enddo
      if(xp.lt.x(1)) yp=y(1)
      if(xp.gt.x(n)) yp=y(n)
      return
      end
