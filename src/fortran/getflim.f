
      parameter(nmax=1000)
      real x(nmax),y(nmax)

      open(unit=1,file='in',status='old')
      open(unit=2,file='sed.in',status='old')
      open(unit=11,file='out',status='unknown')
      n=0
      do i=1,1000
         read(2,*,end=666) x1,x2
         n=n+1
         x(n)=x1
         y(n)=max(0.01,x2)
      enddo
 666  continue
      close(2)
      y(n)=y(n-1)

      do i=1,nmax
         read(1,*,end=667) x1,x2
         call xlinint(x1,n,x,y,y1)
         flim=(6.626e-27)*(3.e18/x1)/360./5.e5/y1*x2
         write(11,*) x1,x2,flim,flim
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


