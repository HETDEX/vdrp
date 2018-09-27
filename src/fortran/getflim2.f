
      parameter(nmax=10000)
      real x(nmax),y(nmax)

      open(unit=1,file='in1',status='old')
      open(unit=2,file='in2',status='old')
      open(unit=11,file='out',status='unknown')
      n=0
      do i=1,10000
         read(2,*,end=666) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 666  continue
      close(2)

      do i=1,nmax
         read(1,*,end=667) x1,x2,x3,x4,x5,x6,x7
         call xlinint(x1,n,x,y,y1)
         if(y1.gt.0) then
            x2=x2/y1
            x3=x3/y1
            x4=x4/y1
            x5=x5/y1
            x6=x6/y1
            x7=x7/y1
         endif
         write(11,1101) x1,x2,x3,x4,x5,x6,x7
      enddo
 667  continue
      close(1)
      close(11)
 1101 format(f7.2,6(1x,1pe10.3))
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


