
      parameter(nmax=10000)
      real w(nmax),x(nmax),y(nmax)

      ws=3480.
      we=5515.
      wd=2.0
      nw=nint((we-ws)/wd)+1
      do i=1,nw
         w(i)=ws+wd*float(i-1)
      enddo

      open(unit=1,file="spec.in",status='old')
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 666  continue
      close(1)
      open(unit=11,file='spec.out',status='unknown')
      do i=1,nw
         call xlinint(w(i),n,x,y,yp)
         write(11,*) w(i),yp
      enddo
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
