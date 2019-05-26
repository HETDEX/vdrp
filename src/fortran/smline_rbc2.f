
      parameter(nmax=20000)
      real x(nmax),y(nmax),xs(nmax),ys(nmax),y3(nmax)
      real xn(nmax),yn(nmax)
      integer idup(nmax)

      wc1=3800.
      wc2=5400.
      open(unit=1,file='tmp',status='old')

      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 666  continue
      close(1)

      ws=3480.
      we=5540
      nw=(we-ws)/2+1
      open(unit=11,file='out',status='unknown')
      do i=1,nw
         w=ws+float(i-1)*2.
         call xlinint(w,n,x,y,yp)
         if(w.lt.wc1) yp=0
         if(w.gt.wc2) yp=0
         write(11,1101) w,yp+1
      enddo
      close(11)
 1101 format(f7.2,1x,f7.3)

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
