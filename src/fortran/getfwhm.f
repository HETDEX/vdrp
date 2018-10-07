
      parameter(nmax=10000)
      real x(nmax),y(nmax),xs(nmax),ys(nmax)

      open(unit=1,file='in',status='old')

      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 666  continue
      close(1)

      x1=-10.
      nf=1
      do i=1,n
         if(y(i).ge.x1) then
            x1=y(i)
            nf=nf+1
         else
            goto 667
         endif
      enddo
 667  continue
      nf=nf+4

      ns=100
      xst=x(1)
      xe=x(nf)
      ymax=0.
      do i=1,ns
         xs(i)=xst+(xe-xst)*float(i-1)/float(ns-1)
         call xlinint(xs(i),nf,x,y,ys(i))
         ymax=max(ymax,ys(i))
      enddo

      yhalf=ymax/2.
      ydiff=1e10
      do i=1,ns
         yd=yhalf-ys(i)
         if(yd.ge.0) then
            ihalf1=i
         else
            goto 668
         endif
      enddo
 668  continue
      do i=ns,1,-1
         yd=yhalf-ys(i)
         if(yd.ge.0) then
            ihalf2=i
         else
            goto 669
         endif
      enddo
 669  continue
      xfw=(xs(ihalf2)-xs(ihalf1))/2.35
      open(unit=11,file='out',status='unknown')
      write(11,*) xfw
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

