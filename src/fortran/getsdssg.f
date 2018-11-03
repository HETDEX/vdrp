     
      parameter(nmax=1000)
      real ws(nmax),fs(nmax),w(nmax),f(nmax)

      open(unit=1,file='sdssg.dat',status='old')
      ns=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         ns=ns+1
         ws(ns)=x1
         fs(ns)=x2
      enddo
 666  continue
      close(1)
      open(unit=1,file='s1',status='old')
      n=0
      xnum=0.
      xden=0.
      do i=1,nmax
         read(1,*,end=667) x1,x2
         n=n+1
         w(n)=x1
         call xlinint(x1,ns,ws,fs,f1)
         f(n)=x2
         if(x1.gt.ws(1).and.x1.lt.ws(ns)) then
            xnum=xnum+f1*x2
            xden=xden+f1
         endif
      enddo
 667  continue
      close(1)
      open(unit=11,file='out',status='unknown')
      write(11,*) xnum/xden
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

