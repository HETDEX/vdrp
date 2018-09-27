
      parameter(nmax=10000)
      real x(nmax),y(nmax),xa(nmax),ya(nmax)
      real sky(nmax),wsky(nmax),xin(nmax)
      real skyb(nmax),skys(nmax)
      integer nsa(5000)
      character file1*80,file2*80

      xmin=3490.
      xmax=5510.
      ymin=0.
      ymax=300.

      open(unit=1,file='sky.out',status='old')
      nsky=0
      do i=1,10000
         read(1,*,end=555) x1,x2
         nsky=nsky+1
         wsky(nsky)=x1
         sky(nsky)=x2
      enddo
 555  continue
      close(1)

      open(unit=1,file='list',status='old')

      ia=0
      ns1=0
      do il=1,1000
         read(1,*,end=666) file1,file2
         open(unit=2,file=file1,status='old')
         open(unit=3,file=file2,status='old')
         na=0
         do i=1,10000
            read(3,*,end=667) x1,x2
            na=na+1
            xa(na)=x1
            ya(na)=x2
         enddo
 667     continue
         close(3)
         n=0
         do i=1,10000
            read(2,*,end=668) x1,x2
            n=n+1
            x(n)=x1
            call xlinint(x1,na,xa,ya,yp)
            call xlinint(x1,nsky,wsky,sky,yp2)
            y(n)=x2/yp-yp2
            print *,i,x(n),y(n),x2/yp,yp2
         enddo
 668     continue
         close(2)
         call biwgt(y,n,xb,xs)
         print *,xb,file1
      enddo
 666  continue
      close(1)

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
