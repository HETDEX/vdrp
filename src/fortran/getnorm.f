
      parameter(nmax=10000)
      real xt(nmax),yt(nmax),xa(nmax),ya(nmax)

      open(unit=1,file="in1",status='old')
      open(unit=2,file="in2",status='old')
      open(unit=3,file="in3",status='old')

      nt=0
      do i=1,nmax
         read(2,*,end=666) x1,x2
         nt=nt+1
         xt(nt)=x1
         yt(nt)=x2
      enddo
 666  continue
      close(2)
      na=0
      do i=1,nmax
         read(3,*,end=667) x1,x2
         na=na+1
         xa(na)=x1
         ya(na)=x2
      enddo
 667  continue
      close(3)

      open(unit=11,file="out",status="unknown")
      do i=1,nmax
         read(1,*,end=668) x1,x2
         call xlinint(x1,nt,xt,yt,y1)
         call xlinint(x1,na,xa,ya,y2)
         xnew=x2/y1/y2
c         write(11,1101) x1,xnew,x2
         write(11,*) x1,xnew,x2
      enddo
 668  continue
      close(11)
      close(1)
 1101 format(f7.2,2(1x,f10.2))

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

