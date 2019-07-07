
      real xin1(100000),xin2(100000),xin3(100000)
      real wave(10000),wv(10000)

      read *,vel

      open(unit=1,file='in',status='old')

      wvoff=vel*8600./3e5
      print *,wvoff
      w1=8460.+wvoff
      w2=8600.+wvoff
      w3=8780.+wvoff
      ws=20.

      nwave=3
      wave(1)=w1
      wave(2)=w2
      wave(3)=w3

      nb1=0
      nb2=0
      nb3=0
      do i=1,100000
         read(1,*,end=666) x1,x2
         if(x1.gt.(w1-ws).and.x1.lt.(w1+ws)) then
            nb1=nb1+1
            xin1(nb1)=x2
         endif
         if(x1.gt.(w2-ws).and.x1.lt.(w2+ws)) then
            nb2=nb2+1
            xin2(nb2)=x2
         endif
         if(x1.gt.(w3-ws).and.x1.lt.(w3+ws)) then
            nb3=nb3+1
            xin3(nb3)=x2
         endif
      enddo
 666  continue
      call biwgt(xin1,nb1,xb1,xs1)
      call biwgt(xin2,nb2,xb2,xs2)
      call biwgt(xin3,nb3,xb3,xs3)
      rewind(1)
      nb=nint(float(nb1)*0.5)
      do i=1,nb
         xin1(i)=xin1(nb1-nb+i)
      enddo
      nb1=nb
      nb=nint(float(nb2)*0.5)
      do i=1,nb
         xin2(i)=xin2(nb2-nb+i)
      enddo
      nb2=nb
      nb=nint(float(nb3)*0.5)
      do i=1,nb
         xin3(i)=xin3(nb3-nb+i)
      enddo
      nb3=nb
      call biwgt(xin1,nb1,xb1,xs1)
      call biwgt(xin2,nb2,xb2,xs2)
      call biwgt(xin3,nb3,xb3,xs3)
      wv(1)=xb1
      wv(2)=xb2
      wv(3)=xb3
      print *,wave(1),wave(2),wave(3)
      print *,xb1,xb2,xb3
c      if(vel.eq.0) then
c         xb=(xb1+xb2+xb3)/3.
c         xb1=xb
c         xb2=xb
c         xb3=xb
c      endif
c      wv(1)=xb1
c      wv(2)=xb2
c      wv(3)=xb3

      open(unit=11,file='out',status='unknown')

      do i=1,100000
         read(1,*,end=667) x1,x2
         call xlinint(x1,nwave,wave,wv,xb)
         write(11,*) x1,x2/xb
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
