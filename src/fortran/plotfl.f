
      parameter(nmax=10000)
      real xp(nmax),yp(nmax),xp2(nmax),yp2(nmax)
      real avg(nmax),w(nmax),xf(nmax),yf(nmax)

      xmin=3500.
      xmax=5500.
      ymin=3e-17
      ymax=4e-16
      call pgbegin(0,'/null',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.5)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Wavelength','F20','')
         
      open(unit=1,file='in1',status='old')
      n=0
      do i=1,10000
         read(1,*,end=669) x1,x2,x3,x4
         n=n+1
         xp(n)=x1
         yp(n)=x4
      enddo
 669  continue
      close(1)
      nf=10
      do i=1,nf
         xf(i)=xp(1)+(xp(n)-xp(1))*float(i-1)/float(nf-1)
         call xlinint(xf(i),n,xp,yp,yt)
         yf(i)=yt
      enddo
      call pgline(nf,xf,yf)

      open(unit=1,file='in2',status='old')
      n2=0
      do i=1,10000
         read(1,*,end=666) x1,x2
         n2=n2+1
         xp2(n2)=x1
         yp2(n2)=sqrt(max(10.,x2))
      enddo
 666  continue
      close(1)
      do j=1,nf
         if(j.eq.1) then
            xlo=xf(1)
         else
            xlo=(xf(j)+xf(j-1))/2.
         endif
         if(j.eq.nf) then
            xup=xf(nf)
         else
            xup=(xf(j)+xf(j+1))/2.
         endif
         sum=0.
         ns=0
         do i=1,n2
            if(xp2(i).ge.xlo.and.xp2(i).le.xup) then
               sum=sum+yp2(i)
               ns=ns+1
            endif
         enddo
         avg(j)=sum/float(ns)/yf(j)
c         print *,avg(j),ns
      enddo
      do i=1,n2
         call xlinint(xp2(i),nf,xf,avg,ap)
         yp2(i)=yp2(i)/ap
c         print *,i,ap
      enddo

c      wbin=2.
c      ws=xp2(1)
c      we=xp2(n2)
c      nw=(we-ws)/wbin
      wbin=2.
      ws=3500.
      we=5500.
      nw=(we-ws)/wbin
      do i=1,nw
         w(i)=ws+float(i-1)*wbin
      enddo
      open(unit=11,file='out',status='unknown')
      do i=1,nw
         if(i.eq.1) then
            xlo=ws
         else
            xlo=(w(i)+w(i-1))/2.
         endif
         if(i.eq.nw) then
            xup=w(nw)
         else
            xup=(w(i)+w(i+1))/2.
         endif
         sum=0.
         ns=0
         do j=1,n2
            if(xp2(j).ge.xlo.and.xp2(j).le.xup) then
               sum=sum+yp2(j)
               ns=ns+1
            endif
         enddo
         avg(i)=sum
         if(ns.gt.0) avg(i)=sum/float(ns)
         write(11,*) w(i),avg(i)
      enddo
      close(11)
         
c      call pgline(n2,xp2,yp2)
      call pgline(nw,w,avg)

      call pgend
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

