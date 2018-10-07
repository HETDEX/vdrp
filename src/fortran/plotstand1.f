
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax),xin(nmax),xfr(100)
      real xb(nmax),yb(nmax),xwa(nmax,100),xfa(nmax,100),yavg(nmax)
      real xw1(nmax),xf1(nmax),xs(nmax),ys(nmax),y3(nmax),xin2(nmax)
      real yin(nmax)
      integer na(100)
      character file1*80,file2*80,c1*18

c      read *,xsmooth
      xsmooth=0.1
      ibin=1
      ib1=(ibin-1)/2
      xib=float(ibin)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      xmin=3500.
      xmax=5500.
      ymin=0.
      ymax=28000.
      ymax=11000.
      call pgsls(1)
      call pgslw(2)
      call pgsci(1)
      call pgsch(1.2)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgsch(1.4)
      call pglabel('Wavelength','Counts','')
      call pgslw(3)
      call pgsch(1.)

      open(unit=1,file='list',status='old')

      nl=0
      ic=0
      do il=1,10000
         read(1,*,end=666) file1
         nl=nl+1
         open(unit=2,file=file1,status='old')
         n=0
         do i=1,nmax
            read(2,*,end=667) x1,x2,x3,x4,x5,x6,x7
            if(x1.gt.4490..and.x1.lt.4540.) goto 777
            if(x2.ne.0) then
               n=n+1
               x(n)=x1
               y(n)=x2
c               y(n)=x6
               xwa(n,il)=x(n)
               xfa(n,il)=y(n)
               na(il)=n
               if(il.eq.1) then
                  xw1(n)=x(n)
                  xf1(n)=y(n)
                  n1=n
               endif
            endif
 777        continue
         enddo
 667     continue
         close(2)
         nbb=0
         do j=1,n,ibin
            nbb=nbb+1
            istart=max(0,j-ib1)
            iend=istart+ibin-1
            if(iend.gt.n) then
               iend=n
               istart=n-ibin+1
            endif
            sum=0.
            nb=0
            do is=istart,iend
               sum=sum+y(is)
               nb=nb+1
               yb(nb)=y(is)
               xb(nb)=x(is)
            enddo
            call biwgt(yb,nb,xbb,xsb)
            yn(nbb)=xbb
            call biwgt(xb,nb,xbb,xsb)
            xn(nbb)=xbb
         enddo
         c1=file1(1:17)
         ic=ic+1
         icp=int(float(ic)/4.)+1
         if(icp.eq.14) ic=1
         if(ic.eq.14) ic=1
         call pgsci(icp)
         call pgsci(ic)
c         call pgline(n,x,y)
         call pgline(nbb,xn,yn)
 866     continue
         close(2)
      enddo
 666  continue
      close(1)
      
      w1=4000.
      w2=5000.
      do il=1,nl
         nb=0
         do i=1,na(il)
            if(xwa(i,il).gt.w1.and.xwa(i,il).lt.w2) then
               nb=nb+1
               xin(nb)=xfa(i,il)
            endif
         enddo
         call biwgt(xin,nb,xb1,xs1)
         xfr(il)=xb1
      enddo            

      ymin=0.8
      ymax=1.2
      call pgsls(1)
      call pgslw(2)
      call pgsci(1)
      call pgsch(1.2)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgsch(1.4)
      call pglabel('Wavelength','Counts','')
      call pgslw(3)
      call pgsch(1.)

      do ii=1,na(1)
         xv=xwa(ii,1)
         do il=1,nl
            n=na(il)
            do i=1,n
               xin(i)=xwa(i,il)
               yin(i)=xfa(i,il)/xfr(il)
            enddo
            call xlinint(xv,n,xin,yin,xff)
            xin2(il)=xff
         enddo
         call biwgt(xin2,nl,xbb,xbs)
         yavg(ii)=xbb
      enddo

      do il=1,nl
         do i=1,na(il)
            xn(i)=xwa(i,il)
c            call xlinint(xn(i),n1,xw1,xf1,xff)
            call xlinint(xn(i),n1,xw1,yavg,xff)
            yval=xfa(i,il)/xfr(il)/xff
c            yn(i)=xfa(i,il)*xfr(1)/xfr(il)/xff
            if(yval.lt.0.9) yval=yn(max(1,i-1))
            if(yval.gt.1.1) yval=yn(max(1,i-1))
            yn(i)=yval
            xs(i)=xn(i)
         enddo
         call pgsci(il)
c         call pgline(na(il),xn,yn)
         call smooth(na(il),xn,yn,na(il),xs,ys,y3,xsmooth)
c         call pgsci(1)
         call pgline(na(il),xs,ys)
         call getslope(na(il),xs,ys,slope)
         do i=1,na(il)
            yn(i)=yn(i)/ys(i)
         enddo
c         call pgline(na(il),xn,yn)
         call biwgt(yn,na(il),xbout,xsout)
         print *,il,xsout,slope
      enddo

      call pgend

      end

      subroutine getslope(n,x,y,slope)
      real x(10000),y(10000)
      w1=3550.
      w2=3650.
      ns=0
      sum=0.
      do i=1,n
         if(x(i).gt.w1.and.x(i).lt.w2) then
            ns=ns+1
            sum=sum+y(i)
         endif
      enddo
      suml=sum/float(ns)
      w1=5350.
      w2=5450.
      ns=0
      sum=0.
      do i=1,n
         if(x(i).gt.w1.and.x(i).lt.w2) then
            ns=ns+1
            sum=sum+y(i)
         endif
      enddo
      sumh=sum/float(ns)
      slope=(sumh-suml)
      return
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

      subroutine smooth(n,x,y,n2,x2,y2,y3,xsmooth)
      parameter(nmax=20000,mm=2,nwk=nmax+6*(nmax*mm+1),mm2=mm*2)
      real x(n),y(n),x2(n2),y2(n2),y3(n)
      real*8 dx(nmax),dy(nmax),wx(nmax),cf(nmax),wk(nwk),val,splder
      real*8 q(mm2)

      if(n.gt.nmax) print *,'make nmax bigger in smooth'

      val=dble(xsmooth)
      md=3
      if(val.eq.0.) md=2
      m=2

      do i=1,n
         dx(i)=dble(x(i))
         dy(i)=dble(y(i))
         wx(i)=1.d0
      enddo

      call gcvspl(dx,dy,nmax,wx,1.d0,m,n,1,md,val,cf,nmax,wk,ier)
      if(ier.ne.0) print *,'ier= ',ier

      do i=1,n2
         in=i
         y2(i)=sngl(splder(0,m,n,dble(x2(i)),dx,cf,in,q))
      enddo

      do i=1,n
         in=i
         y3(i)=sngl(splder(0,m,n,dble(x(i)),dx,cf,in,q))
      enddo

      return
      end
