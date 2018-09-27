
      parameter(nmax=10000,nmax2=10000000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax),yss(nmax)
      real xb(nmax),yb(nmax),xm(nmax2),ym(nmax2)
      real xsm(nmax),ysm(nmax),yin(nmax),ysma(nmax,3*nmax)
      integer nsym(nmax)
      character file1*80,file2*80,c1*18

      open(unit=1,file='mastersky',status='old')
      nm=0
      do i=1,nmax2
         read(1,*,end=668) x1,x2
         nm=nm+1
         xm(nm)=x1
         ym(nm)=x2
      enddo
 668  continue
      close(1)

      ibin=101
      ib1=(ibin-1)/2
      xib=float(ibin)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=3500.
      xmax=5450.
      xmin=3500.
      xmax=5500.
      ymin=-40.
      ymax=50.

      ns=30
      do i=1,ns
         xsm(i)=xmin+float(i-1)/float(ns-1)*(xmax-xmin)
         nsym(i)=0
      enddo
      call pgsls(1)
      call pgslw(1)
      call pgsci(1)
      call pgsch(1.)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgsch(1.)

      open(unit=1,file='list',status='old')

      nl=0
      ic=0
      do il=1,10000
         read(1,*,end=666) file1
         if(il.le.112) iline=1
         if(il.gt.112.and.il.le.224) iline=2
         if(il.gt.224.and.il.le.336) iline=3
         if(il.gt.336.and.il.le.448) iline=4
         open(unit=2,file=file1,status='old')
         n=0
         do i=1,nmax
            read(2,*,end=667) x1,x2,x3,x4,x5
            call xlinint(x1,nm,xm,ym,ysub)
            n=n+1
            x(n)=x1
            y(n)=x5
            yss(n)=x2-ysub
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
         if(ic.eq.16) ic=1
         call pgsci(ic)
         call pgsls(iline)
c         call pgline(n,x,y)
         call pgline(nbb,xn,yn)
         do ism=1,ns-1
c            do ib=1,nbb
c               if(xn(ib).ge.xsm(ism).and.xn(ib).lt.xsm(ism+1)) then
c                  nsym(ism)=nsym(ism)+1
c                  ysma(ism,nsym(ism))=yn(ib)
c               endif
c            enddo
            do ib=1,n
               if(x(ib).ge.xsm(ism).and.x(ib).lt.xsm(ism+1)) then
                  nsym(ism)=nsym(ism)+1
                  ysma(ism,nsym(ism))=yss(ib)
               endif
            enddo
         enddo
 866     continue
         close(2)
      enddo
 666  continue
      close(1)

      open(unit=11,file='out',status='unknown')
      do i=1,ns-1
         do j=1,nsym(i)
            yin(j)=ysma(i,j)
         enddo
         call biwgt(yin,nsym(i),xbb,xss)
         ysm(i)=xbb
         write(11,*) xsm(i),ysm(i)
      enddo
      close(11)
      call pgsci(2)
      call pgslw(10)
      call pgline(ns-1,xsm,ysm)

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
