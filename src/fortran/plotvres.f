
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      real xb(nmax),yb(nmax),xr(nmax),yr(nmax)
      real xbr(nmax),ybr(nmax),xnr(nmax),ynr(nmax)
      character file1*80,file2*80,c1*18

      ibin=13
      ibin=7
      ib1=(ibin-1)/2
      xib=float(ibin)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

c      xmin=3500.
c      xmax=5500.
      xmin=1.
      xmax=224.
c      ymin=-0.03
c      ymax=0.03
      ymin=0.
      ymax=0.18
      ymin=600.
      ymax=1150.
      ymin=-0.4
      ymax=0.4
      ymin=2.0
      ymax=2.85
      call pgsls(1)
      call pgslw(2)
      call pgsci(1)
      call pgsch(1.2)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
c      call pglabel('Wavelength','Residual from mastersky (600 cts)','')
c      call pglabel('Fiber','FtF correction','')
      call pgsch(1.4)
c      call pglabel('Wavelength','Throughput for 50 sq-m','')
c      call pglabel('Wavelength','Resolving Power','')
c      call pglabel('Fiber','Fiber Renormalization','')
c      call pglabel('Fiber','Wavelength Offset','')
      call pglabel('Fiber','Sigma instrumental','')
      call pgslw(3)
      call pgsch(1.)

      open(unit=1,file='list',status='old')

      nl=0
      ic=1
      do il=1,10000
         read(1,*,end=666) file1
         do ia=1,4
         i21=0
         open(unit=21,file=file1(1:17)//"_LL.res",status='old',err=621)
         goto 721
 621     i21=1
 721     continue
         i22=0
         open(unit=22,file=file1(1:17)//"_LU.res",status='old',err=622)
         goto 722
 622     i22=1
 722     continue
         i23=0
         open(unit=23,file=file1(1:17)//"_RL.res",status='old',err=623)
         goto 723
 623     i23=1
 723     continue
         i24=0
         open(unit=24,file=file1(1:17)//"_RU.res",status='old',err=624)
         goto 724
 624     i24=1
 724     continue
         enddo
         n=0
         if(i21.eq.0) then
            do i=1,nmax
               read(21,*,end=667) x1,x2
               n=n+1
               x(n)=x1
               y(n)=x2
            enddo
 667        continue
            close(21)
         endif
         if(i22.eq.0) then
            do i=1,nmax
               read(22,*,end=668) x1,x2
               n=n+1
               x(n)=x1+112
               y(n)=x2
            enddo
 668        continue
            close(22)
         endif
         nr=0
         if(i23.eq.0) then
            do i=1,nmax
               read(23,*,end=669) x1,x2
               nr=nr+1
               xr(nr)=113-x1
               yr(nr)=x2
            enddo
 669        continue
            close(23)
         endif
         if(i24.eq.0) then
            do i=1,nmax
               read(24,*,end=670) x1,x2
               nr=nr+1
               xr(nr)=113-x1+112
               yr(nr)=x2
            enddo
 670        continue
            close(24)
         endif
         call sort2(nr,xr,yr)
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
            sumr=0.
            nb=0
            do is=istart,iend
               sum=sum+y(is)
               sumr=sumr+yr(is)
               nb=nb+1
               yb(nb)=y(is)
               xb(nb)=x(is)
               ybr(nb)=yr(is)
               xbr(nb)=xr(is)
            enddo
            call biwgt(yb,nb,xbb,xsb)
            yn(nbb)=xbb
            call biwgt(xb,nb,xbb,xsb)
            xn(nbb)=xbb
            call biwgt(ybr,nb,xbbr,xsbr)
            ynr(nbb)=xbbr
            call biwgt(xbr,nb,xbbr,xsbr)
            xnr(nbb)=xbbr
         enddo
         c1=file1(1:17)
         ic=ic+1
         if(ic.eq.14) ic=2
         call pgsci(ic)
c         call pgline(n,x,y)
c         call pgline(nr,xr,yr)
         call pgline(nbb,xn,yn)
         call pgline(nbb,xnr,ynr)
 866     continue
         close(2)
      enddo
 666  continue
      close(1)

      call pgend

      end
