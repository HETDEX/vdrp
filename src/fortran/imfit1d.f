
      parameter (narrm=400000,npm=50,nl=narrm)
      real xd(narrm,100),x(narrm),y(narrm),sig(narrm),xo(narrm)
      real a(npm),covar(npm,npm),alpha(npm,npm),xo2(narrm)
      real xl(nl),yl(nl),xr(nl),yr(nl),xt(nl),yt(nl),ylo(nl)
      integer naxes(2),ia(npm)
      character file1*40
      logical simple,extend,anyf
      external funcs

      data big,tol/1.e30,1.e-5/
      data itermax/100/
      nca=npm

      hcut=3.0e10

      ilo=-10000
      ihi=10000
      open(unit=1,file='limits.imfit',err=978,status='old')
      read(1,*) ilo,ihi
 978  close(1)

 1    call qc1('Image ','imfit.def',file1)
      call qi1('Which extension ','imfit.def',iext)
      call savdef

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         print *,ier,im1,iread,iblock
         goto 1
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      call ftg2de(im1,igc,0.,narrm,ncol,1,xd,anyf,ier)
      print *,ncol,ier

      call qi1('Order of poly ','imfit.def',iord)
      call qr1('Lower cutoff ','imfit.def',cutl)
      call qi1('Number for peak values ','imfit.def',nbi)
      call qi1('Linear (0) or Smooth (1) ','imfit.def',itype)
      call savdef

      ymin=big
      ymax=-big
      ilo=max(1,ilo)
      ihi=min(ncol,ihi)
      n0=0
      do i=1,ncol
         y(i)=xd(i,1)
         x(i)=float(i)
         xo(i)=y(i)
         sig(i)=1.
c         print *,i,y(i)
         if(y(i).le.0.1) then
            n0=n0+1
            sig(i)=1000.
         endif
c         if(y(i).ge.5000.) sig(i)=1000.
         if(i.gt.ilo.and.i.lt.ihi) then
            ymin=min(ymin,y(i))
            ymax=max(ymax,y(i))
         else
            sig(i)=1000.
         endif
      enddo
      if(n0.gt.ncol*0.75) then
         print *,"Too Many Zeros: ",n0,ncol
         goto 20
      endif

c      call pgbegin(0,'?',1,1)
c      call pgpap(0.,1.)
c      call pgscf(2)
c      call pgslw(2)
c      call pgsch(1.3)

 10   continue
c 10   call pgenv(x(1),x(ncol),0.,ymax,0,0)
c      call pgline(ncol,x,y)
c      call pglabel('','',file1)

      a(1)=y(ncol/2)
      ia(1)=1
      do i=2,iord
         a(i)=0.
         ia(i)=1
      enddo

      alamda=-1.
      cold=1.e10
      do iter=1,itermax
 
         call mrqmin(x,y,sig,ncol,a,ia,iord,covar,alpha,nca,
     $        chisq,funcs,alamda)
         
         chirel=abs(cold-chisq)/chisq
         if(chirel.lt.tol) goto 666
      enddo
 666  continue
 
      call mrqmin(x,y,sig,ncol,a,ia,iord,covar,alpha,nca,
     $     chisq,funcs,0.)
 
      if(iter.ge.itermax) print *,'Hit max iteration'

c      print *,a(1),a(2)
      do i=1,ncol
c         xl(i)=x(1)+(x(ncol)-x(1))*float(i-1)/float(nl-1)
         xl(i)=x(i)
         yl(i)=0.
         do j=1,iord
            yl(i)=yl(i)+a(j)*xl(i)**(j-1)
         enddo
      enddo

c      call smooth(ncol,x,y,ncol,xl,yl)

c      call pgsci(2)
c      call pgline(ncol,xl,yl)
c      call pgsci(1)

      ymax=0.
      do i=1,ncol
         diff=(y(i)-yl(i))/yl(i)
         xo(i)=y(i)/yl(i)
         ylo(i)=yl(i)
         if(nint(y(i)).eq.-666) xo(i)=-666.
         if(diff.lt.cutl.or.diff.gt.hcut) then
c            sig(i)=10000.
c            call pgpt1(x(i),y(i),17)
         else
            ymax=max(ymax,y(i))
         endif       
         
      enddo

      write(*,"('Go Again (1-yes) : '$)")
      read *,ians
      if(ians.eq.1) goto 10
      if(ians.eq.0) goto 20
      if(ians.eq.-1) goto 30

 30   continue

c      call pgenv(x(1),x(ncol),0.,2.,0,0)
c      call pgline(ncol,xl,xo)

      open(unit=11,file='regions.dat',status='old')
      nr=0
      do i=1,narrm
         read(11,*,end=671) x1,x2
         i1=max(1,nint(x1))
         i2=min(ncol,nint(x2))
         nr=nr+1
         ns=0
         do j=i1,i2
            if(sig(j).lt.1000.) then
               ns=ns+1
               x(ns)=j
               y(ns)=xo(j)
            endif
         enddo
         call sort2(ns,y,x)
         do j=1,ns
            xt(ns-j+1)=x(j)
            yt(ns-j+1)=y(j)
         enddo
         call biwgt(xt,min(ns,nbi),xb,xs)
         xr(nr)=xb
         call biwgt(yt,min(ns,nbi),xb,xs)
         yr(nr)=xb
         if(ns.eq.0) then
            xr(nr)=float(i2+i1)/2.
            yr(nr)=1.
         endif
c         print *,xr(nr),yr(nr)
      enddo
 671  continue
      close(11)
c      call pgsch(2.0)
c      call pgpoint(nr,xr,yr,17)
c      call pgsch(1.3)
      if(itype.eq.1) then
         call smooth(nr,xr,yr,ncol,xl,yl)
      else
         do i=1,ncol
            xp=xl(i)
            do j=1,nr-1
               if(xp.ge.xr(j).and.xp.lt.xr(j+1)) then
                  yp=yr(j)+(yr(j+1)-yr(j))*(xp-xr(j))/(xr(j+1)-xr(j))
               endif
            enddo
            if(xp.lt.xr(1)) yp=yr(1)
            if(xp.gt.xr(nr)) yp=yr(nr)
            yl(i)=yp
         enddo
      endif
c      call pgsci(2)
c      call pgline(ncol,xl,yl)
c      call pgsci(1)
c      call pglabel('','',file1)

      do i=1,ncol
         xo2(i)=0.
         if(nint(xo(i)).ne.-666) then
            if(yl(i).gt.0) then
               xo(i)=xo(i)/yl(i)
               xo2(i)=yl(i)*ylo(i)
            else
               xo(i)=0.
               xo2(i)=0.
            endif
         endif
      enddo

 20   continue
c      call pgend

      call ftclos(im1,ier)
      ier=0
      call ftinit(50,'imfit.fits',iblock,ier)
c      call ftcopy(im1,50,0,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         print *,'Try deleting imfit.fits first'
         goto 706
      endif
      call ftp2de(50,igc,narrm,naxes(1),1,xo,ier)
      call ftclos(50,ier)
      ier=0

 706  continue
      end

      subroutine funcs(x,a,y,dyda,na)
      real a(na),dyda(na)
      y=0.
      do i=1,na
         y=y+a(i)*x**(i-1)
         dyda(i)=x**(i-1)
      enddo
      return
      end
