
      parameter (narrm=5048)
      real xd(narrm,narrm),wave(narrm),xp(narrm)
      real xsp(narrm),waved(narrm),wtp(narrm),tp(narrm)
      real wa(narrm),fa(narrm),f2f(narrm),wsky(10),sky(narrm)
      real xin(narrm),alp(10),fl(10),xsp2(narrm),yb(narrm)
      integer naxes(2),iwave(narrm)
      character file1*130,file2*130,file3*130
      logical simple,extend,anyf

      ibin=51
      ib1=(ibin-1)/2
      xib=float(ibin)

      wsky0=100
      nsky=4
      wsky(1)=3750
      wsky(2)=4250
      wsky(3)=4750
      wsky(4)=5250
      open(unit=1,file='para.dat',status='old')
      read(1,*) alp(1),fl(1),alp(2),fl(2),alp(3),fl(3),alp(4),fl(4)
      close(1)
      if(fl(1).le.0) then
         alp(1)=-3.31
         fl(1)=375.3
         alp(2)=-3.32
         fl(2)=402.7
         alp(3)=-3.36
         fl(3)=417.8
         alp(4)=-3.40
         fl(4)=431.2
      endif

      convfac=(6.626e-27)*(3.e18)/360./5.e5
      read *,file1
      read *,ifib,w0,ww
      read *,file2
      read *,file3
      wmin=w0-ww
      wmax=w0+ww
      wsp=2.
      nw=nint((wmax-wmin)/wsp)
      do i=1,nw
         wave(i)=wmin+float(i-1)*wsp
      enddo

      open(unit=1,file=file2,status='old')
      ntp=0
      do i=1,narrm
         read(1,*,end=676) x1,x2
         ntp=ntp+1
         wtp(ntp)=x1
         tp(ntp)=x2
      enddo
 676  continue
      close(1)
      open(unit=1,file=file3,status='old',err=678)
      na=0
      do i=1,narrm
         read(1,*,end=677) x1,x2
         na=na+1
         wa(na)=x1
         fa(na)=x2
      enddo
 677  continue
      close(1)
      goto 679
 678  continue
      na=2
      wa(1)=3500.
      fa(1)=1.
      wa(2)=5500.
      fa(2)=1.
      write(*,*) "Amp Norm does not exist: ",file3
 679  continue

      im1=0
      ier=0
      iread=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the wavelength
      iext=12
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      n=0
      wmin=w0-ww
      wmax=w0+ww
      do i=1,ncol
         w=xd(i,ifib)
         if(w.gt.wmin.and.w.lt.wmax) then
            n=n+1
            waved(n)=w
            iwave(n)=i
         endif
      enddo

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the spectrum wth sky in it
      iext=11
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      do i=1,n
         xsp(i)=xd(iwave(i),ifib)
      enddo

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the sky
      iext=17
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      do i=1,n
         xsp2(i)=xd(iwave(i),ifib)
      enddo

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the F2F
      iext=14
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      do i=1,n
         xsp2(i)=xsp2(i)/xd(iwave(i),ifib)
         f2f(i)=xd(iwave(i),ifib)
      enddo

      do j=1,n
         istart=max(0,j-ib1)
         iend=istart+ibin-1
         if(iend.gt.n) then
            iend=n
            istart=n-ibin+1
         endif
         nb=0
         do is=istart,iend
            nb=nb+1
            yb(nb)=xsp2(is)
         enddo
         call biwgt(yb,nb,xbb,xsb)
         sky(j)=xbb
      enddo

      open(unit=11,file='out.sp',status='unknown')
      do i=1,nw
         call xlinint(wave(i),n,waved,xsp,yp)
         call xlinint(wave(i),n,waved,f2f,yf2f)
         call xlinint(wave(i),ntp,wtp,tp,ytp)
         call xlinint(wave(i),na,wa,fa,yfp)
         call xlinint(wave(i),n,waved,sky,ysky)
         call xlinint(wave(i),nsky,wsky,fl,yfl)
         yf2f=max(yf2f,0.)
c         print *,i,yp,ysky,sqrt(yp/ysky)
         if(yfp.gt.0.and.yf2f.gt.0.1.and.ysky.gt.0.and.yf2f.lt.10.) then
            if(yp/ysky.lt.0.3.or.yp/ysky.gt.100.) then
               yp=0.
               yflux=0.
            else
               yp=yfl*sqrt(yp/ysky)
               yflux=yp*convfac/wave(i)/ytp/yfp/yf2f
               yp=1./yp
               yflux=1./yflux               
            endif
            write(11,1101) wave(i),yp,yflux,yfp,ytp,yf2f
         else
            write(11,1101) wave(i),0.,0.,0.,ytp,0.
         endif
      enddo
      close(11)

 1101 format(1x,f8.2,1x,f9.2,1x,1pe11.2,3(1x,0pf7.3))
 706  continue
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
