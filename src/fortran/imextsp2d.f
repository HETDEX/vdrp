
      parameter (narrm=5048)
      real xd(narrm,narrm),wave(narrm),xp(narrm),xderr(narrm,narrm)
      real xsp(narrm),waved(narrm),wtp(narrm),tp(narrm)
      real wa(narrm),fa(narrm),f2f(narrm),spo(narrm,100),spp(narrm)
      integer naxes(2),iwave(narrm)
      character file1*130,file2*130,file3*130
      logical simple,extend,anyf

      nrowe=4
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
      na=3
      wa(1)=3500.
      fa(1)=1.
      wa(2)=4500.
      fa(2)=1.
      wa(3)=5500.
      fa(3)=1.
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
c - this is the trace position
      iext=13
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      nhalf=nint(float(n)/2.)
      xtr=xd(iwave(nhalf),ifib)
      ixtr=nint(xtr)

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
         f2f(i)=xd(iwave(i),ifib)
      enddo

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the sky-subtracted 2d image
      iext=3
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this the error frame
      iext=2
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xderr,anyf,ier)
      call ftclos(im1,ier)

      nrt=nrowe*2+1
      do j=1,nrt
         nr=ixtr-nrowe-1+j
         do i=1,n
            xsp(i)=xd(iwave(i),nr)
            xerr=xderr(iwave(i),nr)
            if(xerr.le.0) xsp(i)=0.
         enddo
         do i=1,nw
            call xlinint(wave(i),n,waved,xsp,yp)
            call xlinint(wave(i),n,waved,f2f,yf2f)
            call xlinint(wave(i),ntp,wtp,tp,ytp)
            call xlinint(wave(i),na,wa,fa,yfp)
            if(yf2f.gt.0) then
               spo(i,j)=yp/yfp/yf2f
            else
               spo(i,j)=0.
            endif
         enddo
      enddo

      open(unit=11,file='out2d.sp',status='unknown')
      do i=1,nw
         do j=1,nrt
            spp(j)=spo(i,j)
         enddo
         write(11,1101) wave(i),(spp(j),j=1,nrt)
      enddo
      close(11)

 1101 format(1x,f7.2,9(1x,f8.2))
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
