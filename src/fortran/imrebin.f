
      parameter (narrm=5000)
      real xd(narrm,narrm),xo(narrm,narrm),xw(narrm)
      integer naxes(2)
      character file1*80
      logical simple,extend,anyf

 1    call qc1('Image ','imrebin.def',file1)
      call qi1('Which extension ','imrebin.def',iext)
      call savdef

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 1
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      call qr2('Wavemin and scale of image ','imrebin.def',wave,sc)
      call qr3('Desired Wavemin, Wavemax and scale ','imrebin.def',
     $     waveb,wavef,scd)
      call savdef

      npix=nint(wavef-waveb)/scd+1
      npix=2048

      do i=1,ncol
         xw(i)=wave+float(i-1)*sc
      enddo

      do i=1,npix
         xv=waveb+(i-1)*scd
         call xlinint2(xv,ncol,xw,frac,ip)
         if(xv.lt.xw(1).or.xv.gt.xw(ncol)) then
            do j=1,nrow
               xo(i,j)=-666.
            enddo
         else
            do j=1,nrow
               xo(i,j)=xd(ip,j)+(xd(ip+1,j)-xd(ip,j))*frac
            enddo
         endif
      enddo

      naxes(1)=npix
      naxes(2)=nrow
      im1=0
      ier=0
      call ftinit(50,'imrebin.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm,naxes(1),naxes(2),xo,ier)
      call ftclos(50,ier)

 706  continue
      end

      subroutine xlinint2(xp,n,x,yp,ip)
      real x(n)
      do j=1,n-1
         if(xp.ge.x(j).and.xp.lt.x(j+1)) then
            yp=(xp-x(j))/(x(j+1)-x(j))
            ip=j
            return
         endif
      enddo
      if(xp.lt.x(1)) yp=0.
      if(xp.gt.x(n)) yp=0.
      return
      end
