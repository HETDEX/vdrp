
      parameter (narrm=5048)
      real xd(narrm,narrm),xa(narrm,narrm)
      integer naxes(2)
      character file1*40
      logical simple,extend,anyf

 1    call qc1('Image ','imars.def',file1)
      call savdef

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         call ftclos(im1,ier)
         write(*,*) 'Error opening image : ',file1
         goto 1
      endif
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxis.eq.1) naxes(2)=1
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)

      do i=1,ncol
         do j=1,nrow
            xa(i,j)=xd(ncol-i+1,j)
         enddo
      enddo

      ier=0
      call ftclos(im1,ier)
      call ftgiou(im1,ier)
      call ftinit(im1,'imflip.fits',iblock,ier)
      call ftphps(im1,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(im1,igc,narrm,naxes(1),naxes(2),xa,ier)
      call ftclos(im1,ier)

 706  continue
      end
