
      parameter (narrm1=2064,narrm2=2064)
      real xd(narrm1,narrm2),xd2(narrm1,narrm2)
      integer naxes(2)
      character file1*120
      logical simple,extend,anyf

c      write(*,"('First image : '$)")
      read *,file1
      write(*,*) "Doing ",file1
      iext1=1

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im1,iext1,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxis.eq.1) naxes(2)=1
      if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
         print *,naxes(1),naxes(2)
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,-666.,narrm1,ncol,nrow,xd,anyf,ier)

c      write(*,"('Second image : '$)")
      read *,file1
      iext2=1

      im2=0
      ier=0
      call ftgiou(im2,ier)
      iread=0
      call ftopen(im2,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im2,iext2,ihd,ier)
      call ftghpr(im2,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im2,igc,-666.,narrm1,ncol,nrow,xd2,anyf,ier)
      call ftclos(im2,ier)

      do j=1,nrow
         do i=1,ncol
            if(xd2(i,j).lt.0.) xd(i,j)=0.
         enddo
      enddo

c-- open the output file
      call ftclos(im1,ier)
      call ftinit(51,'imrep.fits',iblock,ier)
      call ftphps(51,-32,naxis,naxes,ier)
c      call ftcopy(im1,50,0,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(51,igc,narrm1,naxes(1),naxes(2),xd,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftclos(51,ier)

 706  continue
      end
