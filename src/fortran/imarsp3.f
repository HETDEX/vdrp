
      parameter (narrm1=14000,narrm2=4000)
      real xd(narrm1,narrm2),xd2(narrm1,narrm2),xd3(narrm1,narrm2)
      integer naxes(2)
      character file1*120,file2*120,file3*160
      logical simple,extend,anyf

c 1    call qc1('First image ','imar.def',file1)
c      call qi1('Which extension ','imar.def',iext1)
c      call savdef
      read(*,'(a)') file1
      read(*,'(a)') file2
      read *,ix1,ix2,iy1,iy2
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
      call ftclos(im1,ier)

      im2=0
      ier=0
      call ftgiou(im2,ier)
      iread=0
      call ftopen(im2,file2,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file2
         goto 706
      endif
      call ftmahd(im2,iext1,ihd,ier)
      call ftghpr(im2,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im2,igc,-666.,narrm1,ncol,nrow,xd2,anyf,ier)
      call ftclos(im2,ier)

      do j=iy1,iy2
         do i=ix1,ix2
            xd(i,j)=xd2(i,j)
         enddo
      enddo

c-- open the output file
      call ftinit(50,'imar.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
c      call ftcopy(im1,50,0,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xd,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftclos(50,ier)

 706  continue
      end
