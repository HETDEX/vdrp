
      parameter (narrm=10000)
      real xd(narrm,narrm),xin(narrm*narrm)
      integer naxes(2)
      character file1*180
      logical simple,extend,anyf

c      print *,"Image"
      read *,file1
c      print *,"Which extension"
      read *,iext

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
      naxis=2
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxis.eq.1) naxes(2)=1
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

c      print *,"X range"
      read *,ix1,ix2
c      print *,"Y range"
      read *,iy1,iy2

      ymin=1e10
      ymax=-ymin
      nh=0
      do j=iy1,iy2
         do i=ix1,ix2
            ymin=min(ymin,xd(i,j))
            ymax=max(ymax,xd(i,j))
            if(xd(i,j).gt.50000) nh=nh+1
         enddo
      enddo
      open(unit=11,file='out',status='unknown')
      write(11,*) ymin,ymax,nh
      close(11)

 706  continue
      end
