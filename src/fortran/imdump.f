
      parameter (narrm=14000)
      real xd(narrm,narrm),xin(narrm*narrm)
      integer naxes(2)
      character file1*120
      logical simple,extend,anyf

c 1    call qc1('Image ','imstat.def',file1)
c      call qi1('Which extension ','imstat.def',iext)
c      call savdef
      print *,"Image"
      read *,file1
      print *,"Which extension"
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

c      call qi2('X range ','imstat.def',ix1,ix2)
c      call qi2('Y range ','imstat.def',iy1,iy2)
c      call savdef
      write(*,"('x limits : '$)")
      read *,ix1,ix2
      write(*,"('y limits : '$)")
      read *,iy1,iy2

      open(unit=11,file='out',status='unknown')
      do j=iy1,iy2
         do i=ix1,ix2
            write(11,*) xd(i,j),i,j
         enddo
      enddo
      close(11)

 706  continue
      end
