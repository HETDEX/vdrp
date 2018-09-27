
      parameter (narrm=100000)
      real xd(narrm,1)
      integer naxes(2)
      character file1*80
      logical simple,extend,anyf


c 1    call qc1('Image ','ftoa.def',file1)
c      call qi1('Which extension ','ftoa.def',iext)
c      call savdef
      read *,file1
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
c      print *,ier
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      call ftg2de(im1,igc,0.,narrm,ncol,1,xd,anyf,ier)
      call ftclos(im1,ier)

      open(unit=1,file='ftoa.out',status='unknown')
      do i=1,ncol
         write(1,*) i,xd(i,1)
      enddo
      close(1)

 706  continue
      end
