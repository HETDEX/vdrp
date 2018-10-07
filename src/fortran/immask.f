
      parameter (narrm=5000)
      real xd(narrm,narrm)
      integer naxes(2)
      character file1*80
      logical simple,extend,anyf

 1    call qc1('Image ','immask.def',file1)
c      call qi1('Which extension ','immask.def',iext)
      iext=1
      call savdef

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
c         goto 1
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)

c      call qi2('x range ','immask.def',ix1,ix2)
c      call qi2('y range ','immask.def',iy1,iy2)
      call qr2('x range ','immask.def',x1,x2)
      call qr2('y range ','immask.def',y1,y2)
      call qr1('Mask Value ','immask.def',val)
      call savdef
      ix1=nint(x1)
      ix2=nint(x2)
      iy1=nint(y1)
      iy2=nint(y2)

      do i=max(ix1,1),min(ix2,ncol)
         do j=max(iy1,1),min(iy2,nrow)
            xd(i,j)=val
         enddo
      enddo

      call ftclos(im1,ier)
      ier=0
      call ftinit(50,'immask.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
c      call ftcopy(im1,50,0,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm,naxes(1),naxes(2),xd,ier)
      call ftclos(50,ier)

 706  continue
      end
