
      real xo(14000,14000),vala(14000)
      integer naxes(2)
      character file1*120
      logical simple,extend,anyf

 1    call qc1('Image ','imsect.def',file1)
      call qi1('Which extension ','imsect.def',iext)
      call savdef

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      iblock=1
      call ftopen(im1,file1,iread,iblock,ier)
      print *,ier
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 1
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))

      call qi2('x range ','imsect.def',ix1,ix2)
      call qi2('y range ','imsect.def',iy1,iy2)
      call savdef

      ix1=max(1,ix1)
      ix2=min(ix2,ncol)
      iy1=max(1,iy1)
      iy2=min(iy2,nrow)

      print *,"Before : ",naxes(1),naxes(2)

      naxes(1)=ix2-ix1+1
      naxes(2)=iy2-iy1+1

      print *,"After  : ",naxes(1),naxes(2)

      do j=iy1,iy2
         iy=j-iy1+1
         ip=(j-1)*ncol+ix1
         call ftgpve(im1,igc,ip,naxes(1),0.,vala,anyf,ier)
         do icol=1,naxes(1)
            xo(icol,iy)=vala(icol)
c            print *,icol,iy,xo(icol,iy)
         enddo
      enddo

      call ftclos(im1,ier)

      im1=0
      ier=0
      call ftinit(50,'imsect.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,14000,naxes(1),naxes(2),xo,ier)
      call ftclos(50,ier)

 706  continue
      end
