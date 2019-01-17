
      parameter(narrm=5000)
      real xo(narrm,narrm),vala(narrm)
      integer naxes(2)
      character file1*120
      logical simple,extend,anyf

      ibox=10
      read *,file1
      read *,icol0,irow0,ifib

c - raw
      iext=1
      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      iblock=1
      call ftopen(im1,file1,iread,iblock,ier)
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))

      ix1=max(1,icol0-ibox)
      ix2=min(icol0+ibox,ncol)
      iy1=max(1,irow0-ibox)
      iy2=min(irow0+ibox,nrow)

      naxes(1)=ix2-ix1+1
      naxes(2)=iy2-iy1+1

      do j=iy1,iy2
         iy=j-iy1+1
         ip=(j-1)*ncol+ix1
         call ftgpve(im1,igc,ip,naxes(1),0.,vala,anyf,ier)
         do icol=1,naxes(1)
            xo(icol,iy)=vala(icol)
         enddo
      enddo
      call ftclos(im1,ier)
      ncoln=naxes(1)+3

c - flagged
      iext=2
      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      iblock=1
      call ftopen(im1,file1,iread,iblock,ier)
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))

      naxes(1)=ix2-ix1+1
      naxes(2)=iy2-iy1+1

      do j=iy1,iy2
         iy=j-iy1+1
         ip=(j-1)*ncol+ix1
         call ftgpve(im1,igc,ip,naxes(1),0.,vala,anyf,ier)
         do icol=1,naxes(1)
            xo(icol+ncoln,iy)=vala(icol)
         enddo
      enddo
      call ftclos(im1,ier)
      ncoln=2*(naxes(1)+3)

c - sky-sub
      iext=3
      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      iblock=1
      call ftopen(im1,file1,iread,iblock,ier)
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))

      naxes(1)=ix2-ix1+1
      naxes(2)=iy2-iy1+1

      do j=iy1,iy2
         iy=j-iy1+1
         ip=(j-1)*ncol+ix1
         call ftgpve(im1,igc,ip,naxes(1),0.,vala,anyf,ier)
         do icol=1,naxes(1)
            xo(icol+ncoln,iy)=vala(icol)
         enddo
      enddo
      call ftclos(im1,ier)
      ncoln=3*(naxes(1)+3)
      naxes(1)=ncoln

      im1=0
      ier=0
      call ftinit(50,'imout.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm,naxes(1),naxes(2),xo,ier)
      call ftclos(50,ier)

 706  continue
      end
