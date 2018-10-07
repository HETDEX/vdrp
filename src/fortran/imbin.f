
      real xd(27000,2300),xo(27000,2300)
      integer naxes(2)
      character file1*40
      logical simple,extend,anyf

 1    call qc1('Image ','imbin.def',file1)
c      call qi1('Which extension ','imbin.def',iext)
      iext=1
      call savdef

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         call ftclos(im1,ier)
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxis.eq.1) naxes(2)=1
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,27000,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)
      
      call qi2('X and Y binning factor ','imbin.def',nx,ny)
      call savdef

      ii=0
      do i=1,ncol-nx+1,nx
         ii=ii+1
         jj=0
         do j=1,nrow-ny+1,ny
            jj=jj+1
            sum=0.
            do is=i,i+nx-1
               do js=j,j+ny-1
                  sum=sum+xd(is,js)
               enddo
            enddo
            xo(ii,jj)=sum
         enddo
      enddo

      naxes(1)=ii
      naxes(2)=jj

      ier=0
      call ftinit(50,'imbin.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,27000,naxes(1),naxes(2),xo,ier)
      call ftclos(50,ier)

 706  continue
      end
