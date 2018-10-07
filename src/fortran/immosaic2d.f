
      parameter (narrm=5000)
      real xc1(narrm,narrm),xc2(narrm,narrm),xc3(narrm,narrm)
      real xc4(narrm,narrm),xd(narrm*2,narrm*2)
      integer naxes(2)
      character file1*40

      call qc1('RL ','immosaic.def',file1)
      call savdef
      call geti(file1,ncol,nrow,xc2)
      call qc1('RU ','immosaic.def',file1)
      call savdef
      call geti(file1,ncol,nrow,xc4)

      do i=1,ncol
         do j=1,2064
            xd(i,j)=0.
         enddo
      enddo

      do i=1,ncol
         do j=1,1032
            xd(i,j)=xc4(i,j)
         enddo
      enddo
      do i=1,ncol
         do j=1,1032
            xd(i,j+1032)=xc2(ncol+1-i,1033-j)
         enddo
      enddo

      naxis=2
      naxes(1)=ncol
      naxes(2)=2064
      iblock=1
      igc=1
      ier=0
      call ftinit(50,'immosaic.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm*2,naxes(1),naxes(2),xd,ier)
      call ftclos(50,ier)

 706  continue
      end

      subroutine geti(file1,ncol,nrow,x)
      parameter(narr=5000)
      real x(narr,narr)
      integer naxes(2)
      character file1*40
      logical simple,extend,anyf
      ier=0
      im1=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
c      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narr.or.naxes(2).gt.narr) then
         write(*,"('Arrays too small - make narr bigger')")
         write(*,"('Axes equal to ')") naxes(1),naxes(2)
         goto 706
      endif
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narr,ncol,nrow,x,anyf,ier)
      call ftclos(im1,ier)
      return
 706  continue
      return
      end
