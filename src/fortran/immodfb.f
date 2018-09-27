
      parameter (narrm1=15000,narrm2=15000)
      real xd(narrm1,narrm2)
      integer naxes(2)
      character file1*180
      logical simple,extend,anyf

c 1    call qc1('Image ','imars.def',file1)
c      call qi1('Which extension ','imars.def',iext)
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
      print *,im1
      if(ier.ne.0) then
         call ftclos(im1,ier)
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
      ier=0
c      print *,iext,ihd,ier
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxis.eq.1) naxes(2)=1
      print *,naxes(1),naxes(2)
      if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xd,anyf,ier)

      print *,"Row ratio at col 1"
      read *,r1b,r1t
      print *,"Row ratio at col 1032"
      read *,r2b,r2t

      do j=1,nrow
         do i=1,ncol
            r1=r1b+float(i-1)/float(ncol-1)*(r2b-r1b)
            r2=r1t+float(i-1)/float(ncol-1)*(r2t-r1t)
            rat=r1+float(j-1)/float(nrow-1)*(r2-r1)
            xd(i,j)=xd(i,j)*rat
         enddo
      enddo

      ier=0
c      call ftgiou(51,ier)
      call ftinit(51,'imars.fits',iblock,ier)
      call ftphps(51,-32,naxis,naxes,ier)
c      call ftcopy(im1,51,0,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(51,igc,narrm1,naxes(1),naxes(2),xd,ier)
      call ftclos(51,ier)
      call ftclos(im1,ier)

 706  continue
      end
