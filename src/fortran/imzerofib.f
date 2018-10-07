
      parameter (narrm1=5000,narrm2=5000)
      real xd(narrm1,narrm2),xdrep(narrm1,narrm2)
      real xdfb(narrm1,narrm2)
      integer naxes(2)
      character file1*120,kname*120,kname0*160
      logical simple,extend,anyf

      print *,'Image and Fiber'
      read *,file1,ifib

      im1=0
      ier=0
c      call ftgiou(51,ier)
      iread=0
      call ftopen(51,file1,iread,iblock,ier)
      call ftinit(50,'imrep.fits',iblock,ier)
      if(ier.ne.0) then
         call ftclos(51,ier)
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      do iext=1,18
         ier=0
         call ftmahd(51,iext,ihd,ier)
         call ftghpr(51,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
         if(naxis.eq.1) naxes(2)=1
         if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
            write(*,"('Arrays too small - make narrm bigger')")
            goto 706
         endif
         ncol=naxes(1)
         nrow=naxes(2)
         call ftg2de(51,igc,0.,narrm1,ncol,nrow,xd,anyf,ier)
         call ftcopy(51,50,0,ier)
         if(iext.eq.16) then
            do i=1,ncol
               xd(i,ifib)=0.
            enddo
            call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xd,ier)
         elseif(iext.eq.18) then
            do i=1,ncol
               xd(i,ifib)=0.
            enddo
            call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xd,ier)
         else
            call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xd,ier)
         endif
      enddo
      call ftclos(51,ier)
      call ftclos(50,ier)

 706  continue
      end
