
      parameter (narrm1=5000,narrm2=5000)
      real xd(narrm1,narrm2),xderr(narrm1,narrm2)
      integer naxes(2)
      character file1*120,kname*120,kname0*160
      logical simple,extend,anyf

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,"err.fits",iread,iblock,ier)
      if(ier.ne.0) then
         call ftclos(im1,ier)
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      iext=1
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xderr,anyf,ier)
      call ftclos(im1,ier)

      print *,'Image'
      read *,file1

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
      do iext=1,17
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
         call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xd,ier)
      enddo
      call ftiimg(50,-32,2,naxes,ier)
      call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xderr,ier)
      call ftpkls(50,'EXTNAME','error1Dfib',"Label",ier)
      call ftclos(51,ier)
      call ftclos(50,ier)

 706  continue
      end
