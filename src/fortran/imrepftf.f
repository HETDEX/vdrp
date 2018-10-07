
      parameter (narrm1=5000,narrm2=5000)
      real xd(narrm1,narrm2),xdrep(narrm1,narrm2)
      real xdfb(narrm1,narrm2)
      integer naxes(2)
      character file1*120,kname*120,kname0*160
      logical simple,extend,anyf

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,"newsky.fits",iread,iblock,ier)
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
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xdrep,anyf,ier)
      call ftclos(im1,ier)
      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,"newfb.fits",iread,iblock,ier)
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
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xdfb,anyf,ier)
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
c         call ftghdn(51,nhdu)
         call ftghpr(51,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
         if(naxis.eq.1) naxes(2)=1
c         print *,naxes(1),naxes(2),ier
         if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
            write(*,"('Arrays too small - make narrm bigger')")
            goto 706
         endif
         ncol=naxes(1)
         nrow=naxes(2)
         call ftg2de(51,igc,0.,narrm1,ncol,nrow,xd,anyf,ier)
c         if(iext.gt.1) then
c            call ftgkys(51,'EXTNAME',kname,file1,ier)
c         else
c            kname='orig'
cc            call ftgkys(51,'RAWFN',kname0,file1,ier)
         call ftcopy(51,50,0,ier)
c         endif
c         call ftiimg(50,-32,naxis,naxes,ier)
         if(iext.eq.14) then
            call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xdfb,ier)
         elseif(iext.eq.16) then
            call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xdrep,ier)
         else
            call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xd,ier)
         endif
c         call ftpkls(50,'EXTNAME',kname,"Label",ier)
cc         call ftpkls(50,'RAWFN',kname0,"Label",ier)
      enddo
      call ftclos(51,ier)
      call ftclos(50,ier)

 706  continue
      end
