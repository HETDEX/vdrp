
      parameter (narrm1=3100,narrm2=3100)
      real xd(narrm1,narrm2),xderr(narrm1,narrm2),xderr2(narrm1,narrm2)
      real xderr3(narrm1,narrm2),xderr4(narrm1,narrm2)
      integer naxes(2)
      character file1*120,kname*120,kname0*160,filei*30
      logical simple,extend,anyf

      ncol=1036
      nrow=112

      open(unit=1,file='tlist',status='old')
      do i=1,nrow
         read(1,*) filei
         open(unit=2,file=filei,status='old',err=888)
         do j=1,ncol
            read(2,*,end=777,err=778) x1,x2,x3,x4,x5,x6,x7,x8,x9
            xderr(j,i)=x3*1.0e17
            xderr2(j,i)=x9*1.0e17
            xderr3(j,i)=x4
            xderr4(j,i)=x5
            goto 779
 778        continue
            xderr(j,i)=0.
            xderr2(j,i)=0.
            xderr3(j,i)=0.
            xderr4(j,i)=0.
 779        continue
         enddo
 777     continue
         close(2)
         goto 889
 888     continue
         do j=1,ncol
            xderr(j,i)=0.
            xderr2(j,i)=0.
            xderr3(j,i)=0.
            xderr4(j,i)=0.
         enddo
 889     continue
      enddo
      close(1)

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
      do iext=1,18
c         ier=0
         call ftmahd(51,iext,ihd,ier)
         call ftghpr(51,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
         if(naxis.eq.1) naxes(2)=1
         if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
            print *,iext,naxes(1),naxes(2)
            write(*,"('Arrays too small - make narrm bigger')")
            goto 706
         endif
         ncol=naxes(1)
         nrow=naxes(2)
         call ftg2de(51,igc,0.,narrm1,ncol,nrow,xd,anyf,ier)
         call ftcopy(51,50,0,ier)
         call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xd,ier)
      enddo
      ncol=1036
      nrow=112
      naxes(1)=ncol
      naxes(2)=nrow
      call ftiimg(50,-32,2,naxes,ier)
      call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xderr,ier)
      call ftpkls(50,'EXTNAME','calfib',"Label",ier)

      call ftiimg(50,-32,2,naxes,ier)
      call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xderr2,ier)
      call ftpkls(50,'EXTNAME','calfibe',"Label",ier)

      call ftiimg(50,-32,2,naxes,ier)
      call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xderr3,ier)
      call ftpkls(50,'EXTNAME','Amp2Amp',"Label",ier)

      call ftiimg(50,-32,2,naxes,ier)
      call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xderr4,ier)
      call ftpkls(50,'EXTNAME','Throughput',"Label",ier)

      if(ier.ne.0) call ftdelt(50,ier)

      call ftclos(51,ier)
      call ftclos(50,ier)

 706  continue
      end
