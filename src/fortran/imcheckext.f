
      parameter (narrm1=3100,narrm2=3100)
      real xd(narrm1,narrm2),xderr(narrm1,narrm2),xderr2(narrm1,narrm2)
      real xderr3(narrm1,narrm2),xderr4(narrm1,narrm2)
      integer naxes(2)
      character file1*120,kname*120,kname0*160,filei*30
      logical simple,extend,anyf

      ncol=1036
      nrow=112

c      print *,'Image'
      read *,file1

      im1=0
      ier=0
c      call ftgiou(51,ier)
      iread=0
      call ftopen(51,file1,iread,iblock,ier)
      if(ier.ne.0) then
         call ftclos(51,ier)
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      isum=0
      do iext=1,19
         call ftmahd(51,iext,ihd,ier)
         call ftghpr(51,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
         if(naxis.eq.1) naxes(2)=1
         ncol=naxes(1)
         nrow=naxes(2)
c         print *,iext,ncol,nrow,ier
         if(ier.ne.0) goto 666
         isum=isum+1
      enddo
 666  continue
      call ftclos(51,ier)
      print *,isum,file1

 706  continue
      end
