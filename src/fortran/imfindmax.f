
      parameter (narrm1=15000,narrm2=15000)
      real xd(narrm1,narrm2),xin(1000)
      integer naxes(2)
      character file1*180
      logical simple,extend,anyf

c      print *,"Image"
      read *,file1
c      print *,"Which extension"
      read *,iext

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
      ier=0
c      print *,iext,ihd,ier
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxis.eq.1) naxes(2)=1
      if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      open(unit=11,file='out',status='unknown')
      do j=1,nrow
         nin=0
         do i=100,200
            nin=nin+1
            xin(nin)=xd(i,j)
         enddo
         call biwgt(xin,nin,sumb,xs)
         nin=0
         do i=800,900
            nin=nin+1
            xin(nin)=xd(i,j)
         enddo
         call biwgt(xin,nin,sumr,xs)
         if(abs(sumb).gt.999999.) sumb=999999.
         if(abs(sumr).gt.999999.) sumr=999999.
         if(j.lt.10) write(11,1101) j,sumb,sumr
         if(j.ge.10.and.j.lt.100) write(11,1102) j,sumb,sumr
         if(j.ge.100) write(11,1103) j,sumb,sumr
      enddo
      close(11)

 1101 format("00",i1,2(1x,f10.2))
 1102 format("0",i2,2(1x,f10.2))
 1103 format(i3,2(1x,f10.2))
 706  continue
      end
