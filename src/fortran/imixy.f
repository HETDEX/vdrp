
      parameter (narrm=5048)
      real xd(narrm,narrm)
      integer naxes(2)
      character file1*130,file2*130
      logical simple,extend,anyf

      read *,file1
      file2=file1(1:20)//"_666.ixy"

      im1=0
      ier=0
      iread=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
      igood=0
      if(ier.ne.0) igood=1
      if(igood.eq.1) goto 666
c - this is the wavelength                                                                                           
      iext=7
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      open(unit=11,file='out',status='unknown')
      do i=1,nrow
         if(i.lt.10) write(file2(22:24),1001) i
         if(i.ge.10.and.i.lt.100) write(file2(22:24),1002) i
         if(i.ge.100) write(file2(22:24),1003) i
         write(11,2001) xd(1,i),xd(2,i),file2
      enddo
      close(11)
 666  continue

 1001 format("00",i1)
 1002 format("0",i2)
 1003 format(i3)
 2001 format(2(1x,f7.3),1x,a29)
      end
