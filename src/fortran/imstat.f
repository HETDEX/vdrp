
      parameter (narrm=10000)
      real xd(narrm,narrm),xin(narrm*narrm)
      integer naxes(2)
      character file1*180
      logical simple,extend,anyf

c 1    call qc1('Image ','imstat.def',file1)
c      call qi1('Which extension ','imstat.def',iext)
c      call savdef
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
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
      naxis=2
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxis.eq.1) naxes(2)=1
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

c      print *,"X range"
      read *,ix1,ix2
c      print *,"Y range"
      read *,iy1,iy2

      n=0
      sum=0.
      do j=iy1,iy2
         do i=ix1,ix2
            n=n+1
            xin(n)=xd(i,j)
            sum=sum+xin(n)
         enddo
      enddo
      call biwgt(xin,n,xb,xs)
      open(unit=11,file='imstat.out',status='unknown')
c      print *
c      print *,file1(1:120),n,xb,xs,sum,sum/float(n)
c      print *
      write(11,1101) file1,n,xb,xs,sum
      close(11)
 1101 format(a120,1x,i10,3(1x,1pe13.5))

 706  continue
      end
