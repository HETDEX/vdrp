
      parameter (narrm=14000)
      real xd(narrm,narrm),xin(narrm*narrm)
      integer naxes(2)
      character file1*120
      logical simple,extend,anyf

c 1    call qc1('Image ','imstat.def',file1)
c      call qi1('Which extension ','imstat.def',iext)
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
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
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

c      call qi2('X range ','imstat.def',ix1,ix2)
c      call qi2('Y range ','imstat.def',iy1,iy2)
c      call savdef
      write(*,"('x limits : '$)")
      read *,ix1,ix2
      write(*,"('y limits : '$)")
      read *,iy1,iy2
      write(*,"('Limits : '$)")
      read *,xmin,xmax

      xmin1=xmin
      xmax1=xmax
      xmin2=-10
      xmax2=0
      sum1=0
      sum2=0
      n=0
      do j=iy1,iy2
         do i=ix1,ix2
            n=n+1
            xin(n)=xd(i,j)
            if(xin(n).gt.xmin1.and.xin(n).lt.xmax1) sum1=sum1+1.
            if(xin(n).gt.xmin2.and.xin(n).lt.xmax2) sum2=sum2+1.
         enddo
      enddo
      open(unit=11,file='out',status='unknown')
      write(11,*) sum1,sum2
      close(11)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(2)

      call pghist(n,xin,xmin,xmax,30,0)

 706  continue
      end
