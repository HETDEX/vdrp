
      parameter (narrm1=4000,narrm2=4000)
      real xd(narrm1,narrm2)
      integer naxes(2)
      character file1*120,file2*120,a1*10,a2*10
      logical simple,extend,anyf

c 1    call qc1('Image ','imars.def',file1)
c      call qi1('Which extension ','imars.def',iext)
c      call savdef
      read *,file1
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

      open(unit=3,file='listbad',status='old',err=766)
      do i=1,10000
         read(3,*,end=766) a1,a2,i3,i4,i5,i6
         do ix=i3,i4
            do ij=i5,i6
               xd(ix,ij)=-1
            enddo
         enddo
      enddo
 766  continue
      close(3)

      rad=2.5
      open(unit=1,file='list',status='old')
      do i=1,1000
         read(1,*,end=666) file1,file2
         open(unit=2,file=file1,status='old')
         open(unit=11,file=file2,status='unknown')
         ng=0
         do j=1,10000
            read(2,*,end=667) x1,x2,x3
            yc=x3
            ys=yc-rad
            ye=yc+rad
            is=max(0,nint(ys))
            ie=min(nrow,nint(ye))
            do ic=is,ie
               if(xd(j,ic).le.0) x2=0
            enddo
            write(11,*) x1,x2
         enddo         
 667     continue
         close(2)
         close(11)
      enddo
 666  continue
      close(1)

 706  continue
      end
