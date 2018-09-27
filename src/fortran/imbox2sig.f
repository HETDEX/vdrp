
      parameter (narrm=5000)
      real xd(narrm,narrm),xd2(narrm,narrm),xin(narrm)
      integer naxes(2)
      character file1*40
      logical simple,extend,anyf

      iflag=0

c 1    call qc1('Image ','imbox.def',file1)
c      write(*,*) "Image"
      read *,file1
      write(*,*) "Doing ",file1
      iext=1
c      call qi2('Box Size x and y (odd) ','imbox.def',ibox,jbox)
c      write(*,*) "Box Size x and y"
      read *,ibox,jbox
c      call qi1('Biweight (1) or Median (0) ','imbox.def',ibi)
c      call qr1('Inner radius to ignore ','imbox.def',rad)
      rad=0.
c      call savdef

      ibi=1

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
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)

      ihalf=(ibox-1)/2
      jhalf=(jbox-1)/2
      do j=1,nrow
c         if((float(j)/10.-int(j/10)).eq.0) 
c     $        write(6,"('Doing ',i4,a1,$)") j,char(13)
c         call flush(6)
         jlo=max(1,j-jhalf)
         jup=min(nrow,j+jhalf)
         do i=1,ncol
            ilo=max(1,i-ihalf)
            iup=min(ncol,i+ihalf)
            nin=0
            do j1=jlo,jup
               do i1=ilo,iup
                  radp=sqrt(float(i1-i)**2+float(j1-j)**2)
c                  if(nint(xd(i1,j1)).ne.iflag
                  if(xd(i1,j1).ne.iflag
     $                 .and.radp.gt.rad) then
                     nin=nin+1
                     xin(nin)=xd(i1,j1)
                  endif
               enddo
            enddo
            if(nin.gt.0) then
               if(ibi.eq.0) then
                  if(nin.gt.1) then
                     call sort(nin,xin)
                     nuse=nint(float(nin)*0.7)
c                     xb=xin(nin/2)
                     xb=xin(nuse)
                  else
                     xb=xin(1)
                  endif
               else
                  call biwgt(xin,nin,xb,xs)
               endif
            else
               xs=0
            endif
            xd2(i,j)=xs
c            if(xd(i,j).eq.iflag) xd2(i,j)=iflag
         enddo
      enddo

      call ftclos(im1,ier)
      ier=0
      call ftinit(51,'imbox.fits',iblock,ier)
      call ftphps(51,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(51,igc,narrm,naxes(1),naxes(2),xd2,ier)
      call ftclos(51,ier)

 706  continue
      end
