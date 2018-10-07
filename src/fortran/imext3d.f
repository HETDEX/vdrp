
      parameter (narrm=500,narrm3=2060)
      real xd(narrm,narrm,narrm3),xr(narrm3),xp(narrm3)
      integer naxes(3)
      character file1*80
      logical simple,extend,anyf

 1    call qc1('Image ','imext.def',file1)
      call qi1('Which extension ','imext.def',iext)
      call savdef

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 1
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,3,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      num=naxes(3)
      print *,naxis,ncol,nrow,num
      call ftg3de(im1,igc,0.,narrm,narrm,ncol,nrow,num,xd,anyf,ier)
      call ftclos(im1,ier)

c      call qi1('Dispersion along rows (1) or col (2) ',
c     $     'imext.def',ispec)
      ispec=3
      call qi2('x1,x2 ','imext.def',ix1,ix2)
      call qi2('y1,y2 ','imext.def',iy1,iy2)

      iflip=0
c      call qi1('Flip it? (1-yes, 0-no) ','imext.def',iflip)
      call qi1('Biweight (0) or Avg (1) ','imext.def',isum)
      call savdef

      do k=1,num
         nr=0
         sum=0.
         do j=iy1,iy2
            do i=ix1,ix2
               if(nint(xd(i,j,k)).ne.-666.and.
     $              abs(xd(i,j,k)).gt.0.001) then
                  nr=nr+1
                  xr(nr)=xd(i,j,k)
                  sum=sum+xr(nr)
               endif
            enddo
         enddo
         call biwgt(xr,nr,xb,xs)
         if(nr.gt.0) then
c            sum=sum/float(nr)
            sum=sum
         else
            sum=-666.
         endif   
         xp(k)=xb
         if(isum.eq.1) xp(k)=sum
c         print *,k,xp(k)
      enddo

      naxis=1
      naxes(1)=num
      im1=0
      ier=0
      iblock=1
      igc=0
      call ftinit(50,'imars.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm3,naxes(1),1,xp,ier)
      call ftclos(50,ier)

 706  continue
      end
