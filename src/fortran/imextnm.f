
      parameter (narrm=5048)
      real xd(narrm,narrm),xr(narrm),xp(narrm)
      integer naxes(2)
      character file1*160
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
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      call qi1('Dispersion along rows (1) or col (2) ',
     $     'imext.def',ispec)
      call qr2('Region ','imext.def',r1,r2)
      ir1=nint(r1)
      ir2=nint(r2)
c      call qi2('Region ','imext.def',ir1,ir2)
      call qi1('Flip it? (1-yes, 0-no) ','imext.def',iflip)
      call qi1('Biweight (0) or Avg (1) ','imext.def',isum)
      call qc1('Output name ','imext.def',file1)
      call savdef

      if(ispec.eq.2) then
         do j=1,nrow
            nr=0
            sum=0.
            do i=ir1,ir2
               if(nint(xd(i,j)).ne.-666.and.
     $              abs(xd(i,j)).gt.-0.001) then
                  nr=nr+1
                  xr(nr)=xd(i,j)
                  sum=sum+xr(nr)
               endif
            enddo
            call biwgt(xr,nr,xb,xs)
            if(nr.gt.0) then
               sum=sum/float(nr)
            else
               sum=-666.
            endif   
            xp(j)=xb
            if(isum.eq.1) xp(j)=sum
         enddo
         num=nrow
      elseif(ispec.eq.1) then
         suma=0.
         nsum=0
         do i=1,ncol
            nr=0
            sum=0.
            sum1=0.
            sum2=0.
            do j=ir1,ir2
c               if(nint(xd(i,j)).ne.-666.and.
c     $              abs(xd(i,j)).gt.0.001) then
               if(nint(xd(i,j)).ne.-666) then
                  nr=nr+1
                  xr(nr)=xd(i,j)
                  sum=sum+xr(nr)
                  sum1=sum1+xd(i,j)*float(j)
                  sum2=sum2+xd(i,j)
               endif
            enddo
            call biwgt(xr,nr,xb,xs)
            if(nr.gt.0) then
               sum=sum/float(nr)
               if(sum2.gt.0) then
                  suma=suma+sum1/sum2
                  nsum=nsum+1
               endif
            else
               sum=-666.
            endif   
            xp(i)=xb
            if(isum.eq.1) xp(i)=sum
         enddo
c         print *,file1(1:10),suma/float(nsum)
        num=ncol
      endif
 
c - flip it or not
 
      do j=1,num
         if(iflip.eq.1) xr(num-j+1)=xp(j)
         if(iflip.eq.0) xr(j)=xp(j)
      enddo

      do i=1,80
         if(file1(i:i).eq.' ') then
            nname=i-1
            goto 966
         endif
      enddo
 966  continue

      naxis=1
      naxes(1)=num
      im1=0
      ier=0
      call ftinit(50,file1(1:nname),iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm,naxes(1),1,xr,ier)
      call ftclos(50,ier)

 706  continue
      end
