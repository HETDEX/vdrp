
      parameter (narrm=3048)
      real xd(narrm,narrm),arr(narrm,narrm),xin(narrm*narrm)
      real xov(narrm)
      integer naxes(2)
      character file1*40,string*80
      logical simple,extend,anyf

 1    call qc1('Image ','improc.def',file1)
      call qi1('Which extension ','improc.def',iext)
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
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)

      call qi2('Overscan x range ','improc.def',ix1,ix2)
      call qi2('Overscan y range ','improc.def',iy1,iy2)
      call qi2('Trim x range ','improc.def',ixt1,ixt2)
      call qi2('Trim y range ','improc.def',iyt1,iyt2)
      call savdef

      n=0
      nov=0
      do j=iy1,iy2
         nov=nov+1
         do i=ix1,ix2
            n=n+1
            xin(n)=xd(i,j)
         enddo
      enddo
      call biwgt(xin,n,xb,xs)
      xov(nov)=xb
      print *,'Overscan subtraction is : ',xb

      ny=0
      do j=iyt1,iyt2
         ny=ny+1
         nx=0
c         xb=xov(ny)
         do i=ixt1,ixt2
            nx=nx+1
            arr(nx,ny)=xd(i,j)-xb
         enddo
      enddo      

      naxes(1)=ixt2-ixt1+1
      naxes(2)=iyt2-iyt1+1

      ier=0
      call ftclos(im1,ier)
      call ftinit(51,'improc.fits',iblock,ier)
      call ftphps(51,-32,naxis,naxes,ier)
c      call ftcopy(im1,50,0,ier)
c      call ftmkyj(50,'NAXIS1',naxes(1),'length of data axis 1',ier)
c      call ftmkyj(50,'NAXIS2',naxes(2),'length of data axis 2',ier)
c      do i=8,1000
c         call ftgrec(im1,i,string,ier)
c         if(ier.ne.0.or.string(1:3).eq.'END') then
c            ier=0
c            goto 807
c         endif
c         call ftprec(50,string,ier)
c      enddo
c      ier=0
c 807  continue
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(51,igc,narrm,naxes(1),naxes(2),arr,ier)
      call ftclos(51,ier)

 706  continue
      end
