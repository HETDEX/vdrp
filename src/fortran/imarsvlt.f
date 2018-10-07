
      parameter (narrm=100)
      real xd(narrm,narrm,2300),xo(narrm,narrm)
      real xb(narrm*narrm)
      integer naxes(4)
      character file1*80,string*80
      logical simple,extend,anyf

 1    call qc1('Image ','imars.def',file1)
      call qi1('Which extension ','imars.def',iext)
c      call qi1('Which array ','imars.def',iarr)
      call savdef

      ier=0
      im1=0
      istatus=0
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

      call qi2('iz range ','imars.def',iz1,iz2)
      call savdef

      do j=1,nrow
         do i=1,ncol
            sum=0.
            n=0
            do iz=iz1,iz2
               sum=sum+xd(i,j,iz)
               n=n+1
               xb(n)=xd(i,j,iz)
            enddo
            call biwgt(xb,n,xb1,xb2)
            xo(i,j)=sum/float(n)
            xo(i,j)=xb1
c            print *,i,j,n,sum/float(n),xb1
         enddo
      enddo
      
      call ftclos(im1,ier)
      ier=0
      naxis=2
      call ftinit(50,'imars.fits',iblock,ier)
c      call ftcopy(im1,50,0,ier)
c      call ftmkyj(50,'NAXIS',2,'Number of axes',ier)
c      call ftdkey(50,'NAXIS3',ier)
      call ftphps(50,-32,naxis,naxes,ier)
      do i=8,1000
         call ftgrec(im1,i,string,ier)
         if(ier.ne.0.or.string(1:3).eq.'END') goto 807
         call ftprec(50,string,ier)
      enddo
 807  ier=0
      continue
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm,naxes(1),naxes(2),xo,ier)
      call ftclos(50,ier)

 706  continue
      end
