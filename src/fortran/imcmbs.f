
      parameter (nmax=3000,narrm=3000)
      real val(nmax,narrm),arr(narrm,narrm),xb(nmax),vala(narrm)
      real tarray(2),xd2(narrm,narrm)
      real*8 bzero,bscale
      integer iext(nmax),naxes(2)
      character file1*120,filea(nmax)*120,comm*120
      logical simple,extend,anyf

c 1    call qc1('Image List ','imcmb.def',file1)
c      call savdef
      read *,file1
      open(unit=1,file=file1,status='old',err=706)

      tim=etime(tarray)
      n=0
      do i=1,nmax
c         read(1,*,end=666) file1,i2
         read(1,*,end=666) file1
         i2=1
         n=n+1
         filea(n)=file1
         iext(n)=i2
c         print *,file1
      enddo
 666  continue
      if(n.eq.0) goto 706

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,filea(1),iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',filea(1)
         goto 706
      endif
      call ftmahd(im1,iext(1),ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      call ftclos(im1,ier)
      if(naxis.eq.1) naxes(2)=1
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      if(nrow.eq.0) nrow=1

      print *
      do irow=1,nrow
c         if((float(irow)/10.-int(irow/10)).eq.0) 
c     $        write(6,"('Doing ',i4,a1,$)") irow,char(13)
c         call flush(6)
         do i=1,n
            call ftfiou(-1,ier)
            call ftgiou(im2,ier)
            call ftopen(im2,filea(i),0,iblock,ier)
c            call ftmahd(im2,iext(i),ihd,ier)
            ip=(irow-1)*ncol+1
            call ftgpve(im2,ig,ip,ncol,0.,vala,anyf,ier)
c            call ftg2de(im2,ig,-666.,narrm,ncol,nrow,xd2,anyf,ier)
            call ftclos(im2,ier)
            do icol=1,ncol
               val(i,icol)=vala(icol)
c               val(i,icol)=xd2(icol,irow)
            enddo
         enddo
         do icol=1,ncol
            nin=0
            xsum=0.
            do i=1,n
c               if(nint(val(i,icol)).ne.-666) then
                  nin=nin+1
c                  xb(nin)=val(i,icol)
                  xsum=xsum+val(i,icol)
c               endif
            enddo
c            call biwgt(xb,nin,xl,xs)
c            arr(icol,irow)=xl
c            arr(icol,irow)=xsum
            arr(icol,irow)=xsum
         enddo
      enddo
      print *

      call ftfiou(-1,ier)
      call ftinit(52,'imcmb.fits',iblock,ier)
      call ftphps(52,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(52,igc,narrm,ncol,nrow,arr,ier)
      call ftclos(52,ier)

 706  continue
      end
