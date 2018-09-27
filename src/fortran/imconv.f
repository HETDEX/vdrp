
      parameter (narrm=56000,pi=3.141592e0)
      real xd(narrm,10),xo(narrm,10)
      integer naxes(2)
      character file1*80
      logical simple,extend,anyf

 1    call qc1('Image ','imconv.def',file1)
      call qi1('Which extension ','imconv.def',iext)
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
      ncol=naxes(1)
      naxes(2)=1
      nrow=max(1,naxes(2))
      if(ncol.gt.narrm.or.nrow.gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

 2    call qr2('Gaussian sigma and h4 to convolve ',
     $     'imconv.def',sig,h4)
      call savdef

      den2=2.*sig*sig

      sum1=0.
      sum2=0.
      do j=1,nrow
         do i=1,ncol
            xp=float(i)
            yp=0.
            jcs=nint(xp-5.*sig)
            jce=nint(xp+5.*sig)
            jcs=max(1,jcs)
            jce=min(ncol,jce)
            do jc=jcs,jce
               w=(xp-float(jc))/sig
               gaus=exp(-w*w/2.)/sqrt(den2*pi)
               yp=yp+xd(jc,j)*gaus*(1.+h4*fh4(w))
            enddo
            xo(i,j)=yp
            sum1=sum1+xd(i,j)
            sum2=sum2+xo(i,j)
         enddo
      enddo
      do j=1,nrow
         do i=1,ncol
            xo(i,j)=xo(i,j)*sum1/sum2
         enddo
      enddo

      im1=0
      ier=0
      call ftinit(50,'imconv.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm,naxes(1),naxes(2),xo,ier)
      call ftclos(50,ier)

 706  continue
      end

      function fh4(x)
      fh4=1./sqrt(24.)*(4.*x*x*x*x-12.*x*x+3.)
      return
      end
