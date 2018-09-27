      parameter(nmax=100000)
      real arr(nmax,3)
      integer naxes(2)
      character comm*132

      open(unit=1,file='asciispec.dat',status='old')
      n=0
      sum=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         arr(n,1)=x2
         arr(n,2)=x2
c         arr(n,3)=x4
         if(i.eq.1) wave0=x1
c         if(i.eq.2) dw=x1-wave0
         if(i.ge.2) sum=sum+x1-x1old
         x1old=x1
      enddo
 666  continue
      close(1)
      dw=sum/float(n-1)

      naxes(1)=n
      naxis=2
      naxes(2)=2
      naxis=1
      naxes(2)=1
      iblock=1
      igc=0
      ier=0

      call ftinit(50,'spec.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
      endif
      call ftp2de(50,igc,10000,naxes(1),naxes(2),arr,ier)
      comm='Wavemin'
      call ftpkye(50,'CRVAL1',wave0,5,comm,ier)
      comm='Delta wave/pix'
      call ftpkye(50,'CDELT1',dw,5,comm,ier)
      call ftclos(50,ier)

      end
