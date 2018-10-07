
      parameter(nmax=50000)
      real x(nmax),y(nmax),z(nmax),arr(1000,1000)
      integer naxes(2)

      inum=1

      open(unit=1,file='2dsp.out',status='old')

      do i=1,1000
         do j=1,1000
            arr(i,j)=0.
         enddo
      enddo

      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10
         n=n+1
         arr(n,1)=x2
         arr(n,2)=x3
         arr(n,3)=x4
         arr(n,4)=x5
         arr(n,5)=x6
         arr(n,6)=x7
         arr(n,7)=x8
         arr(n,8)=x9
         arr(n,9)=x10
      enddo
 666  continue
      close(1)

      naxis=2
      naxes(1)=n
      naxes(2)=9
      iblock=1
      igc=0
      ier=0

      call ftinit(50,'2dsp.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
      endif
      print *,naxes(1),naxes(2)
      call ftp2de(50,igc,1000,naxes(1),naxes(2),arr,ier)
      call ftclos(50,ier)

      end
