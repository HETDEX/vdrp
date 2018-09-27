
      parameter(nmax=10000)
      real x(nmax),y(nmax),z(nmax),arr(1000,1000)
      integer naxes(2)

      inum=1

      open(unit=1,file='j4',status='old')

      do i=1,1000
         do j=1,1000
            arr(i,j)=0.
         enddo
      enddo

      dx=0.1
      nx=70
      ny=70
      nh=35
      nx=101
      ny=101
      xnh=50.

      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3
         n=n+1
         x(n)=x1+xnh*dx
         y(n)=x2+xnh*dx
         if(abs(x3).eq.666) x3=0.
         z(n)=x3
      enddo
 666  continue
      close(1)

      naxis=2
      naxes(1)=nx
      naxes(2)=ny
      iblock=1
      igc=0
      ier=0

      do i=1,nx
         xp=float(i-1)*dx
         do j=1,ny
            yp=float(j-1)*dx
            diff=1.e10
            diff=1.5
            do k=1,n
               rad=(xp-x(k))**2+(yp-y(k))**2
               if(rad.lt.diff) then
                  diff=rad
                  arr(i,j)=z(k)
               endif
            enddo
         enddo
      enddo

      call ftinit(50,'image.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
      endif
      print *,naxes(1),naxes(2)
      call ftp2de(50,igc,1000,naxes(1),naxes(2),arr,ier)
      call ftclos(50,ier)

      end
