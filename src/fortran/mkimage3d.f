
      parameter(nmax=10000)
      real x(nmax),y(nmax),z(nmax),arr(100,100,1000)
      integer naxes(3)
      character file1*40
      parameter(pi=3.14159)

      inum=1
      rfw=1.8
      rsig=rfw/2.35
      w=1.0/rsig
      gaus0=exp(-w*w/2.)

      do i=1,100
         do j=1,100
            do k=1,1000
               arr(i,j,k)=0.
            enddo
         enddo
      enddo
      dx=1.0
      nx=70
      ny=70
      nh=35
      ntot=0

      open(unit=2,file='list',status='old')
      do iz=1,1000
         read(2,*,end=667) file1
         ntot=ntot+1
         
         open(unit=1,file=file1,status='old')

         n=0
         do i=1,nmax
            read(1,*,end=666) x1,x2,x3
            n=n+1
            x(n)=x1+nh
            y(n)=x2+nh
            if(abs(x3).eq.666) x3=0.
            z(n)=x3
         enddo
 666     continue
         close(1)
         
         do i=1,nx
c            xp=float(i-1)*dx
            xp=float(i)*dx
            do j=1,ny
c               yp=float(j-1)*dx
               yp=float(j)*dx
               diff=3.0
               do k=1,n
                  rad=sqrt((xp-x(k))**2+(yp-y(k))**2)
                  if(rad.lt.diff) then
                     diff=rad
                     w=rad/rsig
                     gaus=(exp(-w*w/2.))/gaus0
                     if(rad.lt.1.) gaus=1.
                     arr(i,j,iz)=z(k)*gaus
                  endif
               enddo
            enddo
         enddo
      enddo
 667  continue
      close(2)
      
      naxis=3
      naxes(1)=nx
      naxes(2)=ny
      naxes(3)=ntot
      iblock=1
      igc=0
      ier=0

      call ftinit(50,'image3d.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
      endif
      print *,naxes(1),naxes(2)
      call ftp3de(50,igc,100,100,naxes(1),naxes(2),ntot,arr,ier)
      call ftclos(50,ier)

      end