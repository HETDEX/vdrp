      parameter(nmax=10000)
      real x(nmax),y(nmax),s(nmax),rad(nmax),rg(nmax)
      real sg(nmax)
      parameter(pi=3.14159)

      fwhm=2.1
      sig=fwhm/2.35

      xmin=1e10
      xmax=-1e10
      ymin=1e10
      ymax=-1e10
      smax=0.
      open(unit=1,file='s6',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3
         n=n+1
         x(n)=x1
         y(n)=x2
         s(n)=x3
         xmin=min(xmin,x(n))
         xmax=max(xmax,x(n))
         ymin=min(ymin,y(n))
         ymax=max(ymax,y(n))
         smax=max(smax,s(n))
      enddo
 666  continue
      close(1)

      open(unit=1,file='out',status='old')
      read(1,*) x1,x2,amps

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      radmax=(xmax-xmin)*0.8
      ng=100
      do i=1,ng
         rg(i)=float(i-1)/float(ng-1)*radmax
      enddo

      do ia=1,n
         rad(ia)=sqrt((x(ia))**2+(y(ia))**2)
      enddo

      amp=amps
      do ia=1,ng
         w=rg(ia)/sig
         gaus=amp*exp(-w*w/2)/sqrt(2.*pi*sig**2)
         sg(ia)=gaus
      enddo
      call pgenv(0.,radmax,-30.,smax,0,0)
      do ia=1,n
         rad(ia)=sqrt((x(ia))**2+(y(ia))**2)
         call pgpt1(rad(ia),s(ia),17)
      enddo

      do ia=1,ng
         w=rg(ia)/sig
         gaus=amp*exp(-w*w/2)/sqrt(2.*pi*sig**2)
         sg(ia)=gaus
      enddo
      call pgline(ng,rg,sg)
      
      call pgend

      end

      subroutine xlinint(xp,n,x,y,yp)
      real x(n),y(n)
      do j=1,n-1
         if(xp.ge.x(j).and.xp.lt.x(j+1)) then
            yp=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
            return
         endif
      enddo
      if(xp.lt.x(1)) yp=y(1)
      if(xp.gt.x(n)) yp=y(n)
      return
      end
