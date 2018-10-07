
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      real wadc(10),adc(10),xl1(nmax),yl1(nmax),xl2(nmax),yl2(nmax)
      character file1*80,file2*80,a1,a2,a3,a4

      dtr=180./3.14159
      nw=5
      wadc(1)=3500.
      wadc(2)=4000.
      wadc(3)=4500.
      wadc(4)=5000.
      wadc(5)=5500.
      adc(1)=-0.75
      adc(2)=-0.4
      adc(3)=0.0
      adc(4)=0.08
      adc(5)=0.21

      open(unit=1,file='in',status='old')
      read(1,*) x1,x2,x3,x4,a1,a2,a3,a4,az
      close(1)

c      az=136.455
c      az=308.45
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      xmin=3450.
      xmax=5550.
      ymin=-0.7
      ymax=0.7
      call pgsls(1)
      call pgslw(2)
      call pgsci(1)
      call pgsch(1.2)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Wavelength','RA and DEC Offset (arcsec)','')
      call pgslw(3)
      call pgsch(1.)

      open(unit=1,file='adc.dat',status='old')

      n=0
      do il=1,10000
         read(1,*,end=666) x1,x2,x3,x4,x5,x6
         n=n+1
         call xlinint(x1,nw,wadc,adc,adcv)
         xaoff=adcv*sin(az/dtr)
         yaoff=adcv*cos(az/dtr)
         print *,xaoff,yaoff
         xl1(n)=x1
         yl1(n)=xaoff
         xl2(n)=x1
         yl2(n)=yaoff
         call pgsci(1)
         call pgpt1(x1,x5,17)
         call pgsci(2)
         call pgpt1(x1,x6,17)
      enddo
 666  continue
      close(1)
      call pgsci(1)
      call pgline(n,xl1,yl1)
      call pgsci(2)
      call pgline(n,xl2,yl2)

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
