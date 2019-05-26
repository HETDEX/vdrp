
      parameter(nmax=1000)
      real wave(nmax),xinr(nmax,nmax),xind(nmax,nmax)
      real wadc(5),adc(5)
      real ra(nmax),dec(nmax),ylr(nmax),yld(nmax),xin(nmax),xin2(nmax)
      integer nwa(nmax)
      character file1*40

      nw=9
      wave(1)=3700.
      wave(2)=3900.
      wave(3)=4100.
      wave(4)=4300.
      wave(5)=4500.
      wave(6)=4700.
      wave(7)=4900.
      wave(8)=5100.
      wave(9)=5300.
      do i=1,9
         nwa(i)=0
      enddo

      na=5
      wadc(1)=3500.
      wadc(2)=4000.
      wadc(3)=4500.
      wadc(4)=5000.
      wadc(5)=5500.
      adc(1)=-0.74
      adc(2)=-0.4
      adc(3)=0.0
      adc(4)=0.08
      adc(5)=0.20
      do i=1,na
         adc(i)=-adc(i)
      enddo

      cut1=400.
      cut2=1600.
      cut1=200.
      cut2=1000.
      chicut=10.

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      xmin=3500.
      xmax=5500.
      ymin=-0.5
      ymax=0.7
      call pgsls(1)
      call pgsci(1)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel("Wavelength \(2078)","Offset (arcsec)","")

      call pgsch(0.7)
      open(unit=1,file='list2',status='old')
      n=0
      do i=1,nmax
         do j=1,nw
            read(1,*,end=667) file1
            read(file1(14:18),1002) id
            read(file1(20:23),1001) iwave
            wave1=float(iwave)
            open(unit=2,file=file1,status='old')
            read(2,*) x1,x2,x3
            close(2)
            ra(j)=x1
            dec(j)=x2
c            print *,id,iwave,ra(j),dec(j)
         enddo
         if(x3.gt.0) then
            r0=ra(5)
            d0=dec(5)
            cosd=cos(d0/57.3)
            do j=1,nw
               dr=3600.*(r0-ra(j))*cosd
               dd=3600.*(d0-dec(j))
               rad=sqrt(dr*dr+dd*dd)
               call pgsci(2)
c               call pgpt1(wave(j),dr,17)
               call pgsci(4)
c               call pgpt1(wave(j),dd,17)
               nwa(j)=nwa(j)+1
               xinr(j,nwa(j))=dr
               xind(j,nwa(j))=dd
            enddo
         endif
      enddo
 666  continue
 667  continue

      do i=1,nw
         do j=1,nwa(i)
            xin(j)=xinr(i,j)
            xin2(j)=xind(i,j)
         enddo
         call biwgt(xin,nwa(i),xb,xs)
         ylr(i)=xb
         call biwgt(xin2,nwa(i),xb,xs)
         yld(i)=xb
c         print *,wave(i),ylr(i),xs,nwa(i)
      enddo
      call pgslw(5)
      call pgsci(2)
      call pgline(nw,wave,ylr)

      sum=0.
      do i=1,nw
         call xlinint(wave(i),na,wadc,adc,yv)
         sum=sum+(yv-ylr(i))
      enddo
      sum=sum/float(nw)
      do i=1,na
         adc(i)=adc(i)-sum
      enddo
      call pgsls(4)
      call pgline(na,wadc,adc)
      call pgsls(1)
      call pgsci(4)
      call pgline(nw,wave,yld)

      call pgend
 1001 format(i4)
 1002 format(i4)
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
