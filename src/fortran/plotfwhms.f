
      parameter(nmax=1000)
      real wave(nmax),fw(nmax),xin(nmax),xina(nmax,nmax)
      real xl(nmax),yl(nmax),ylf(nmax),xrat(nmax)
      integer nwa(nmax)
      character file1*40

      open(unit=1,file='fwin',status='old')
      read(1,*) fw_g
      close(1)
c      fw_g=1.69

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

      cut1=400.
      cut2=1600.
      cut1=200.
      cut2=1000.
      fwl=1.8
      fwh=2.4
      fwl=1.4
      fwh=2.8
      chicut=10.

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      xmin=3500.
      xmax=5500.
      ymin=1.4
      ymax=2.7
      call pgsls(1)
      call pgsci(1)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel("Wavelength \(2078)","FWHM (arcsec)","")

      call pgsch(0.7)
      open(unit=1,file='list',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=667) file1
         read(file1(20:23),1001) iwave
         wave1=float(iwave)
         open(unit=2,file=file1,status='old')
         read(2,*,end=666) x1,x2,x3
         close(2)
         cut=cut1+(wave1-3500)/2000.*(cut2-cut1)
         if(x2.lt.chicut.and.x3.gt.cut.and.x1.gt.fwl.and.x1.lt.fwh) then
            call pgpt1(wave1,x1,17)
            do iw=1,nw
               if(abs(wave(iw)-wave1).lt.10) then
                  nwa(iw)=nwa(iw)+1
                  xina(iw,nwa(iw))=x1
               endif
            enddo
         endif
      enddo
 666  continue
 667  continue

      do i=1,nw
         do j=1,nwa(i)
            xin(j)=xina(i,j)
         enddo
         call biwgt(xin,nwa(i),xb,xs)
         yl(i)=xb
         print *,wave(i),yl(i),xs,nwa(i)
      enddo
      call pgslw(5)
      call pgsci(2)
c      call pgline(nw,wave,yl)

      xrat0=yl(5)
      do i=1,nw
         xrat(i)=yl(i)/(xrat0*(4500./wave(i))**0.2)
      enddo
      call biwgt(xrat,nw,xb,xs)
      print *,xb
      xl(1)=xmin
      xl(2)=xmax
      ylf(1)=yl(5)*xb*(4500./xmin)**0.2
      ylf(2)=yl(5)*xb*(4500./xmax)**0.2
      call pgslw(5)
      call pgsls(1)
      call pgsci(2)
      call pgline(2,xl,ylf)
      call pgsls(1)

      yl(1)=fw_g
      yl(2)=yl(1)
      call pgsci(4)
      call pgslw(4)
      call pgline(2,xl,yl)
      ylf(1)=yl(1)*(4790./xmin)**0.2
      ylf(2)=yl(1)*(4790./xmax)**0.2
      call pgsci(4)
      call pgslw(4)
      call pgsls(4)
      call pgline(2,xl,ylf)


      call pgend
 1001 format(i4)
      end
