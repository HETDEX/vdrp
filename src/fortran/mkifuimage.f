
      parameter(nmax=10000)
      real x(nmax),y(nmax),z(nmax),arr(1000,1000),xa(nmax)
      real xpa(nmax),ypa(nmax),xamp(4)
      integer naxes(2)
      character file1*50,aname(448)*24,aamp(4)*2

c      wavec=4550.
      wavec=3550.
      aamp(1)="RU"
      aamp(2)="RL"
      aamp(3)="LL"
      aamp(4)="LU"
      xamp(1)=1.0
      xamp(2)=1.04
      xamp(3)=1.0
c      xamp(4)=0.94
      xamp(4)=0.90

      open(unit=1,file='list',status='old')
      do i=1,448
         read(1,*,end=667) file1
         open(unit=2,file=file1,status='old')
         wdiff=1.e10
         do j=1,10000
            read(2,*,end=668) x1,x2
            if(abs(x1-wavec).lt.wdiff) then
               wdiff=abs(x1-wavec)
               val=x2
            endif
         enddo
 668     continue
         close(2)
         aname(i)=file1(1:24)
         do j=1,4
            if(file1(19:20).eq.aamp(j)) xmult=xamp(j)
         enddo
         xa(i)=val*xmult
      enddo
 667  continue
      close(1)

      open(unit=1,file='405.ixy',status='old')
      do i=1,448
         read(1,*,end=766) x1,x2,file1
         do j=1,448
            if(file1(1:24).eq.aname(j)) then
               xpa(j)=x1
               ypa(j)=x2
               goto 767
            endif
         enddo
 767     continue
      enddo
 766  continue
      close(1)

      do i=1,1000
         do j=1,1000
            arr(i,j)=0.
         enddo
      enddo

      dx=0.5
      nx=101
      ny=101
      xnh=50.
      diff0=1.5

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
            diff=diff0
            do k=1,448
               xpi=xpa(k)+25.
               ypi=ypa(k)+25.
               rad=sqrt((xp-xpi)**2+(yp-ypi)**2)
               if(rad.lt.diff) then
                  diff=rad
                  arr(i,j)=xa(k)
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
