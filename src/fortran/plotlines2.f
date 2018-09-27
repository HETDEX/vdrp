
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      real xb(nmax),yb(nmax),yin(nmax)
      real yall(1000,nmax),ynl(nmax),ynu(nmax)
      character file1*80,file2*80,c1*18

      ibin=1
      ib1=(ibin-1)/2
      xib=float(ibin)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=3500.
      xmax=5450.
      ymin=0.
      ymax=0.18
      ymin=0.5
      ymax=1.5
      call pgsls(1)
      call pgslw(1)
      call pgsci(1)
      call pgsch(1.3)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel("Wavelength","Multi/SpStandard","")
      call pgsch(1.)

      open(unit=1,file='list2',status='old')

      nl=0
      ic=1
      do il=1,10000
         read(1,*,end=666) file1,file2
         nl=nl+1
         open(unit=2,file=file1,status='old')
         open(unit=3,file=file2,status='old')
         n=0
         do i=1,nmax
            read(2,*,end=667) x1,x2
            n=n+1
            x(n)=x1
            y(n)=x2
         enddo
 667     continue
         close(2)
         nb=0
         do i=1,nmax
            read(3,*,end=668) x1,x2
            nb=nb+1
            xb(nb)=x1
            yb(nb)=x2
         enddo
 668     continue
         close(3)
         do i=1,n
            call xlinint(x(i),nb,xb,yb,yl)
            yn(i)=y(i)/yl
            yin(i)=yn(i)
         enddo
         call biwgt(yin,n,xbout,xsout)
         print *,xbout
c         xbout=1.25
         do i=1,n
            yn(i)=yn(i)/xbout
            yall(il,i)=yn(i)
         enddo
         ic=ic+1
         call pgsci(ic)
c         call pgline(n,x,y)
c         call pgline(nb,xb,yb)
         call pgline(n,x,yn)
 866     continue
         close(2)
      enddo
 666  continue
      close(1)

      do i=1,n
         do j=1,nl
            yin(j)=yall(j,i)
         enddo
         call biwgt(yin,nl,xbout,xsout)
         yn(i)=xbout
         ynl(i)=xbout-xsout
         ynu(i)=xbout+xsout
      enddo
      call pgslw(5)
      call pgsci(1)
c      call pgline(n,x,yn)
      call pgsls(4)
c      call pgline(n,x,ynl)
c      call pgline(n,x,ynu)

      nin=0
      do i=1,n
         if(x(i).lt.4500) then
            nin=nin+1
            yin(nin)=yn(i)
         endif
      enddo
      call biwgt(yin,nin,xbout,xsout)
      print *,xbout,xsout
      nin=0
      do i=1,n
         if(x(i).gt.4500.and.x(i).lt.5400.) then
            nin=nin+1
            yin(nin)=yn(i)
         endif
      enddo
      call biwgt(yin,nin,xbout,xsout)
      print *,xbout,xsout

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
