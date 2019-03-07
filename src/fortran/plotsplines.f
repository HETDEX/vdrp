
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real xf(nmax),yf(nmax)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax),ysum(nmax)
      character file1*80,file2*80,c1*8

      irange=0
      open(unit=1,file='sprange.dat',status='old',err=555)
      read(1,*) x1,x2
      xw1=x1-x2
      xw2=x1+x2
      irange=1
 555  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      ylocut=-500.
      yhicut=5000.
      ylocut=-100.
      yhicut=1000.

      ymin=-100.
      ymax=250.
      ymin=-1
      ymax=10.
      call pgsls(1)
      call pgslw(1)

      open(unit=1,file='list',status='old')

      ic=0
      nl=0
      nsum=0
      ymaxs=0.
      do il=1,1000
         read(1,*,end=666) file1
         nsum=nsum+1
         open(unit=2,file=file1,status='old')
         n=0
         ireject=0
         do i=1,2000
            read(2,*,end=667) x1,x2
            n=n+1
            x(n)=x1
            y(n)=x2
            if(il.eq.1) then
               yf(n)=y(n)
            endif
            y(n)=y(n)/yf(n)
            if(y(n).gt.ylocut.and.y(n).lt.yhicut) then
               ysum(n)=ysum(n)+y(n)
               ymaxs=max(ymaxs,ysum(n))
            else
               ireject=ireject+1
            endif
         enddo
 667     continue
         close(2)
         if(il.eq.1) then
            call pgsci(1)
            xmin=x(1)
            xmax=x(n)
            if(irange.eq.1) then
               xmin=xw1
               xmax=xw2
            endif
            call pgenv(xmin,xmax,ymin,ymax,0,0)
            call pglabel('Wavelength','Counts','')
         endif
         ic=ic+1
         call pgsci(ic)
         if(ireject.le.10) call pgline(n,x,y)
c         call pgline(n,x,y)
         if(ic.eq.12) ic=1
      enddo
 666  continue
      close(1)

      nin=0
      do i=1,n
         if(i.le.10.or.i.ge.(n-10)) then
            nin=nin+1
            xin(nin)=ysum(i)
         endif
      enddo
      call biwgt(xin,nin,xb,xs)
      if(xb.le.0) then
         do i=1,n
            ysum(i)=ysum(i)-xb
            ymaxs=max(ymaxs,ysum(i))
         enddo
      endif

      call pgsci(1)
      call pgslw(5)
      open(unit=11,file='splines.out',status='unknown')
      frac=ymaxs/ymax
      frac=max(1.0,frac)
c      frac=xb
      do i=1,n
c         ysum(i)=ysum(i)/float(nsum)
c         print *,i,x(i),ysum(i)
         write(11,*) x(i),ysum(i)
         ysum(i)=ysum(i)/frac
c         ysum(i)=ysum(i)-frac
      enddo
      close(11)
      call pgline(n,x,ysum)
      print *,frac,xb
      write(c1,1001) frac
 1001 format(f8.1)
      call pgsch(1.5)
      call pgslw(2)
c      call pgmtxt('B',-1.4,0.5,0.5,c1)

      call pgend

      end
