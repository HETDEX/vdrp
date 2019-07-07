
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,200),xin(200)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax)
      real yorig(nmax)
      integer ica(nmax)
      character file1*80,file2*80

      call pgbegin(0,'?',1,1)
c      call pgbegin(0,'?',4,3)
      call pgpap(0.,1.)
      call pgsch(1.4)
      call pgscf(2)
      call pgslw(2)

      xmin=3500.
      xmax=5450.
c      xmin=3700.
c      xmax=4600.
      ymin=0.7
      ymax=1.4
c      ymin=0.85
c      ymax=1.25
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Wavelength','A2A','')
      call pgsls(1)
      call pgslw(4)

      open(unit=1,file='list',status='old')

      nl=0
      do il=1,1000
         read(1,*,end=666) file1,ic,it
         open(unit=2,file=file1,status='old')
c         call pgenv(xmin,xmax,ymin,ymax,0,0)
c         call pglabel('Wavelength','FtF','')
         call pgsch(2.0)
c         call pgmtxt('T',-1.5,0.5,0.5,file1(1:20))
         call pgsch(1.4)
         nl=nl+1
         n=0
         do i=1,2000
            read(2,*,end=667) x1,x2
            n=n+1
            x(n)=x1
            y(n)=x2
            ya(n,il)=x2
         enddo
 667     continue
         close(2)
         ica(il)=ic
         call pgsci(ic)
         call pgsls(it)
         call pgslw(5)
         call pgline(n,x,y)
         call pgsci(1)
         call pgsls(1)
      enddo
 666  continue
      close(1)

      do i=1,n
         do j=1,nl
            xin(j)=ya(i,j)
         enddo
         call biwgt(xin,nl,xb,xs)
         y(i)=xb
         yel(i)=y(i)-xs/sqrt(float(nl))
         yeu(i)=y(i)+xs/sqrt(float(nl))
c         yel(i)=y(i)-xs
c         yeu(i)=y(i)+xs
c         print *,x(i),y(i)
      enddo

c      call pgline(n,x,y)
      fac=1.0
      do i=1,n
         y(i)=y(i)*fac
      enddo
c      call pgsls(4)
c      call pgline(n,x,yel)
c      call pgline(n,x,yeu)

      do j=1,nl
         do i=1,n
            ydiff(i)=y(i)/ya(i,j)
            yorig(i)=ya(i,j)
         enddo
         call biwgt(ydiff,n,xb,xs)
         do i=1,n
            ydiff(i)=ya(i,j)*xb
         enddo
         do i=1,n
            ydiffn(i)=ydiff(i)-y(i)
         enddo
         call biwgt(ydiffn,n,xb,xs)
         if(xs.gt.0.0) then
c         if(xs.gt.0.02) then
            print *,j,xs,ic
            ic=ic+1
            if(ic.eq.15) ic=1
            ic=ica(j)
            call pgsci(ic)
c            call pgline(n,x,yorig)
         endif
      enddo

      call pgend

      end
