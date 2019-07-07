
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,1000),xin(1000)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax)
      real yorig(nmax)
      integer ica(nmax)
      character file1*80,file2*80,name*14,nameo(nmax)*50

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.)
      call pgscf(2)
      call pgslw(2)

      xmin=3500.
      xmax=5450.
c      xmin=3700.
c      xmax=4600.
      ymin=0.7
      ymax=1.5
c      ymin=0.85
c      ymax=1.25
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Wavelength','FtF','')
      call pgsls(1)
      call pgslw(1)

      open(unit=1,file='list',status='old')

      nl=0
      icp=1
      do il=1,1000
         read(1,*,end=666) file1,ic
         if(il.eq.1) name=file1(20:43)
         open(unit=2,file=file1,status='old')
         nl=nl+1
         nameo(nl)=file1(1:43)
         n=0
         do i=1,2000
            read(2,*,end=667) x1,x2
            if(x2.eq.0) then
               nl=nl-1
               goto 777
            endif
            n=n+1
            x(n)=x1
            y(n)=x2
            ya(n,nl)=x2
         enddo
 667     continue
         ica(il)=ic
         icp=icp+1
         if(icp.eq.13) icp=2
         call pgsci(icp)
c         call pgline(n,x,y)
 777     continue
         close(2)
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

      xscut=0.01
      icp=1
      open(unit=12,file='outbad',status='unknown')
      do j=1,nl
         do i=1,n
            ydiff(i)=y(i)/ya(i,j)
            yorig(i)=ya(i,j)
         enddo
         call biwgt(ydiff,n,xb,xs)
         do i=1,n
            ydiff(i)=ya(i,j)*xb
         enddo
         if(xs.le.xscut) then
            icp=icp+1
            if(icp.eq.13) icp=2
            call pgsci(icp)
            call pgline(n,x,ydiff)
         else
            write(12,*) xs,nameo(j)
         endif
      enddo
      close(12)

      call pgsci(1)
      call pgslw(5)
      call pgline(n,x,y)
      call pgsch(2.1)
      call pgmtxt('T',-1.5,0.5,0.5,name)
      open(unit=11,file='out',status='unknown')
      do i=1,n
         write(11,1101) x(i),y(i)
      enddo
 1101 format(1x,f6.1,1x,f5.3)

      call pgend

      end
