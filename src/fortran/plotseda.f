
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax)
      character file1*80,file2*80,c1*3

c      call pgbegin(0,'?',3,3)
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=1.
      xmax=1032.
      xmin=3500.
      xmax=5500.
      ymin=-1.5
      ymax=1.5
      ymin=0.
      ymax=0.2
      call pgsls(1)
      call pgslw(2)

      open(unit=1,file='in2',status='old')
      nl=0
      do i=1,1000
         read(1,*,end=677)
         nl=nl+1
      enddo
 677  continue
      rewind(1)

      ic=1
      do ia=1,nl
         read(1,*,end=666) file1
         open(unit=2,file=file1,status='old')
         if(ia.eq.1) then
            call pgsci(1)
            call pgenv(xmin,xmax,ymin,ymax,0,0)
            call pglabel('Wavelength','Throughput','')
            call pgsch(1.5)
         endif
         n=0
         do i=1,8000
            read(2,*,end=667) x1,x2
            n=n+1
            x(n)=x1
            y(n)=x2
         enddo
 667     continue
         close(2)
         n=n-1
         if(ia.eq.nl) then
            call pgslw(8)
            ic=1
         else
            ic=ic+1
            if(ic.eq.13) ic=2
            call pgslw(3)
         endif
         call pgsci(ic)
         call pgline(n,x,y)
         call pgslw(1)
      enddo
 666  continue
      close(1)

      call pgend

      end
