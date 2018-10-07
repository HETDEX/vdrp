
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax)
      character file1*80,file2*80,c1*3

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=3500.
      xmax=5450.
      ymin=0.
      ymax=0.19
      call pgsls(1)
      call pgslw(3)

      open(unit=1,file='list2',status='old')
c      open(unit=1,file='list',status='old')
c      open(unit=2,file='tpnew.dat',status='old')

      nl=0
      ic=1
      do il=1,1000
c         read(1,*,end=666) file1
         read(1,*,end=666) file1,itype
         open(unit=2,file=file1,status='old')
         if(il.eq.1) then
            call pgsci(1)
            call pgenv(xmin,xmax,ymin,ymax,0,0)
            call pglabel('Wavelength','Throughput','')
            call pgsch(1.5)
            call pgslw(4)
         endif
         n=0
         do i=1,2000
            read(2,*,end=667) x1,x2
            n=n+1
            x(n)=x1
            y(n)=x2
         enddo
 667     continue
         close(2)
         ic=ic+1
         if(ic.eq.14) ic=2
         call pgsci(ic)
         if(itype.eq.0) then
            call pgsci(1)
            call pgslw(5)
            call pgsls(4)
         else
            call pgslw(4)
            call pgsls(1)
         endif
         call pgline(n,x,y)
      enddo
 666  continue
      close(1)

      call pgend

      end
