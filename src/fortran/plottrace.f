
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      real xb(nmax),yb(nmax)
      character file1*80,file2*80,c1*18

      ibin=1
      ib1=(ibin-1)/2
      xib=float(ibin)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=1.
      xmax=100.
      ymin=-1.
      ymax=2.
      call pgsls(1)
      call pgslw(1)
      call pgsci(1)
      call pgsch(1.)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Number','Trace Start','')
      call pgsch(1.)

      open(unit=1,file='list',status='old')

      nl=0
      ic=0
      do il=1,10000
         read(1,*,end=666) file1
         open(unit=2,file=file1,status='old')
         n=0
         do i=1,nmax
            read(2,*,end=667) x1,x2
            n=n+1
            x(n)=x1
            y(n)=x2
         enddo
 667     continue
         close(2)
         c1=file1(1:17)
         ic=ic+1
         call pgsci(ic)
         call pgline(n,x,y)
 866     continue
         close(2)
      enddo
 666  continue
      close(1)

      call pgend

      end
