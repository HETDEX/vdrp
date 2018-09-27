
      parameter(nmax=15000)
      real x(nmax),y(nmax)
      character file1*80

c 1    write(*,"('Data file : '$)")
c      read *,file1
      file1='in'
      open(unit=1,file=file1,status='old')
      
      scale=1.
      n=0
      xmin=1e30
      xmax=-1e30
      do i=1,nmax
         read(1,*,end=666) x2
         n=n+1
         x(n)=x2*scale
         xmin=min(xmin,x(n))
         xmax=max(xmax,x(n))
      enddo
 666  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(2)

c      write(*,"('Limits : '$)")
c      read *,xmin,xmax
      xmin=0.06
      xmax=0.17
      call pghist(n,x,xmin,xmax,12,0)
      call pglabel('Throughput at 4540AA','Number','')

c      xfac=0.78/(0.15*0.78)
      xfac=1.0/0.16
      yfac=12.
      open(unit=1,file='pilot.dat',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=667) x1,x2
         n=n+1
         x(n)=x1/xfac
         y(n)=x2/yfac
      enddo
 667  continue
      close(1)

      call pgslw(4)
      call pgsci(2)
      call pgline(n,x,y)

      n=2
      x(1)=0.62/xfac
      x(2)=0.62/xfac
      y(1)=0.
      y(2)=1000.
      call pgsci(4)
      call pgline(n,x,y)

      call pgend

      end
