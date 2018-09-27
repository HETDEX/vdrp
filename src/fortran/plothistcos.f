
      parameter(nmax=15000)
      real x(nmax),xl(1000),yl(1000)
      character file1*80

c 1    write(*,"('Data file : '$)")
c      read *,file1
      
      open(unit=1,file='j2',status='old')
      
      scale=0.22
      scale=1.
      n=0
      xmin=1e30
      xmax=-1e30
      do i=1,nmax
c         read(1,*,end=666) x1,x2,x3
         read(1,*,end=666) x2
         n=n+1
         x(n)=x2*scale
c         print *,n,x(n)
         xmin=min(xmin,x(n))
         xmax=max(xmax,x(n))
      enddo
 666  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(3)

      xmin=0.07
      xmax=0.18
      call pghist(n,x,xmin,xmax,15,0)
      call pglabel('Throughput at 4500','Number','')
      xl(1)=0.1
      xl(2)=0.1
      yl(1)=0.
      yl(2)=100.
      call pgslw(4)
      call pgsci(2)
      call pgline(2,xl,yl)

      call pgend

      end
