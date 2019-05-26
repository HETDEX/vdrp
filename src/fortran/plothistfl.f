
      parameter(nmax=15000)
      real x(nmax),xl(2),yl(2)
      character file1*80

      fl0=4.1
      fl1=5.1
      fl2=9.0

 1    write(*,"('Data file : '$)")
      read *,file1
      open(unit=1,file=file1,status='old',err=1)
      
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

c      write(*,"('Limits : '$)")
c      read *,xmin,xmax
      xmin=2.
      xmax=18.
      ymin=0.
      ymax=300.
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgslw(5)
      xl(1)=fl0
      xl(2)=xl(1)
      yl(1)=ymin
      yl(2)=ymax
      call pgsci(3)
      call pgline(2,xl,yl)
      xl(1)=fl1
      xl(2)=xl(1)
      yl(1)=ymin
      yl(2)=ymax
      call pgsci(4)
      call pgline(2,xl,yl)
      xl(1)=fl2
      xl(2)=xl(1)
      yl(1)=ymin
      yl(2)=ymax
      call pgsci(2)
      call pgline(2,xl,yl)

      call pgsci(1)
      call pghist(n,x,xmin,xmax,19,1)
      call pgslw(3)
      call pglabel('Flux Limit (1e-17 ergs/cm\U2\D/s)','Number','')

      call pgend

      end
