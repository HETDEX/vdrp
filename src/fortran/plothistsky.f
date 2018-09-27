
      parameter(nmax=15000)
      real x(nmax),y(nmax)
      character file1*80

 1    write(*,"('Data file : '$)")
      read *,file1
      open(unit=1,file=file1,status='old',err=1)
      
      scale=0.22
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

      write(*,"('Lower, Upper, Nbin : '$)")
      read *,xmin,xmax,nbin

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(2)

      call pghist(n,x,xmin,xmax,nbin,0)

      open(unit=1,file='skyn.dat',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=667) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 667  continue
      close(1)

      call pgsci(2)
      call pgline(n,x,y)
    

      call pgend

      end
