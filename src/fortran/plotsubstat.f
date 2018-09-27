
      parameter(nmax=15000)
      real xl(nmax),xh(nmax)
      character file1*80

 1    write(*,"('Data file : '$)")
      read *,file1
      open(unit=1,file=file1,status='old',err=1)
      
      scale=1.
      n1=0
      n2=0
      xmin=1e30
      xmax=-1e30
      do i=1,nmax
         read(1,*,end=666) x1
         if(x1.lt.0) then
            n1=n1+1
            xl(n1)=-x1*scale
         endif
         if(x1.gt.0) then
            n2=n2+1
            xh(n2)=x1*scale
         endif
      enddo
 666  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(2)

      write(*,"('X Limits : '$)")
      read *,xmin,xmax
      write(*,"('Y Limits : '$)")
      read *,ymin,ymax
c      xmin=0.
c      xmax=40.
      nbin=15
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgsci(1)
      call pglabel('Residual',
     $     'Number','')
      call pgslw(5)
      call pgsci(4)
      call pghist(n1,xl,xmin,xmax,nbin,5)
      call pgsci(2)
      call pghist(n2,xh,xmin,xmax,nbin,5)

      call pgend

      end
