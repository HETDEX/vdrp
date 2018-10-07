
      parameter(nmax=35000)
      real xa1(nmax),xa2(nmax),xa3(nmax)
      character file1*80

      xmin=0.35
      xmax=1.8
      ymax=3900.

      open(unit=1,file="alln.fib",status='old')
      
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3
         n=n+1
         xa1(n)=x1
         xa2(n)=x2
         xa3(n)=x3
      enddo
 666  continue
      close(1)

      call pgbegin(0,'?',1,3)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(2.0)

      call pgsci(1)
      call pgslw(5)
      call pgenv(xmin,xmax,0.,ymax,0,0)
      call pglabel("Relative throughput at 3550","","")
      call pghist(n,xa1,xmin,xmax,19,1)
      call pgenv(xmin,xmax,0.,ymax,0,0)
      call pglabel("Relative throughput at 4550","","")
      call pghist(n,xa2,xmin,xmax,19,1)
      call pgenv(xmin,xmax,0.,ymax,0,0)
      call pglabel("Relative throughput at 5450","","")
      call pghist(n,xa3,xmin,xmax,19,1)

      call pgend

      end
