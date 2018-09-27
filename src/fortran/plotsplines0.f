
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax),ysum(nmax)
      character file1*80,file2*80,c1*6

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      ymin=-100.
      ymax=250.
      call pgsls(1)
      call pgslw(1)

      open(unit=1,file='list',status='old')

      ic=0
      nl=0
      nsum=0
      ymaxs=0.
      do il=1,1000
         read(1,*,end=666) file1
         nsum=nsum+1
         open(unit=2,file=file1,status='old')
         n=0
         do i=1,2000
            read(2,*,end=667) x1,x2
            n=n+1
            x(n)=x1
            y(n)=x2
            ysum(n)=ysum(n)+y(n)
            ymaxs=max(ymaxs,ysum(n))
         enddo
 667     continue
         close(2)
         ic=ic+1
      enddo
 666  continue
      close(1)

      call pgsci(1)
      call pgsci(1)
      open(unit=11,file='splines.out',status='unknown')
      ymin=1e10
      ymax=-1e10
      do i=1,n
         write(11,*) x(i),ysum(i)
         ymin=min(ymin,ysum(i))
         ymax=max(ymax,ysum(i))
      enddo
      close(11)
      call pgenv(x(1),x(n),ymin,ymax,0,0)
      call pglabel('Wavelength','Counts','')
      call pgslw(2)
      call pgline(n,x,ysum)

      call pgend

      end
