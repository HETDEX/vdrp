
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax)
      character file1*80,file2*80,c1*3

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=-0.32
      xmax=0.32
      ymin=0.
      ymax=0.7
      call pgsls(1)
      call pgslw(1)

      open(unit=1,file='list',status='old')

      ia=0
      nl=0
      do il=1,1000
         read(1,*,end=666) file1
         open(unit=2,file=file1,status='old')
         if(il.eq.1) then
            call pgsci(1)
            call pgenv(xmin,xmax,ymin,ymax,0,0)
            call pglabel('PA offset','Position Accuracy (arcsec)','')
         endif
         n=0
         do i=1,2000
            read(2,*,end=667) x1,x2,x3
            n=n+1
            x(n)=x1
            y(n)=x2
         enddo
 667     continue
         close(2)
         if(x3.gt.7.) then
            ia=ia+1
            if(ia.eq.15) ia=1
            call pgsci(ia)
            call pgline(n,x,y)
         endif
      enddo
 666  continue
      close(1)

      call pgend

      end
