
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),yin(nmax),xin(100)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax)
      character file1*80,file2*80,c1*20

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=3500.
      xmax=5500.
      ymin=0.93
      ymax=1.07
      ymin=0.8
      ymax=1.2
      call pgsls(1)
      call pgslw(1)

      open(unit=1,file='list',status='old')

      ic0=0
      nl=0
      do il=1,1000
         do ia=1,112
            read(1,*,end=666) file1
            open(unit=2,file=file1,status='old')
            if(ia.eq.1) then
               c1=file1(7:20)
               call pgsci(1)
               call pgenv(xmin,xmax,ymin,ymax,0,0)
               call pglabel('Wavelength','Relative Normalization',c1)
               call pgsch(1.5)
            endif
            n=0
            do i=1,2000
               read(2,*,end=667) x1,x2
               n=n+1
               x(n)=x1
               y(n)=x2
               yin(n)=x2
            enddo
 667        continue
            close(2)
            call biwgt(yin,n,xb,xs)
            do i=1,n
c               y(i)=y(i)/xb
            enddo
            ic=2
            if(ia.lt.37) ic=1
            if(ia.gt.75) ic=4
            ic0=ic0+1
            if(ic0.eq.15) ic0=0
            call pgsci(ic0)
            call pgline(n,x,y)
         enddo
      enddo
 666  continue
      close(1)

      call pgend

      end
