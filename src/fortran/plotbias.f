
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real yel(nmax),yeu(nmax)
      character file1*80,file2*80,c1*3

      call pgbegin(0,'?',3,3)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=1.
      xmax=31.
      ymin=-1
      ymax=2
      call pgsls(1)
      call pgslw(1)

      open(unit=1,file='list',status='old')

      nl=0
      do il=1,100
         do ia=1,4
            read(1,*,end=666) file1
            if(ia.eq.1) then
               c1=file1(1:3)
               call pgsci(1)
               call pgenv(xmin,xmax,ymin,ymax,0,0)
               call pglabel('Day','Average','')
               call pgsch(2.0)
               call pgmtxt('B',-1.4,0.5,0.5,c1)
               call pgsch(1.5)
            endif
            open(unit=2,file=file1,status='old')
            nl=nl+1
            n=0
            do i=1,2000
               read(2,*,end=667) x1,x2
               if(i.eq.1) xoff=x1-1.
               n=n+1
               x(n)=x1-xoff
               y(n)=x2
            enddo
 667        continue
            close(2)
            call pgsci(ia)
            call pgline(n,x,y)
         enddo
      enddo
 666  continue
      close(1)

      call pgend

      end
