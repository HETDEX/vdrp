
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax),ysum(nmax)
      real ysum1(nmax),ysum2(nmax),ysum3(nmax),ysum4(nmax),ysum5(nmax)
      real ysum6(nmax),ysum7(nmax),ysum8(nmax),ysum9(nmax)
      character file1*80,file2*80,c1*6

      nr=9
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
            read(2,*,end=667) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10
            n=n+1
            x(n)=x1
            y(n)=x2+x3+x4+x5+x6+x7+x8+x9+x10
            ysum(n)=ysum(n)+y(n)
            ysum1(n)=ysum1(n)+x2
            ysum2(n)=ysum2(n)+x3
            ysum3(n)=ysum3(n)+x4
            ysum4(n)=ysum4(n)+x5
            ysum5(n)=ysum5(n)+x6
            ysum6(n)=ysum6(n)+x7
            ysum7(n)=ysum7(n)+x8
            ysum8(n)=ysum8(n)+x9
            ysum9(n)=ysum9(n)+x10
            ymaxs=max(ymaxs,ysum(n))
         enddo
 667     continue
         close(2)
         if(il.eq.1) then
            call pgsci(1)
            call pgenv(x(1),x(n),ymin,ymax,0,0)
            call pglabel('Wavelength','Counts','')
         endif
         ic=ic+1
         call pgsci(ic)
         call pgline(n,x,y)
         if(ic.eq.12) ic=1
      enddo
 666  continue
      close(1)

      call pgsci(1)
      call pgslw(5)
      open(unit=11,file='splines.out',status='unknown')
      frac=ymaxs/ymax
      frac=max(1.0,frac)
      do i=1,n
c         write(11,*) x(i),ysum(i)
         write(11,1101) x(i),ysum1(i),ysum2(i),ysum3(i),ysum4(i),
     $        ysum5(i),ysum6(i),ysum7(i),ysum8(i),ysum9(i)
         ysum(i)=ysum(i)/frac
      enddo
      close(11)
      call pgline(n,x,ysum)
      print *,frac
      write(c1,1001) frac
 1001 format(f6.2)
      call pgsch(1.5)
      call pgslw(2)
      call pgmtxt('B',-1.4,0.5,0.5,c1)

      call pgend

 1101 format(1x,f7.2,9(1x,f8.2))
      end
