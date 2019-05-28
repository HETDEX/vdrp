
      parameter(nmax=1000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax),y2(nmax)
      character file1*80,file2*80,c1*3,ct*12

      ct='            '
      open(unit=1,file='title',status='old',err=555)
      read(1,*) ct
 555  continue
      close(1)      

c      call pgbegin(0,'?',3,3)
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=1.
      xmax=1032.
      xmin=3500.
      xmax=5500.
      ymin=-1.5
      ymax=1.5
      ymin=0.
      ymax=0.22
c      ymax=0.7
      call pgsls(1)
      call pgslw(2)

      open(unit=1,file='in2',status='old')
      nl=0
      do i=1,1000
         read(1,*,end=677)
         nl=nl+1
      enddo
 677  continue
      rewind(1)

      num1=0
      num2=0
      num3=0
      ic=1
      do ia=1,nl
         if(ia.lt.nl) then
c            read(1,*,end=666) file1
            read(1,*,end=666) file1,i2,x3,x4,i5,x6,x7,x8
            if(i5.eq.0) num1=num1+1
            if(i5.le.1) num2=num2+1
            if(i5.le.2) num3=num3+1
         else
            read(1,*,end=666) file1
            x8=1.
            i5=0
         endif
         open(unit=2,file=file1,status='old')
         if(ia.eq.1) then
            call pgsci(1)
            call pgenv(xmin,xmax,ymin,ymax,0,0)
            call pglabel('Wavelength','Throughput',ct)
            call pgsch(1.5)
         endif
         n=0
         ysum=0.
         do i=1,1000
            if(ia.lt.nl) then
               read(2,*,end=667) x1,x2
            else
               read(2,*,end=667) x1,x2,x3,x4,x5
            endif
            n=n+1
            x(n)=x1
            y(n)=x2
            if(ia.eq.nl) y2(n)=x5
            ysum=ysum+x2
c- this next line applies aperture correction
            y(n)=x2/x8
         enddo
 667     continue
         close(2)
         n=n-1
         if(ia.eq.nl) then
            call pgslw(8)
            ic=1
         else
            ic=ic+1
            if(ic.eq.13) ic=2
            call pgslw(3)
         endif
         call pgsls(1)
         if(x8.lt.0.8) call pgsls(4)
         call pgsci(ic)
         if(i5.eq.1) then
            call pgsci(14)
            call pgslw(1)
         endif
         if(i5.eq.2) then
            call pgsci(14)
            call pgslw(1)
            call pgsls(4)
         endif
         if(ia.eq.nl) then
            call pgsls(4)
            call pgline(n,x,y2)
            call pgsls(1)
         endif
         if(ysum.gt.0.01) call pgline(n,x,y)
         call pgslw(1)
      enddo
 666  continue
      close(1)
      ct='            '
      write(ct(1:3),"(i3)") num1
      write(ct(5:7),"(i3)") num2
      write(ct(9:11),"(i3)") num3
      call pgmtxt('T',0.5,0.9,1.,ct)

      call pgend

      end
