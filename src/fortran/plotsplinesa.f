
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,100),xin(100)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax),ysum(nmax)
      character file1*80,file2*80,c1*6,f1*30,f2*30


      facnorm=1.e17
      call pgbegin(0,'?',3,3)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      ylocut=-500.
      yhicut=5000.
      ylocut=-100.
      yhicut=1000.
      ylocut=-3.
      yhicut=30.

      ymin=-1.5
      ymax=5.

      open(unit=3,file='inlist',status='old')
      do iall=1,10000
         read(3,*,end=866) f1,sn
         nf=0
         do j=1,20
            if(f1(j:j).ne." ") then
               nf=nf+1
            endif
         enddo

      call pgsls(1)
      call pgslw(1)
      f2=f1(1:nf)//"/list"

      open(unit=1,file=f2,status='old')

      do j=1,1000
         ysum(j)=0
      enddo

      ic=0
      nl=0
      nsum=0
      ymaxs=0.
      do il=1,1000
         read(1,*,end=666) file1
         f2=f1(1:nf)//"/"//file1
         nsum=nsum+1
         open(unit=2,file=f2,status='old')
         n=0
         ireject=0
         do i=1,2000
            read(2,*,end=667) x1,x2,x3
            n=n+1
            x(n)=x1
c            y(n)=x2
            y(n)=x3*facnorm
            if(y(n).gt.ylocut.and.y(n).lt.yhicut) then
               ysum(n)=ysum(n)+y(n)
               ymaxs=max(ymaxs,ysum(n))
            else
               ireject=ireject+1
            endif
         enddo
 667     continue
         close(2)
         if(il.eq.1) then
            call pgsci(1)
            call pgsch(1.5)
            call pgenv(x(1),x(n),ymin,ymax,0,0)
c            call pglabel('Wavelength','Counts','')
             call pglabel('Wavelength','1e-17 ergs/cm\U2\D/s','')
         endif
         ic=ic+1
         call pgsci(ic)
         if(ireject.le.10) call pgline(n,x,y)
c         call pgline(n,x,y)
         if(ic.eq.12) ic=1
      enddo
 666  continue
      close(1)

      nin=0
      do i=1,n
         if(i.le.10.or.i.ge.(n-10)) then
            nin=nin+1
            xin(nin)=ysum(i)
         endif
      enddo
      call biwgt(xin,nin,xb,xs)
      if(xb.le.0) then
         do i=1,n
            ysum(i)=ysum(i)-xb
            ymaxs=max(ymaxs,ysum(i))
         enddo
      endif

      call pgsci(1)
      call pgslw(5)
      open(unit=11,file='splines.out',status='unknown')
      frac=ymaxs/ymax
      frac=max(1.0,frac)
      do i=1,n
         write(11,*) x(i),ysum(i)
         ysum(i)=ysum(i)/frac
      enddo
      close(11)
      call pgline(n,x,ysum)
      print *,frac,xb
      write(c1,1001) frac
 1001 format(f6.1)
      call pgsch(1.7)
      call pgslw(2)
      call pgmtxt('T',-2.5,0.9,1.,c1)
      write(c1,1001) sn
      call pgmtxt('T',-1.5,0.9,1.,c1)
      call pgsch(1.9)
      call pgmtxt('B',-1.2,0.5,0.5,f1(1:nf))

      enddo
 866  continue
      close(3)

      call pgend

      end
