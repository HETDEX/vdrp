
      parameter(nmax=10000)
      real x(nmax),y(nmax),y2(nmax),ysum(nmax),ysum2(nmax)
      real ya1(1000,nmax),ya2(1000,nmax),yin1(nmax),yin2(nmax)
      character file1*80,file2*80,c1*8

      ylocut=-100.
      yhicut=1000.

      open(unit=1,file='list',status='old')

      do i=1,nmax
         ysum(i)=0.
         ysum2(i)=0.
      enddo

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
            read(2,*,end=667) x1,x2,x3
            n=n+1
            x(n)=x1
            y(n)=x2
            y2(n)=x3
            if(y(n).gt.ylocut.and.y(n).lt.yhicut) then
               ysum(n)=ysum(n)+y(n)
               ysum2(n)=ysum2(n)+y2(n)
            endif
            ya1(nsum,n)=x2
            ya2(nsum,n)=x3
         enddo
 667     continue
         close(2)
      enddo
 666  continue
      close(1)

      open(unit=11,file='splines.out',status='unknown')
      do i=1,n
         do j=1,nsum
            yin1(j)=ya1(j,i)
            yin2(j)=ya2(j,i)
         enddo
         call biwgt(yin1,nsum,xb1,xs1)
         call biwgt(yin2,nsum,xb2,xs2)
         xs1=xs1*sqrt(float(nsum))
         xs2=xs2*sqrt(float(nsum))
         write(11,*) x(i),ysum(i),ysum2(i),xs1,xs2
      enddo
      close(11)

      end
