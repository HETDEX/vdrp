
      parameter(nmax=10000)
      real*8 r(nmax),d(nmax),p(nmax),x3,x4,x5
      integer ic(nmax)
      character id(nmax)*8,is(nmax)*3,i1*8,i2*3

      open(unit=2,file='j1',status='old')
      open(unit=3,file='out',status='unknown')

      n=0
      do i=1,nmax
         read(2,*,end=666) i1,i2,x3,x4,x5
         n=n+1
         id(n)=i1
         is(n)=i2
         r(n)=x3
         d(n)=x4
         p(n)=x5
         ic(n)=0
      enddo
 666  continue
      rewind(2)
      do i=1,n
         read(2,*,end=667) i1,i2,x3,x4,x5
         if(ic(i).eq.0) then
            do j=1,n
               if(i.ne.j) then
                  if(i1.eq.id(j).and.i2.eq.is(j).and.ic(j).eq.0) then
                     write(3,1001) id(j),is(j),r(j),d(j),p(j)
                     ic(j)=1
                     goto 866
                  endif
               endif
            enddo
            write(3,1001) i1,i2,x3,x4,x5
 866        continue
            ic(i)=1
         endif
      enddo
 667  continue

c      close(1)
      close(2)
      close(3)
 1001 format(a8,1x,a3,2(1x,f11.6),1x,f7.2)

      end
