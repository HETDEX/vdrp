
      parameter (nmax=5048)
      real x(nmax),y(nmax),ft(nmax)
      real wa(nmax),fa(nmax),yo(nmax)
      character file1*130,file2*130,file3*130

      read *,file2
      open(unit=1,file=file2,status='old',err=678)
      na=0
      do i=1,nmax
         read(1,*,end=677) x1,x2
         na=na+1
         wa(na)=x1
         fa(na)=x2
      enddo
 677  continue
      close(1)
      goto 679
 678  continue
      na=2
      wa(1)=3500.
      fa(1)=1.
      wa(2)=5500.
      fa(2)=1.
      write(*,*) "Amp Norm does not exist: ",file3
 679  continue

      open(unit=1,file='in',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4
         n=n+1
         x(n)=x1
         y(n)=x2
         ft(n)=x3
         yo(n)=x4
      enddo
 666  continue
      close(1)

      open(unit=11,file='out',status='unknown')
      do i=1,n
         call xlinint(x(i),na,wa,fa,yfp)
         write(11,*) x(i),y(i)/yfp,ft(i),yfp,yo(i)/yfp
      enddo
      close(11)

 706  continue
      end

      subroutine xlinint(xp,n,x,y,yp)
      real x(n),y(n)
      do j=1,n-1
         if(xp.ge.x(j).and.xp.lt.x(j+1)) then
            yp=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
            return
         endif
      enddo
      if(xp.lt.x(1)) yp=y(1)
      if(xp.gt.x(n)) yp=y(n)
      return
      end
