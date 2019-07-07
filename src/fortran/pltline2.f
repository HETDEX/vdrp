
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      character file1*80,file2*80

      file1='out'
      file2='npdyn.out'
c      file2='ben.dat'

      xmin=log10(0.018)
      xmax=log10(80.)
      ymin=log10(1.0)
      ymax=log10(8.0e5)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      call pgenv(xmin,xmax,ymin,ymax,0,30)
      call pglabel('Radius','L\DH\U/pc\U2','')
      call pgslw(4)
      call pgsch(1.)

      open(unit=2,file=file1,status='old')
      n=0
      do i=1,nmax
         read(2,*,end=667) x1,x2
         n=n+1
         x(n)=log10(x1)
         y(n)=log10(x2)
      enddo
 667  continue
      close(2)
      call pgline(n,x,y)

      open(unit=2,file=file2,status='old')
      n2=0
      do i=1,nmax
         read(2,*,end=666) x1,x2,x3
c         read(2,*,end=666) x1,x3
         n2=n2+1
         xn(n2)=log10(x1)
         yn(n2)=log10(x3)
         call xlinint(xn(n2),n,x,y,ynew)
         print *,10**xn(n2),10**(ynew-yn(n2))
      enddo
 666  continue
      close(2)
      call pgsci(2)
      call pgline(n2,xn,yn)

      call pgend

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
