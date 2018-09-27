
      parameter(nmax=20000)
      real x(nmax),y(nmax),xs(nmax),ys(nmax),y3(nmax)

      open(unit=1,file='tmp',status='old')

      n=0
      ymin=1e10
      ymax=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         if(n.gt.1) then
            do j=1,n-1
               if(x1.eq.x(j)) then
c                  print *,x1,x(j),x2
                  x1=x1*1.000001
c                  print *,x1,x(j),x2
               endif
            enddo
         endif
         x(n)=x1
         y(n)=x2
         ymin=min(ymin,y(n))
         ymax=max(ymax,y(n))
      enddo
 666  continue
      close(1)

c      call pgbegin(0,'/null',1,1)
c      call pgpap(0.,1.)
c      call pgsch(1.4)
c      call pgscf(2)
c      call pgslw(2)

c      ymin=-0.02
c      ymax=0.02
      xmin=x(1)
      xmax=x(n)
c      xmin=4700.
c      xmax=5100.
c      call pgenv(xmin,xmax,ymin,ymax,0,0)
c      call pgline(n,x,y)

      xmin=x(1)
      xmax=x(n)
      n2=1000
c      n2=n*2
c      n2=n
      do i=1,n2
         xs(i)=xmin+(xmax-xmin)*float(i-1)/float(n2-1)
c         xs(i)=x(i)
      enddo
      call smooth(n,x,y,n2,xs,ys,y3)
c      call pgsci(2)
c      call pgline(n2,xs,ys)

      open(unit=11,file='smline.out',status='unknown')
      do i=1,n2
         write(11,*) xs(i),ys(i)
      enddo
      close(11)

c      call pgend
      end

      subroutine smooth(n,x,y,n2,x2,y2,y3)
      parameter(nmax=20000,mm=2,nwk=nmax+6*(nmax*mm+1),mm2=mm*2)
      real x(n),y(n),x2(n2),y2(n2),y3(n)
      real*8 dx(nmax),dy(nmax),wx(nmax),cf(nmax),wk(nwk),val,splder
      real*8 q(mm2)

      if(n.gt.nmax) print *,'make nmax bigger in smooth'

c      call qd1('Enter smoothing val ','smflat.def',val)
c      call savdef
      val=0.d0
      md=3
      if(val.eq.0.) md=2
      m=2

      do i=1,n
         dx(i)=dble(x(i))
         dy(i)=dble(y(i))
         wx(i)=1.d0
      enddo

      call gcvspl(dx,dy,nmax,wx,1.d0,m,n,1,md,val,cf,nmax,wk,ier)
c      if(ier.ne.0) print *,'ier= ',ier

      do i=1,n2
         in=i
         y2(i)=sngl(splder(0,m,n,dble(x2(i)),dx,cf,in,q))
      enddo

      do i=1,n
         in=i
         y3(i)=sngl(splder(0,m,n,dble(x(i)),dx,cf,in,q))
      enddo

      return
      end
