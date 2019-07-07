
      parameter(nmax=20000,np=4)
      real x(nmax,np),cs(nmax),cg(nmax),ct(nmax)
      real xmina(np),xmaxa(np)
      integer ip(nmax)
      character a1*70,lab(np)*20
      common/cosmo/ h0,xmat,xlam,xr

      open(unit=1,file='sorted',status='old')

      do i=1,nmax
         do j=1,np
            x(i,j)=0.
         enddo
      enddo

      cmin=1e10
      do i=1,nmax
         read(1,*,end=666) (x(i,j),j=1,np),cs(i)
         ct(i)=cs(i)
         cmin=min(cmin,ct(i))
         n=i
      enddo
 666  continue
      close(1)

      print *,"chimin = ",cmin
      do i=1,n
         ct(i)=ct(i)-cmin
      enddo

      lab(1)='M/L'
      lab(2)='BH'
      lab(3)='V\Dcirc'
      lab(4)='R\DC'

c      chimin=1890.
c      chimax=2000.
      chimin=-5.
      chimax=110.
      xmina(1)=1.7
      xmaxa(1)=2.9
      xmina(2)=-1.e8
      xmaxa(2)=1.1e9
      xmina(3)=280.
      xmaxa(3)=620.
      xmina(4)=0.8
      xmaxa(4)=3.2

      call pgbegin(0,'?',2,2)
      call pgscf(2)
      call pgslw(1)
      call pgpap(0.,1.)

      do i=1,np
         xmin=xmina(i)
         xmax=xmaxa(i)
         ymin=chimin
         ymax=chimax
         call pgsch(1.5)
         call pgenv(xmin,xmax,ymin,ymax,0,0)
         call pgsch(1.7)
         call pglabel('','\GD\Gx\U2','')
         call pgmtxt('B',2.5,0.5,0.5,lab(i))
         call pgsch(2.5)
         call plot(i,x,ct,n,xmin,xmax,ymin,ymax,cmin)
      enddo

      call pgend

      end

      subroutine plot(ip,xa,ct,n,xmin,xmax,ymin,ymax,cmin)
      parameter(nmax=20000,np=4)
      real xa(nmax,np),x(nmax),y(nmax),ct(nmax)
      do i=1,n
         x(i)=xa(i,ip)
      enddo
      do i=1,n
         y(i)=ct(i)
      enddo
      itype=17
      do i=1,n
         call pgpt(1,x(i),y(i),itype)
      enddo
      return
      end

      subroutine getlim(n,x,y,xmin,xmax,ymin,ymax)
      real x(n),y(n)
      data big/1.e20/
      xmin=big
      xmax=-big
      ymin=big
      ymax=-big
      do i=1,n
         if(nint(x(i)).ne.9999) then
            xmin=min(xmin,x(i))
            xmax=max(xmax,x(i))
         endif
         if(nint(y(i)).ne.9999) then
            ymin=min(ymin,y(i))
            ymax=max(ymax,y(i))
         endif
      enddo
      xbit=(xmax-xmin)/10.
      xmin=xmin-xbit
      xmax=xmax+xbit
      ybit=(ymax-ymin)/10.
      ymin=ymin-ybit
      ymax=ymax+ybit
      return
      end

      subroutine getlim1(ip,x,n,xmin,xmax)
      parameter(nmax=20000,np=4)
      real x(nmax,np)
      data big/1.e20/
      xmin=big
      xmax=-big
      ymin=big
      ymax=-big
      do i=1,n
         val=x(i,ip)
         if(nint(val).ne.9999) then
            xmin=min(xmin,val)
            xmax=max(xmax,val)
         endif
      enddo
      xbit=(xmax-xmin)/10.
      xmin=xmin-xbit
      xmax=xmax+xbit
      return
      end
