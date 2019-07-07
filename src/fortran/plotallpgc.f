
      parameter(nmax=20000,np=4)
      real x(nmax,np),cs(nmax),cg(nmax),ct(nmax)
      real xmina(np),xmaxa(np)
      integer ip(nmax)
      character a1*70,lab(np)*20
      common/cosmo/ h0,xmat,xlam,xr

      cmin=136.59
c      cmin=121.0

      open(unit=1,file='sorted',status='old')

      do i=1,nmax
         do j=1,np
            x(i,j)=0.
         enddo
      enddo

      do i=1,nmax
         read(1,*,end=666) (x(i,j),j=1,np),cs(i)
         ct(i)=cs(i)
         n=i
      enddo
 666  continue
      close(1)

      lab(1)='M/L'
      lab(2)='BH'
      lab(3)='V\Dcirc'
      lab(4)='R\DC'

      xmina(1)=2.
      xmaxa(1)=5.
      xmina(2)=0.
      xmaxa(2)=1.0e9
      xmina(3)=50.
      xmaxa(3)=900.
      xmina(4)=0.
      xmaxa(4)=5.

      call pgbegin(0,'?',np,np)
      call pgscf(2)
      call pgslw(1)
      call pgpap(0.,1.)

      do i=1,np
         do j=1,np
            xmin=xmina(j)
            xmax=xmaxa(j)
            ymin=xmina(i)
            ymax=xmaxa(i)
            if(i.eq.j) then
               call pgpage
               call pgvport(0.,1.,0.,1.)
c               call getlim1(i,x,n,xmin,xmax)
               call pgwindow(xmin,xmax,xmin,xmax)
               call pgsch(6.0)
               xcen=(xmax+xmin)/2.
               call pgptxt(xcen,xcen,0.,0.5,lab(i))
               call pgsch(3.0)
               if(i.eq.np) then
                  call pgvport(0.05,0.95,0.15,0.85)
                  call pgbox('m',0.,0,'',0.,0)
                  call pgvport(0.15,0.85,0.05,0.95)
                  call pgbox('',0.,0,'n',0.,0)
               else
                  call pgvport(0.05,0.95,0.15,0.85)
                  call pgbox('n',0.,0,'',0.,0)
                  call pgvport(0.15,0.85,0.05,0.95)
                  call pgbox('',0.,0,'m',0.,0)
               endif
               call pgsch(1.0)
            else
               call plot(j,i,x,ct,n,xmin,xmax,ymin,ymax,cmin)
            endif
         enddo
      enddo

      call pgend

      end

      subroutine plot(ip,jp,xa,ct,n,xmin,xmax,ymin,ymax,cmin)
      parameter(nmax=20000,np=4)
      real xa(nmax,np),x(nmax),y(nmax),ct(nmax)
      ccut1=cmin+1.0
      ccut2=cmin+4.0
      do i=1,n
         x(i)=xa(i,ip)
      enddo
      do i=1,n
         y(i)=xa(i,jp)
      enddo
c      call getlim(n,x,y,xmin,xmax,ymin,ymax)
      call pgpage
      call pgvport (0.05,0.95,0.05,0.95)
      call pgwindow(xmin,xmax,ymin,ymax)
      call pgsch(1.5)
      call pgbox('bct',0.,0,'bct',0.,0)
      itype=17
      call pgsch(4.)
      do i=1,n
         call pgsci(14)
         call pgsch(1.0)
         if(ct(i).le.ccut2) then
            call pgsch(2.)
            call pgsci(2)
         endif
         if(ct(i).le.ccut1) then
            call pgsch(5.)
            call pgsci(1)
         endif
         call pgpt(1,x(i),y(i),itype)
      enddo
      call pgsci(1)
      call pgsch(1.0)
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
