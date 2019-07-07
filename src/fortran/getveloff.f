
      parameter(nmax=1000)
      real ro(nmax),vo(nmax),v(nmax)
      character filen*40

      data big/1.e10/

 1    call qc1('pallmc file ','getveloff.def',filen)
      call savdef
      open(unit=1,file=filen,status='old',err=1)

      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         ro(n)=x1
         vo(n)=x2
      enddo
 666  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.3)
      call pgslw(2)
      call pgask(.false.)

      call getlim(n,ro,vo,rmin,rmax,vmin,vmax)
      call pgenv(rmin,rmax,vmin,vmax,0,0)
      call pgpoint(n,ro,vo,17)

 2    call qr1('Vel to add ','getveloff.def',voff)
      call savdef
      if(voff.eq.666) goto 667

      call pgenv(0.,max(abs(rmin),rmax),0.,max(abs(vmin),abs(vmax)),0,0)
      do i=1,n
         v(i)=abs(vo(i)+voff)
         if(ro(i).lt.0) then
            call pgsci(2)
            call pgpoint(1,-ro(i),v(i),21)
         else
            call pgsci(3)
            call pgpoint(1,ro(i),v(i),17)
         endif
      enddo
      call pgsci(1)

      goto 2

 667  continue
      call pgend

      end

      subroutine getlim(n,x,y,xmin,xmax,ymin,ymax)
      real x(n),y(n)

      data big/1.e10/

      xmin=big
      xmax=-big
      ymin=big
      ymax=-big
      do i=1,n
         xmin=min(xmin,x(i))
         xmax=max(xmax,x(i))
         ymin=min(ymin,y(i))
         ymax=max(ymax,y(i))
      enddo
      
      xbit=(xmax-xmin)/10.
      xmin=xmin-xbit
      xmax=xmax+xbit
      ybit=(ymax-ymin)/10.
      ymin=ymin-ybit
      ymax=ymax+ybit

      return
      end
