      parameter(nmax=100000)
      real x(nmax),y(nmax),xa(nmax),ya(nmax)
      real xl(2),yl(2),yb(nmax),yc(nmax),yd(nmax)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=0.
      xmax=6.5
      ymin=0.
      ymax=490.
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Years since 1 Jan 2018',
     $     'IFU Fields Remaining (1000s)','')

      call pgsci(2)
      call pgsch(1.0)
      call pgptxt(4.5,400.,270.,0.,'Spring Complete')
      call pgptxt(5.2,400.,270.,0.,'Fall Complete')
      call pgsci(1)
      call pgsch(1.5)
      call pgsls(4)
      yl(1)=ymin
      yl(2)=ymax
      do i=1,7
         xl(1)=i
         xl(2)=i
         call pgline(2,xl,yl)
      enddo
      xl(1)=xmin
      xl(2)=xmax
      yl(1)=468.
      yl(2)=468.
      call pgsls(1)
      call pgline(2,xl,yl)

      call pgslw(4)
      open(unit=1,file='fields.dat',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 666  continue
      close(1)
      call pgsci(2)
      call pgsls(1)
      call pgline(n,x,y)

      open(unit=1,file='shotperyear',status='old')
      n2=0
      do i=1,nmax
         read(1,*,end=667) x1,x2
         n2=n2+1
         xa(n2)=x1/365.
         ya(n2)=x2*1.1
      enddo
 667  continue
      close(1)
      
      xstart=x(n)
      xend=xmax
c      ystart=y(n)+67.6
      ystart=y(n)+48.
      ystartb=y(n)+48.
      ystartc=y(n)+48.
      ystartd=y(n)+73.
      x(1)=x(n)
      y(1)=y(n)
      yb(1)=y(n)
      yc(1)=y(n)
      yd(1)=y(n)
      nadd=1000
      icheck1=0
      icheck2=0
      icheck3=0
      icheck4=0
      icheck5=0
      icheck6=0
      icheck7=0
      nifu=50
      nifub=50
      nifuc=50
      nifud=78
      do i=2,nadd
         x(i)=xstart+(xend-xstart)*float(i-1)/float(nadd-1)
         xcheck=x(i)
         if(xcheck.gt.1.and.xcheck.lt.2) then
            xcheck=xcheck-1
            if(icheck1.eq.0) ystart=y(i-1)
            if(icheck1.eq.0) ystartb=yb(i-1)
            if(icheck1.eq.0) ystartc=yc(i-1)
            if(icheck1.eq.0) ystartd=yd(i-1)
            icheck1=1
            nifu=67
            nifub=57
         endif
         if(xcheck.gt.2.and.xcheck.lt.3) then
            xcheck=xcheck-2
            if(icheck2.eq.0) ystart=y(i-1)
            if(icheck2.eq.0) ystartb=yb(i-1)
            if(icheck2.eq.0) ystartc=yc(i-1)
            if(icheck2.eq.0) ystartd=yd(i-1)
            icheck2=1
            nifu=78
            nifub=78
         endif
         if(xcheck.gt.3.and.xcheck.lt.4) then
            xcheck=xcheck-3
            if(icheck3.eq.0) ystart=y(i-1)
            if(icheck3.eq.0) ystartb=yb(i-1)
            if(icheck3.eq.0) ystartc=yc(i-1)
            if(icheck3.eq.0) ystartd=yd(i-1)
            icheck3=1
            nifu=78
            nifub=78
         endif
         if(xcheck.gt.4.and.xcheck.le.5) then
            xcheck=xcheck-4
            if(icheck4.eq.0) ystart=y(i-1)
            if(icheck4.eq.0) ystartb=yb(i-1)
            if(icheck4.eq.0) ystartc=yc(i-1)
            if(icheck4.eq.0) ystartd=yd(i-1)
            icheck4=1
            nifu=78
            nifub=78
         endif
         if(xcheck.gt.5.and.xcheck.le.6) then
            xcheck=xcheck-5
            if(icheck5.eq.0) ystart=y(i-1)
            if(icheck5.eq.0) ystartb=yb(i-1)
            if(icheck5.eq.0) ystartc=yc(i-1)
            if(icheck5.eq.0) ystartd=yd(i-1)
            icheck5=1
            nifu=78
            nifub=78
         endif
         if(xcheck.gt.6.and.xcheck.le.7) then
            xcheck=xcheck-6
            if(icheck6.eq.0) ystart=y(i-1)
            if(icheck6.eq.0) ystartb=yb(i-1)
            if(icheck6.eq.0) ystartc=yc(i-1)
            if(icheck6.eq.0) ystartd=yd(i-1)
            icheck6=1
            nifu=78
            nifub=78
         endif
         yp=0
         do j=1,n2-1
            if(xcheck.ge.xa(j).and.xcheck.lt.xa(j+1)) then
               yp=ya(j)
               ypb=ya(j)
               ypc=ya(j)
               ypd=ya(j)
               goto 555
            endif
         enddo
 555     continue
         yp=yp*nifu/1000.
         y(i)=ystart-yp
         ypb=ypb*nifub/1000.
         yb(i)=ystartb-ypb
         ypc=ypc*nifuc/1000.
         yc(i)=ystartc-ypc
         ypd=ypd*nifud/1000.
         yd(i)=ystartd-ypd
      enddo

      do i=1,nadd-1
         if(y(i).le.0) then
            np=i
            goto 777
         endif
      enddo
 777  continue
      npb=nadd-1
      do i=1,nadd-1
         if(yb(i).le.0) then
            npb=i
            goto 778
         endif
      enddo
 778  continue
      npc=nadd-1
      do i=1,nadd-1
         if(yc(i).le.0) then
            npc=i
            goto 779
         endif
      enddo
 779  continue
      npd=nadd-1
      do i=1,nadd-1
         if(yd(i).le.0) then
            npd=i
            goto 780
         endif
      enddo
 780  continue

      call pgline(np,x,y)
      call pgsci(3)
      call pgline(npb,x,yb)
      call pgsci(4)
      call pgline(npc,x,yc)
      call pgsci(1)
      call pgline(npd,x,yd)
      call pgend

      end
