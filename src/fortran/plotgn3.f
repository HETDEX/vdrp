
      parameter (radtodeg=57.29578)
      real xp(100000),yp(100000),xo(1000),yo(1000),xyo(1000)
      real x(1000),y(1000)
      character a1*1

      open(unit=13,file='gn_foot.dat',status='old')
      xcen=189.237
      ycen=62.2457
      xside=60.

      xsize=1.6
      ysize=1.6
      xsize=0.3/cos(ycen/57.3)
      ysize=0.3
      xmin=xcen-xsize/2.
      xmax=xcen+xsize/2.
      ymin=ycen-ysize/2.
      ymax=ycen+ysize/2.
      
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.5)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('RA(deg)','DEC','')

      xoff=0.
      yoff=0.
      open(unit=2,file='gn_W',status='old')
      do i=1,36
         read(2,*,end=669) x1,x2,x3,x4,x5,x6,x7,x8
         xp(1)=x1+xoff
         yp(1)=x2+yoff
         xp(2)=x3+xoff
         yp(2)=x4+yoff
         xp(3)=x5+xoff
         yp(3)=x6+yoff
         xp(4)=x7+xoff
         yp(4)=x8+yoff
         call pgpoly(4,xp,yp)
      enddo
 669  continue
      close(2)

      call pgsfs(2)
      call pgsci(4)

      n=0
      do i=1,4
         read(13,*,end=768) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 768  continue
      close(12)
      call pgslw(4)
      call pgsci(3)
      call pgpoly(n,x,y)

 1001 format(2(1x,f9.5),1x,i2)

      call pgend
      end

