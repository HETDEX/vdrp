
      parameter (radtodeg=57.29578)
      real xp(1000000),yp(1000000),xo(1000)
      character a1*1,fileh(4)*40

      yoffrow1=0.22
      yoffrow2=0.22
      xoffrow1=-0.08
      xoffrow2=-0.08

      xcenorig=191.025
      xcen=191.
      xcen=170.
      xcen=195.
      ycen=51.1279
      xside=60.
      xofft=xcen-xcenorig

      ysize=40.0
c      ysize=10.0
      ysize=15.0
      ysize=2.5
      xsize=ysize/cos(ycen/57.3)
      xsize=70.
      xsize=2.5
c      xsize=80.
      xmin=xcen-xsize/2.
      xmax=xcen+xsize/2.
      ymin=ycen-ysize/2.
      ymax=ycen+ysize/2.
      
      call pgbegin(0,'?',1,1)
c      call pgpap(0.,0.3)
      call pgscf(2)
      call pgsch(2.0)
      print *,xmin,xmax,ymin,ymax
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('RA(deg)','DEC','')

      call pgsch(0.5)
      open(unit=1,file='spring.dat',status='old')
      open(unit=11,file='out',status='unknown')
      open(unit=12,file='out2',status='unknown')
      do i=1,10000
         read(1,*,end=777) x1,x2
c         call pgpt1(x1,x2,17)
      enddo
 777  continue
      close(1)
      call pgsch(1.5)

c      open(unit=2,file="HF_cen.dat",status="old")
c      read(2,*) xcene,ycene,ange
c      read(2,*) xcenw,ycen2,angw
c      close(2)

      xcene=xcen
      ycene=ycen

      no=100
      no=172
      nhalf=no/2
      xoff=0.17
      xoff=0.30
      xoff=0.40
      xoff2=-0.05
      xo(1)=0.-xoff*float(nhalf)+xofft
      do i=2,no
         xo(i)=xo(i-1)+xoff
      enddo

      do io=1,no
         xoff=xo(io)
         call pgsfs(1)
         call pgsci(2)
         open(unit=2,file='sp18_E',status='old')
         do i=1,40
c         do i=1,100000
            read(2,*,end=669) x1,x2,x3,x4,x5,x6,x7,x8
            xp(1)=x1+xoff
            yp(1)=x2
            xp(2)=x3+xoff
            yp(2)=x4
            xp(3)=x5+xoff
            yp(3)=x6
            xp(4)=x7+xoff
            yp(4)=x8
            call pgpoly(4,xp,yp)
            write(12,1201) 
     $           xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
            do j=1,4
               yp(j)=yp(j)+yoffrow1
               xp(j)=xp(j)+xoffrow1
            enddo
            call pgpoly(4,xp,yp)
            write(12,1201) 
     $           xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
            do j=1,4
               yp(j)=yp(j)+yoffrow2
               xp(j)=xp(j)+xoffrow2
            enddo
            call pgpoly(4,xp,yp)
            write(12,1201) 
     $           xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
         enddo
 669     continue
         close(2)
         xplot=xcene+xoff-xofft
         yplot=ycene
         call pgpt1(xplot,yplot,43)
         write(11,1001) xplot,yplot,0
         xplot=xplot+xoffrow1
         yplot=yplot+yoffrow1
         call pgpt1(xplot,yplot,43)
         write(11,1001) xplot,yplot,0
         xplot=xplot+xoffrow2
         yplot=yplot+yoffrow2
         call pgpt1(xplot,yplot,43)
         write(11,1001) xplot,yplot,0

         call pgsfs(1)
         call pgsci(4)
         open(unit=2,file='sp18_W',status='old')
         do i=1,40
c      do i=1,100000
            read(2,*,end=668) x1,x2,x3,x4,x5,x6,x7,x8
            xp(1)=x1+xoff+xoff2
            yp(1)=x2
            xp(2)=x3+xoff+xoff2
            yp(2)=x4
            xp(3)=x5+xoff+xoff2
            yp(3)=x6
            xp(4)=x7+xoff+xoff2
            yp(4)=x8
            call pgpoly(4,xp,yp)
            write(12,1201) 
     $           xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
            do j=1,4
               yp(j)=yp(j)+yoffrow1
               xp(j)=xp(j)+xoffrow1
            enddo
            call pgpoly(4,xp,yp)
            write(12,1201) 
     $           xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
            do j=1,4
               yp(j)=yp(j)+yoffrow2
               xp(j)=xp(j)+xoffrow2
            enddo
            call pgpoly(4,xp,yp)
            write(12,1201) 
     $           xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
         enddo
 668     continue
         close(2)
         xplot=xcene+xoff+xoff2-xofft
         yplot=ycene
         call pgpt1(xplot,yplot,43)
         write(11,1001) xplot,yplot,1
         xplot=xplot+xoffrow1
         yplot=yplot+yoffrow1
         call pgpt1(xplot,yplot,43)
         write(11,1001) xplot,yplot,1
         xplot=xplot+xoffrow2
         yplot=yplot+yoffrow2
         call pgpt1(xplot,yplot,43)
         write(11,1001) xplot,yplot,1
      enddo
      close(11)
      close(12)

      fileh(1)='Hetdex.east.txt'
      fileh(2)='Hetdex.west.txt'
      fileh(3)='Hetdex.lower.txt'
      fileh(4)='Hetdex.upper.txt'

      do i=1,4
         open(unit=1,file=fileh(i),status='old')
         do j=1,2
            read(1,*)
         enddo
         nh=0
         do j=1,1000000
            read(1,*,end=966) x1,x2
            nh=nh+1
            xp(nh)=x1
            yp(nh)=x2
            read(1,*,end=966) x1,x2
            read(1,*,end=966) x1,x2
         enddo
 966     continue
         close(1)
         call pgline(nh,xp,yp)
      enddo

      open(unit=1,file='out2',status='old')
      xmin=1e10
      xmax=-1e10
      ymin=1e10
      ymax=-1e10
      do i=1,100000
         read(1,*,end=556) x1,x2
         xmin=min(xmin,x1)
         xmax=max(xmax,x1)
         ymin=min(ymin,x2)
         ymax=max(ymax,x2)
      enddo
 556  continue
      close(1)
      print *,xmin,xmax,ymin,ymax

      call pgsci(3)
      open(unit=1,file='allifu.dat',status='old')
      open(unit=11,file='out3',status='unknown')
      do i=1,1000000
         read(1,*,end=555) 
     $        xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
         if(yp(1).lt.ymin.or.yp(1).gt.ymax) then
            call pgpoly(4,xp,yp)
            write(11,1201) 
     $           xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
         endif
      enddo
 555  continue
      close(1)
      close(11)

 1001 format(2(1x,f9.5),1x,i2)
 1201 format(8(1x,f10.6))

      call pgend
      end

