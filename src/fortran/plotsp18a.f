
      parameter (radtodeg=57.29578)
      real xp(1000000),yp(1000000),xo(1000)
      character a1*1,fileh(4)*40

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
c      ysize=2.
      xsize=ysize/cos(ycen/57.3)
      xsize=70.
c      xsize=10.
c      xsize=80.
      xmin=xcen-xsize/2.
      xmax=xcen+xsize/2.
      ymin=ycen-ysize/2.
      ymax=ycen+ysize/2.
      
      call pgbegin(0,'?',1,1)
      call pgpap(0.,0.2)
      call pgscf(2)
      call pgsch(1.5)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('RA(deg)','DEC','')

      call pgsch(0.5)
      open(unit=1,file='spring.dat',status='old')
      do i=1,10000
         read(1,*,end=777) x1,x2
         call pgpt1(x1,x2,17)
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
      no=216
      nhalf=no/2
      xoff=0.17
      xoff=0.30
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
         do i=1,34
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
            do j=1,4
               yp(j)=yp(j)+0.24
            enddo
            call pgpoly(4,xp,yp)
            do j=1,4
               yp(j)=yp(j)+0.24
            enddo
            call pgpoly(4,xp,yp)
         enddo
 669     continue
         close(2)
c         call pgpt1(xcene+xoff-xofft,ycene,43)
c         call pgpt1(xcene+xoff-xofft,ycene,17)
c         write(*,1001) xcene+xoff-xofft,ycene,0

         call pgsfs(1)
         call pgsci(4)
         open(unit=2,file='sp18_W',status='old')
         do i=1,34
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
c            call pgpoly(4,xp,yp)
         enddo
 668     continue
         close(2)
c         call pgpt1(xcene+xoff+xoff2,ycene,5)
c         write(*,1001) xcene+xoff+xoff2,ycene,1
      enddo

      fileh(1)='Hetdex.east.txt'
      fileh(2)='Hetdex.west.txt'
      fileh(3)='Hetdex.lower.txt'
      fileh(4)='Hetdex.upper.txt'

      do i=1,0
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
         enddo
 966     continue
         close(1)
         call pgline(nh,xp,yp)
      enddo

 1001 format(2(1x,f9.5),1x,i2)

      call pgend
      end

