
      parameter(narr=4000,radtodeg=57.29578)
      real x(narr),y(narr),sig(narr),xsig(narr),ysig(narr)
      real xin(narr),yin(narr)
      integer ic(narr)
      character c1*40,c2*40

      ifit=0
      call qc1('Input datafile ','plotxy.def',c2)
      open(unit=1,file=c2,status='old')
      
      ilog=0
      call qr2('Xmin and Xmax ','plotxy.def',xmin,xmax)
      call qr2('Ymin and Ymax ','plotxy.def',ymin,ymax)
      call savdef
      
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.5)

      xoff=0
      yoff=0.
      n=0
      scale=1.1
      scale1=1.0
      scale2=1.0
      ang1=12.
      ang1=0.
      ang2=-22.
      ang2=0.
c      ang1=180.
c      ang2=180.
      ang1=ang1/radtodeg
      ang2=ang2/radtodeg
      do i=1,narr
         read(1,*,end=666) x1,x2,i3
c         read(1,*,end=666) x1,x2
         n=n+1
         if(i3.eq.1) ang=ang1
         if(i3.eq.2) ang=ang2
         if(i3.eq.3) ang=ang1
         if(i3.eq.1) scale=scale1
         if(i3.eq.2) scale=scale2
         if(i3.eq.3) scale=scale1
         x1=x1*scale
         x2=x2*scale
         x1r= x1*cos(ang)+x2*sin(ang)
         y1r=-x1*sin(ang)+x2*cos(ang)
         x(n)=x1r
         y(n)=y1r
         if(i3.eq.1) i3=17
         if(i3.eq.2) i3=21
         if(i3.eq.3) i3=14
         ic(n)=i3
      enddo
 666  continue
      close(1)

      sumx0=0.1
      sumy0=0.
      sumax=0.
      sumay=0.
      ns=0
      do i=1,n,3
         sumx=0.
         sumy=0.
         do j=1,3
            sumx=sumx+x(i+j-1)
            sumy=sumy+y(i+j-1)
         enddo
         sumx=sumx/3.-0.81
         sumy=sumy/3.-0.0
         ns=ns+1
         sumax=sumax+sumx
         sumay=sumay+sumy

         xdiff=sumx
         ydiff=sumy
c         xdiff=0.
c         ydiff=0.
         do j=1,3
            x(i+j-1)=x(i+j-1)-xdiff
            y(i+j-1)=y(i+j-1)-ydiff
         enddo
      enddo
c      print *,sumax/float(ns),sumay/float(ns)

      open(unit=11,file='out',status='unknown')
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      do i=1,n,3
c         print *,i,ic(i)
         call pgsci(1)
         call pgpt1(x(i),y(i),ic(i))
         call pgsci(2)
         call pgpt1(x(i+1),y(i+1),ic(i))
         call pgsci(4)
         call pgpt1(x(i+2),y(i+2),ic(i))
         write(11,*) x(i),y(i)
         write(11,*) x(i+1),y(i+1)
         write(11,*) x(i+2),y(i+2)
      enddo
      close(11)
      call pgsci(1)
      call pgpt1(0.,0.,5)
c      call pgpt1(0.,0.,23)
      r12=1.07
      r13=1.89
      ang=-20.
      x2=r12*cos(ang/57.3)
      y2=r12*sin(ang/57.3)
      ang=32.
      x3=r13*cos(ang/57.3)
      y3=r13*sin(ang/57.3)
c      print *,sqrt((x2-x3)**2+(y2-y3)**2)
      call pgsci(2)
      call pgpt1(1.27,-0.73,5)
c      call pgpt1(x2,y2,23)
      call pgsci(4)
      call pgpt1(1.27,0.73,5)
c      call pgpt1(x3,y3,23)
      call pgsci(1)

 865  continue
      call pgsch(1.5)
      open(unit=11,file='labels.dat',status='old',err=866)
      read(11,*,err=866,end=866) c1
      call pgmtxt('B',2.5,0.5,0.5,c1)
      read(11,*,err=866,end=866) c1
      call pgmtxt('L',2.0,0.5,0.5,c1)
      read(11,*,err=866,end=866) c1
      call pgmtxt('T',1.5,0.5,0.5,c1)
 866  close(11)
      call pgend
      end
