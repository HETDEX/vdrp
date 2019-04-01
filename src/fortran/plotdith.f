
      parameter(narr=10000,radtodeg=57.29578)
      real x(narr),y(narr),sig(narr),xsig(narr),ysig(narr)
      real xin(narr),yin(narr)
      integer ic(narr)
      character c1*40,c2*40,a1*8,a2*3,an1(narr)*8,an2(narr)*3

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
         read(1,*,end=666) x1,x2,x3,x4,x5,x6,a1,a2
         n=n+1
         x(n)=x1
         y(n)=x2
         ic(n)=1
         an1(n)=a1
         an2(n)=a2
         n=n+1
         x(n)=x3
         y(n)=x4
         ic(n)=2
         an1(n)=a1
         an2(n)=a2
         n=n+1
         x(n)=x5
         y(n)=x6
         ic(n)=3
         an1(n)=a1
         an2(n)=a2
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
c         write(11,*) x(i),y(i)
c         write(11,*) x(i+1),y(i+1)
c         write(11,*) x(i+2),y(i+2)
         write(11,1101) x(i),y(i),x(i+1),y(i+1),x(i+2),y(i+2),
     $        an1(i),an2(i)
      enddo
      close(11)
 1101 format(6(f7.3,1x),a8,1x,a3)
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
