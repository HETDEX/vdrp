
      parameter (radtodeg=57.29578)
      real xp(100000),yp(100000),xo(1000)
      real xa(4,78),ya(4,78)
      character a1*1,file1*100

      xcen=193.
      ycen=51.1
      xcen=22.
      ycen=0.
      radd=0.8
      radr=3.
      ymin=ycen-radd
      ymax=ycen+radd
      xmin=xcen-radr
      xmax=xcen+radr
      
      call pgbegin(0,'?',1,np)
      call pgpap(0.,0.4)
      call pgscf(2)
      call pgsch(2.0)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('RA(deg)','DEC','')
         
      open(unit=1,file='ifuhdr1.dat',status='old')
      do i=1,100000
         read(1,*,end=669) x1,x2,x3,x4,x5,x6,x7,x8
         xp(1)=x1
         yp(1)=x2
         xp(2)=x3
         yp(2)=x4
         xp(3)=x5
         yp(3)=x6
         xp(4)=x7
         yp(4)=x8
         call pgpoly(4,xp,yp)
      enddo
 669  continue
      close(1)

      xall=22.499768
      yall=0.166609

      open(unit=1,file='all.dat',status='old')
      call pgsci(2)
      do i=1,100000
         read(1,*,end=666) x1,x2,x3,x4,x5,x6,x7,x8
         xa(1,i)=x1-xall
         ya(1,i)=x2-yall
         xa(2,i)=x3-xall
         ya(2,i)=x4-yall
         xa(3,i)=x5-xall
         ya(3,i)=x6-yall
         xa(4,i)=x7-xall
         ya(4,i)=x8-yall
      enddo
 666  continue
      close(1)

      open(unit=1,file='listin',status='old')

      do j=1,100000
         read(1,*,end=667) x1,x2
         do i=1,78
            xp(1)=xa(1,i)+x1
            yp(1)=ya(1,i)+x2
            xp(2)=xa(2,i)+x1
            yp(2)=ya(2,i)+x2
            xp(3)=xa(3,i)+x1
            yp(3)=ya(3,i)+x2
            xp(4)=xa(4,i)+x1
            yp(4)=ya(4,i)+x2
c            call pgpoly(4,xp,yp)
         enddo
      enddo
 667  continue
      close(1)

      call pgend
      end

