
      parameter (radtodeg=57.29578)
      real xp(1000000),yp(1000000),xo(1000)
      character a1*1,fileh(4)*40

      xcen=269.75
      ycen=66.123
      xsize=0.8
      rat=cos(ycen/57.29)
      ysize=xsize*rat
      ysize=xsize*0.3

      xmin=xcen-xsize/2.
      xmax=xcen+xsize/2.
      ymin=ycen-ysize/2.
      ymax=ycen+ysize/2.
      
      call pgbegin(0,'?',1,1)
c      call pgpap(0.,0.3)
      call pgscf(2)
      call pgsch(1.5)
      print *,xmin,xmax,ymin,ymax
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('RA(deg)','DEC','')

      open(unit=11,file='out',status='unknown')
      open(unit=12,file='out2',status='unknown')
      call pgsch(1.5)

      xcene=xcen
      ycene=ycen

      xoffrow1=0.033
      yoffrow1=0.
      xoffrow2=0.
      yoffrow2=-0.014
      xoffrow3=-0.033
      yoffrow3=0.
      xoff=0

      xout=xcene
      yout=ycene
      write(11,1001) xout,yout,0
      xout=xout+xoffrow1
      yout=yout+yoffrow1
      write(11,1001) xout,yout,0
      xout=xout+xoffrow2
      yout=yout+yoffrow2
      write(11,1001) xout,yout,0
      xout=xout+xoffrow3
      yout=yout+yoffrow3
      write(11,1001) xout,yout,0
      
      call pgsfs(1)
      open(unit=2,file='nep1',status='old')
      do i=1,54
         read(2,*,end=669) x1,x2,x3,x4,x5,x6,x7,x8
         xp(1)=x1+xoff
         yp(1)=x2
         xp(2)=x3+xoff
         yp(2)=x4
         xp(3)=x5+xoff
         yp(3)=x6
         xp(4)=x7+xoff
         yp(4)=x8
         call pgsci(2)
         call pgpoly(4,xp,yp)
         write(12,1201) 
     $        xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
         do j=1,4
            yp(j)=yp(j)+yoffrow1
            xp(j)=xp(j)+xoffrow1
         enddo
         call pgsci(4)
         call pgpoly(4,xp,yp)
         write(12,1201) 
     $        xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
         do j=1,4
            yp(j)=yp(j)+yoffrow2
            xp(j)=xp(j)+xoffrow2
         enddo
         call pgsci(3)
         call pgpoly(4,xp,yp)
         write(12,1201) 
     $        xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
         do j=1,4
            yp(j)=yp(j)+yoffrow3
            xp(j)=xp(j)+xoffrow3
         enddo
         call pgsci(5)
         call pgpoly(4,xp,yp)
         write(12,1201) 
     $        xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),xp(4),yp(4)
      enddo
 669  continue
      close(2)
      close(11)
      close(12)

 1001 format(2(1x,f9.5),1x,i2)
 1201 format(8(1x,f10.6))

      call pgend
      end

