
      parameter (radtodeg=57.29578)
      real xp(100000),yp(100000),xo(1000),xc(5),yc(5)
      real x(1000),y(1000)
      character a1*1,a2*10

      open(unit=1,file='cospos.dat',status='old')
      read(1,*) xc0,yc0
      close(1)
      xch=0.7
      xc(1)=xc0-xch
      yc(1)=yc0-xch
      xc(2)=xc0-xch
      yc(2)=yc0+xch
      xc(3)=xc0+xch
      yc(3)=yc0+xch
      xc(4)=xc0+xch
      yc(4)=yc0-xch
      xc(5)=xc0-xch
      yc(5)=yc0-xch

      xmin=149.3
      xmax=150.9
      ymin=1.45
      ymax=2.95
      xcen=(xmax+xmin)/2.
      ycen=(ymax+ymin)/2.
      xside=0.5
      yside=0.5
c      xside=2.0
c      yside=2.0
      xmin=xcen-xside/2.
      xmax=xcen+xside/2.
      ymin=ycen-yside/2.
      ymax=ycen+yside/2.
      rat=(ymax-ymin)/(xmax-xmin)
      print *,1./rat
      
      np=1
      xmin0=xmin
      xrange=(xmax-xmin)/float(np)
      xmax0=xmin+xrange
      call pgbegin(0,'?',1,np)
      call pgpap(0.,1.0)
      call pgscf(2)
c      call pgsch(2.5)
      do ip=1,np
         xmin=xmin0+float(ip-1)*xrange
         xmax=xmin+xrange
         print *,xmin,xmax
c         call pgpap(0.,rat)
         call pgenv(xmin,xmax,ymin,ymax,0,0)
         call pglabel('RA(deg)','DEC','')
         
         open(unit=1,file='inpos',status='old')
         do i=1,100000
            read(1,*,end=669) x1,x2,x3,x4,x5,x6,x7,x8,a2
            xp(1)=x1
            yp(1)=x2
            xp(2)=x3
            yp(2)=x4
            xp(3)=x5
            yp(3)=x6
            xp(4)=x7
            yp(4)=x8
            call pgsci(1)
            if(a2(1:7).eq."COSDEEP") call pgsci(2)
            call pgpoly(4,xp,yp)
         enddo
 669     continue
         close(1)
      enddo

      xoff=0
      yoff=0
c      xoff=0.117
c      yoff=0.125
c      open(unit=1,file='cosnewpos',status='old')
c      open(unit=1,file='cosnewposb',status='old')
c      open(unit=1,file='cosnewposc',status='old')
      open(unit=1,file='cosnewposd',status='old')
      open(unit=11,file='new.out',status='unknown')
      do ii=1,2
         nifu=176
         if(ii.eq.1) then
            nifu=44
            xoff=-0.0065
            yoff=0.012
         endif
         do i=1,nifu
            read(1,*,end=670) x1,x2,x3,x4,x5,x6,x7,x8,a2
            xp(1)=x1+xoff
            yp(1)=x2+yoff
            xp(2)=x3+xoff
            yp(2)=x4+yoff
            xp(3)=x5+xoff
            yp(3)=x6+yoff
            xp(4)=x7+xoff
            yp(4)=x8+yoff
            call pgsci(4)
            call pgpoly(4,xp,yp)
            write(11,1101) xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),
     $           xp(4),yp(4)
         enddo
 670     continue
         rewind(1)
      enddo
      close(1)
      close(11)
 1101 format(8(1x,f11.5))

      open(unit=1,file='3dhst_pointing-wfc3_cosmos.dat',status='old')
      call pgsfs(2)
      call pgslw(4)
      do i=1,100000
         read(1,*,end=671) x1,x2,x3,x4,x5,x6,x7,x8
         xp(1)=x1
         yp(1)=x2
         xp(2)=x3
         yp(2)=x4
         xp(3)=x5
         yp(3)=x6
         xp(4)=x7
         yp(4)=x8
         call pgsci(6)
         call pgpoly(4,xp,yp)
      enddo
 671  continue
      close(1)
      call pgsci(4)
      call pgsch(2.0)
      call pgslw(4)
c      call pgpt1(xc,yc,17)
      call pgline(5,xc,yc)

      open(unit=11,file='muse.dat',status='old')
      n=0
      do i=1,100000
         read(11,*,end=466) a1,x1,x2,x3,x4,x5,x6
         n=n+1
         x(n)=15*(x1+x2/60.+x3/3600.)
         y(n)=x4+x5/60.+x6/3600.
      enddo
 466  continue
      close(11)

      call pgsci(5)
      do i=1,n
         x1=x(i)-0.5/60.
         x2=x(i)+0.5/60.
         y1=y(i)-0.5/60.
         y2=y(i)+0.5/60.
         call pgrect(x1,x2,y1,y2)
      enddo

      open(unit=11,file='muse_cand.dat',status='old')
      n=0
      do i=1,100000
         read(11,*,end=467) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 467  continue
      close(11)
      do i=1,n
         x1=x(i)-0.5/60.
         x2=x(i)+0.5/60.
         y1=y(i)-0.5/60.
         y2=y(i)+0.5/60.
         call pgrect(x1,x2,y1,y2)
      enddo

      xpos=150.22
      call pgsch(1.5)
      call pgsci(4)
      call pgptxt(xpos,2.11,0.,0.,'DEX_18.3')
      call pgsci(1)
      call pgptxt(xpos,2.08,0.,0.,'DEX_all')
      call pgsci(2)
      call pgptxt(xpos,2.05,0.,0.,'DEX_11')
      call pgsci(6)
      call pgptxt(xpos,2.02,0.,0.,'3dHST')
      call pgsci(5)
      call pgptxt(xpos,1.99,0.,0.,'MUSE')

      call pgend
      end

