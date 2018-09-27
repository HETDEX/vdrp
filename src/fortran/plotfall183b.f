
      parameter (radtodeg=57.29578)
      real xp(100000),yp(100000),xo(1000)
      character a1*1,ctrack*1

      ncol=105
      ncol2=13
      cenr0=20.
      cend0=0.
c      cenr=6.25
      cenr=8.5
      cenr2=33.5
      cend=0.0

      xstep=0.27
      xoff=-0.24
      yoff=-0.15

      xoff2=0.
      yoff2=+0.15
      yoff3=-0.01
      yoff4=0.32

      xmin=8.5
      xmax=36.5
      xmin=8.2
      xmax=9.5

      ymin=-0.55
      ymax=0.55
      rat=(ymax-ymin)/(xmax-xmin)
    
c      np=10
      np=1
      xmin0=xmin
      xrange=(xmax-xmin)/float(np)
      xmax0=xmin+xrange
c      call pgbegin(0,'?',1,10)
c      call pgbegin(0,'?',1,np)
c      call pgbegin(0,'?',1,2)
      call pgbegin(0,'?',1,1)
c      call pgpap(0.,1)
      call pgscf(2)
      call pgsch(1.5)
      open(unit=13,file="out3",status='unknown')
      open(unit=14,file="out4",status='unknown')
      nf=0
      do ip=1,np
         xmin=xmin0+float(ip-1)*xrange
         xmax=xmin+xrange
         call pgsci(1)
         call pgenv(xmin,xmax,ymin,ymax,0,0)
         call pglabel('RA(deg)','DEC','')
         
         open(unit=1,file='fallpos.dat',status='old')
         nf=nf+1
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
            write(13,1301) xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),
     $           xp(4),yp(4),"0",nf
         enddo
 669     continue
         close(1)

         call pgsci(2)
         open(unit=1,file='E.out',status='old')
         open(unit=2,file='E183.use',status='old')
         do j=1,10000
            read(2,*,end=866) x1,x2
            xadd=x1-cenr0
            nf=nf+1
            do i=1,100000
               read(1,*,end=670) x1,x2,x3,x4,x5,x6,x7,x8
               xp(1)=x1+xadd
               yp(1)=x2+yoff
               xp(2)=x3+xadd
               yp(2)=x4+yoff
               xp(3)=x5+xadd
               yp(3)=x6+yoff
               xp(4)=x7+xadd
               yp(4)=x8+yoff
               call pgpoly(4,xp,yp)
               write(13,1301) xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),
     $              xp(4),yp(4),"1",nf
            enddo
 670        continue
            rewind(1)
         enddo
 866     continue
         close(2)
         close(1)

         call pgsci(4)
         open(unit=1,file='W.out',status='old')
         open(unit=2,file='W183.use',status='old')
         do j=1,10000
            read(2,*,end=867) x1,x2
            nf=nf+1
            xadd=x1-cenr0
            do i=1,100000
               read(1,*,end=671) x1,x2,x3,x4,x5,x6,x7,x8
               xp(1)=x1+xadd
               yp(1)=x2+yoff2
               xp(2)=x3+xadd
               yp(2)=x4+yoff2
               xp(3)=x5+xadd
               yp(3)=x6+yoff2
               xp(4)=x7+xadd
               yp(4)=x8+yoff2
               call pgpoly(4,xp,yp)
               write(13,1301) xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),
     $              xp(4),yp(4),"1",nf
            enddo
 671        continue
            rewind(1)
         enddo
 867     continue
         close(2)
         close(1)

c - this is extending the center
         call pgsci(3)
         open(unit=1,file='W.out',status='old')
         do j=1,ncol2
            nf=nf+1
            cent=cenr2+xstep*float(j-1)
            xadd=cent-cenr0+xoff2
            write(14,*) cenr0+xadd,cend0+yoff3,1
            call pgpt1(cenr0+xadd,cend0+yoff3,17)
            do i=1,100000
               read(1,*,end=672) x1,x2,x3,x4,x5,x6,x7,x8
               xp(1)=x1+xadd
               yp(1)=x2+yoff3
               xp(2)=x3+xadd
               yp(2)=x4+yoff3
               xp(3)=x5+xadd
               yp(3)=x6+yoff3
               xp(4)=x7+xadd
               yp(4)=x8+yoff3
               call pgpoly(4,xp,yp)
               write(13,1301) xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),
     $              xp(4),yp(4),"1",nf
            enddo
 672        continue
            rewind(1)
         enddo
         close(1)

c - this is extending the bottow east
         call pgsci(5)
         open(unit=1,file='E.out',status='old')
         do j=1,ncol2-1
            nf=nf+1
            cent=cenr2+xstep*float(j-1)
            xadd=cent-cenr0+0.32
            write(14,*) cenr0+xadd,cend0+yoff,0
            call pgpt1(cenr0+xadd,cend0+yoff,17)
            do i=1,100000
               read(1,*,end=674) x1,x2,x3,x4,x5,x6,x7,x8
               xp(1)=x1+xadd
               yp(1)=x2+yoff
               xp(2)=x3+xadd
               yp(2)=x4+yoff
               xp(3)=x5+xadd
               yp(3)=x6+yoff
               xp(4)=x7+xadd
               yp(4)=x8+yoff
               call pgpoly(4,xp,yp)
               write(13,1301) xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),
     $              xp(4),yp(4),"1",nf
            enddo
 674        continue
            rewind(1)
         enddo
         close(1)

c - this is extending the top west
         call pgsci(6)
         open(unit=1,file='W.out',status='old')
         do j=1,ncol2-1
            nf=nf+1
            cent=cenr2+xstep*float(j-1)
            xadd=cent-cenr0+0.32
            write(14,*) cenr0+xadd,cend0+yoff2,1
            call pgpt1(cenr0+xadd,cend0+yoff2,17)
            do i=1,100000
               read(1,*,end=675) x1,x2,x3,x4,x5,x6,x7,x8
               xp(1)=x1+xadd
               yp(1)=x2+yoff2
               xp(2)=x3+xadd
               yp(2)=x4+yoff2
               xp(3)=x5+xadd
               yp(3)=x6+yoff2
               xp(4)=x7+xadd
               yp(4)=x8+yoff2
               call pgpoly(4,xp,yp)
               write(13,1301) xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),
     $              xp(4),yp(4),"1",nf
            enddo
 675        continue
            rewind(1)
         enddo
         close(1)

c- this is a new top row west
         call pgsci(7)
         open(unit=1,file='W.out',status='old')
         do j=1,ncol+1
            nf=nf+1
            cent=cenr+xstep*float(j-1)
            xadd=cent-cenr0-0.02
            write(14,*) cenr0+xadd,cend0+yoff4,1
            call pgpt1(cenr0+xadd,cend0+yoff4,17)
            do i=1,100000
               read(1,*,end=673) x1,x2,x3,x4,x5,x6,x7,x8
               xp(1)=x1+xadd
               yp(1)=x2+yoff4
               xp(2)=x3+xadd
               yp(2)=x4+yoff4
               xp(3)=x5+xadd
               yp(3)=x6+yoff4
               xp(4)=x7+xadd
               yp(4)=x8+yoff4
               call pgpoly(4,xp,yp)
               write(13,1301) xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),
     $              xp(4),yp(4),"1",nf
            enddo
 673        continue
            rewind(1)
         enddo
         close(1)

c- this is adding special shots
         call pgsci(9)
         open(unit=2,file='add.183',status='old')
         do j=1,10000
            read(2,*,end=677) xnew,ynew,ctrack
            nf=nf+1
            if(ctrack.eq."E") then
               open(unit=1,file='E.out',status='old')
               itrack=0
            else
               open(unit=1,file='W.out',status='old')
               itrack=1
            endif
            xadd=xnew-cenr0
            yadd=ynew-cend0
            write(14,*) cenr0+xadd,cend0+yadd,itrack
            call pgpt1(cenr0+xadd,cend0+yadd,17)
            do i=1,100000
               read(1,*,end=676) x1,x2,x3,x4,x5,x6,x7,x8
               xp(1)=x1+xadd
               yp(1)=x2+yadd
               xp(2)=x3+xadd
               yp(2)=x4+yadd
               xp(3)=x5+xadd
               yp(3)=x6+yadd
               xp(4)=x7+xadd
               yp(4)=x8+yadd
               call pgpoly(4,xp,yp)
               write(13,1301) xp(1),yp(1),xp(2),yp(2),xp(3),yp(3),
     $              xp(4),yp(4),"1",nf
            enddo
 676        continue
            rewind(1)
         enddo
 677     continue
         close(2)
         close(1)
      enddo
      close(13)
      close(14)

 1301 format(8(1x,f10.6),1x,a1,1x,i4)

      call pgend
      end

