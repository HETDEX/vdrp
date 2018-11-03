
      parameter (radtodeg=57.29578)
      real xp(100000),yp(100000),xo(1000),xl(100),yl(100)
      real yl2(100)
      character a1*1,file1*100

      xmax=155.
      xmin=236.
      ymin=45.
      ymax=58.
      rat=(ymax-ymin)/(xmin-xmax)
      print *,1./rat
      
      np=1
      xmin0=xmin
      xrange=(xmax-xmin)/float(np)
      xmax0=xmin+xrange
      call pgbegin(0,'?',1,np)
      call pgpap(0.,0.5)
      call pgscf(2)
      call pgsch(2.0)
      nline=100
      xmid=(xmax+xmin)/2.
      do i=1,nline
         xl(i)=xmin+(xmax-xmin)*float(i-1)/float(nline-1)
         yl(i)=53.-(xmid-xl(i))**2/(15.)**2
         yl2(i)=51.-(xmid-xl(i))**2/(15.)**2
      enddo
c      call pgsch(2.5)
      do ip=1,np
         xmin=xmin0+float(ip-1)*xrange
         xmax=xmin+xrange
         print *,xmin,xmax
c         call pgpap(0.,rat)
         call pgenv(xmin,xmax,ymin,ymax,0,0)
         call pglabel('RA(deg)','DEC','')
         call pgline(nline,xl,yl)
         call pgline(nline,xl,yl2)
         
c         open(unit=1,file='j3',status='old')
         open(unit=1,file='springall.dat',status='old')
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
 669     continue
         close(1)

         call pgsci(2)
         open(unit=2,file='list',status='old')
         do j=1,10000
            read(2,*,end=666) file1
            open(unit=1,file=file1,status='old')
            do i=1,100000
               read(1,*,end=670) x1,x2,x3,x4,x5,x6,x7,x8
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
 670        continue
            close(1)
         enddo
 666     continue
         close(2)

         call pgsci(4)
         open(unit=2,file='listadd',status='old')
         do j=1,10000
            read(2,*,end=665) file1
            open(unit=1,file=file1,status='old')
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
               call pgpoly(4,xp,yp)
            enddo
 671        continue
            close(1)
         enddo
 665     continue
         close(2)

      enddo

      call pgend
      end

