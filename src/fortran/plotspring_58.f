
      parameter (radtodeg=57.29578)
      real xp(100000),yp(100000),xo(1000),xl(100),yl(100)
      real yl2(100)
      character a1*1,file1*100

c      xoff2=0.6
c      yoffw2=-0.3
c      xoffw=-0.3
c      yoffw=0.07
      xoff2=0.38
      yoffw2=-0.45

      xoffw=-0.2
      yoffw=0.26

      xmax=199.5
      xmin=201.6
      ymin=51.6
      ymax=52.3
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
         
         open(unit=2,file='listadd_58',status='old')
         do j=1,2
            read(2,*,end=465) file1
            call pgsci(j+3)
            open(unit=1,file=file1,status='old')
            xoff1=0.
            yoff1=0.
            if(j.eq.2) xoff1=xoffw
            if(j.eq.2) yoff1=yoffw
            do k=1,10
               xoff=xoff1+xoff2*float(k-1)
               yoff=yoff1
               do i=1,100000
                  read(1,*,end=471) x1,x2,x3,x4,x5,x6,x7,x8
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
 471           continue
               rewind(1)
            enddo
            close(1)
         enddo
 465     continue
         close(2)

         open(unit=2,file='listadd_58',status='old')
         do j=1,2
            read(2,*,end=365) file1
            call pgsci(j+3)
            open(unit=1,file=file1,status='old')
            xoff1=0.
            yoff1=yoffw2
            if(j.eq.2) xoff1=xoffw
            if(j.eq.2) yoff1=yoffw+yoffw2
            do k=1,10
               xoff=xoff1+xoff2*float(k-1)
               yoff=yoff1
               do i=1,100000
                  read(1,*,end=371) x1,x2,x3,x4,x5,x6,x7,x8
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
 371           continue
               rewind(1)
            enddo
            close(1)
         enddo
 365     continue
         close(2)

      enddo

      call pgend
      end

