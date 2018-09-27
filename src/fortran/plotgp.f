
      parameter(nmax=10000)
      real t1(nmax),fl1(nmax),fw1(nmax)
      real t2(nmax),fl2(nmax),fw2(nmax)
      real t3(nmax),fl3(nmax),fw3(nmax)
      real xin(nmax),xl(nmax),yl(nmax)
      character c1*20,c2*30

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)

      open(unit=1,file='dpin',status='old')
      read(1,*) x1,x2,x3,x4,x5,x6,x7,x8,x9
      close(1)
      open(unit=1,file='title',status='old')
      read(1,*) c1
      close(1)
      xcen=x4
      ycen=x5
      if(x6.gt.10) then
         xcen=(x4+x6)/2.
         ycen=(x5+x7)/2.
      endif
      if(x8.gt.10) then
         xcen=(x4+x6+x8)/3.
         ycen=(x5+x7+x9)/3.
      endif
      xside=8.
      xmin=xcen-xside
      xmax=xcen+xside
      ymin=ycen-xside
      ymax=ycen+xside

      call pgsch(0.8)
      call pgvport(0.10,0.50,0.60,0.98)
      call pgwindow(xmin,xmax,ymin,ymax)
      call pgbox('bcnst',0.,0,'bcnst',0.,0)
      call pgmtxt('B',2.0,0.5,0.5,'x (pixels)')
      call pgmtxt('L',2.0,0.5,0.5,'y (pixels)')
      call pgsch(1.3)
      call pgmtxt('T',-5.0,1.4,0.5,c1)
      call pgsch(1.0)

      open(unit=1,file='e01in',status='old')
      call pgsci(1)
      n1=0
      n1p=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,x5
         if(x4.lt.xmin.or.x4.gt.xmax.or.x5.lt.ymin.or.x5.gt.ymax) 
     $        n1p=n1p+1
         x4=max(xmin,x4)
         x4=min(xmax,x4)
         x5=max(ymin,x5)
         x5=min(ymax,x5)
         call pgpt1(x4,x5,17)
         n1=n1+1
         t1(n1)=x2*60.*24.
         fw1(n1)=x1
         fl1(n1)=x3
         xin(n1)=x3
      enddo
 666  continue
      close(1)

 2001 format(i3)
      c2="N_exp=    N_off=   "
      write(c2(7:9),2001) n1
      write(c2(17:19),2001) n1p
      call pgsch(1.3)
      call pgmtxt('T',-6.2,1.1,0.,c2)
      call pgsch(1.0)
      if(n1p.gt.0) print *,"CHECK DITHER in 1: ",n1p

      n2=0
      n2p=0
      open(unit=1,file='e02in',status='old')
      call pgsci(2)
      do i=1,nmax
         read(1,*,end=667) x1,x2,x3,x4,x5
         if(x4.lt.xmin.or.x4.gt.xmax.or.x5.lt.ymin.or.x5.gt.ymax) 
     $        n2p=n2p+1
         x4=max(xmin,x4)
         x4=min(xmax,x4)
         x5=max(ymin,x5)
         x5=min(ymax,x5)
         call pgpt1(x4,x5,17)
         n2=n2+1
         t2(n2)=x2*60.*24.
         fw2(n2)=x1
         fl2(n2)=x3
      enddo
 667  continue
      close(1)

      c2="N_exp=    N_off=   "
      write(c2(7:9),2001) n2
      write(c2(17:19),2001) n2p
      call pgsch(1.3)
      call pgmtxt('T',-7.2,1.1,0.,c2)
      call pgsch(1.0)
      if(n2p.gt.0) print *,"CHECK DITHER in 2: ",n2p

      n3=0
      n3p=0
      open(unit=1,file='e03in',status='old')
      call pgsci(4)
      do i=1,nmax
         read(1,*,end=668) x1,x2,x3,x4,x5
         if(x4.lt.xmin.or.x4.gt.xmax.or.x5.lt.ymin.or.x5.gt.ymax) 
     $        n3p=n3p+1
         x4=max(xmin,x4)
         x4=min(xmax,x4)
         x5=max(ymin,x5)
         x5=min(ymax,x5)
         call pgpt1(x4,x5,17)
         n3=n3+1
         t3(n3)=x2*60.*24.
         fw3(n3)=x1
         fl3(n3)=x3
      enddo
 668  continue
      close(1)
      c2="N_exp=    N_off=   "
      write(c2(7:9),2001) n3
      write(c2(17:19),2001) n3p
      call pgsch(1.3)
      call pgmtxt('T',-8.2,1.1,0.,c2)
      call pgsch(1.0)
      if(n3p.gt.0) print *,"CHECK DITHER in 3: ",n3p
      open(unit=11,file='plotgp.out',status='unknown')
      write(11,*) n1,n1p,n2,n2p,n3,n3p
      close(11)

      call pgsci(1)

      open(unit=1,file='compin',status='old')
      read(1,*) x1,x2,x3
      close(1)
      xmin=0.
      xmax=6.1
      ymin=0.9
      ymax=2.4
      xl(1)=xmin
      xl(2)=xmax
      call pgvport(0.55,0.95,0.10, 0.5)
      call pgwindow(xmin,xmax,ymin,ymax)
      call pgbox('bcnst',0.,0,'bcnst',0.,0)
      call pgmtxt('B',2.0,0.5,0.5,'time (min)')
      call pgmtxt('R',2.0,0.5,0.5,'FWHM (arcsec)')
      call pgsci(1)
      call pgpt(n1,t1,fw1,17)
      call pgslw(3)
      yl(1)=x1
      yl(2)=yl(1)
      call pgline(2,xl,yl)
      call pgsci(2)
      call pgpt(n2,t2,fw2,17)
      yl(1)=x2
      yl(2)=yl(1)
      call pgline(2,xl,yl)
      call pgsci(4)
      call pgpt(n3,t3,fw3,17)
      yl(1)=x3
      yl(2)=yl(1)
      call pgline(2,xl,yl)
      call pgsci(1)
      call pgslw(1)

      call biwgt(xin,n1,xb,xs)
      do i=1,n1
         fl1(i)=fl1(i)/xb
      enddo
      do i=1,n2
         fl2(i)=fl2(i)/xb
      enddo
      do i=1,n3
         fl3(i)=fl3(i)/xb
      enddo

      call pgvport(0.1,0.5,0.10, 0.50)
      xmin=0.
      xmax=6.1
      ymin=0.6
      ymax=1.66
      call pgwindow(xmin,xmax,ymin,ymax)
      call pgbox('bcnst',0.,0,'bcnst',0.,0)
      call pgmtxt('B',2.0,0.5,0.5,'time (min)')
      call pgmtxt('L',2.0,0.5,0.5,'Relative Counts')
      call pgsci(1)
      call pgpt(n1,t1,fl1,17)
      call pgsci(2)
      call pgpt(n2,t2,fl2,17)
      call pgsci(4)
      call pgpt(n3,t3,fl3,17)

      call pgvport(0.55,0.95,0.10, 0.50)

      call pgend
      end
