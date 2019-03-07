
      parameter(nmax=200000)
      real x(nmax),y(nmax),sig(nmax),xsig(nmax),ysig(nmax)
      real xin(nmax),yin(nmax),xl(2),yl(2)
      integer ic(10000)
      character c1*40,c2*40,a16*12

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
      call pgslw(2)

      xoff=8.5
      xoff=0.
      yoff=0.
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,
     $        x13,x14,x15,a16,i17,i18
         n=n+1
         x(n)=float(i17)
         y(n)=float(i18)
         ic(n)=1
c         if(x4>5) ic(n)=2
c         if(x7>5) ic(n)=3
c         if(x11<5.6) ic(n)=5
c         if(x13>2.2) ic(n)=6
      enddo
 666  continue
      close(1)

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgsls(1)
      call pgsch(1.3)
      do i=1,n
         call pgsci(ic(i))
         call pgpt1(x(i),y(i),17)
      enddo

      call pgsci(1)

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
