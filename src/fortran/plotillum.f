
      parameter(nmax=200000)
      real x1a(nmax),x2a(nmax),x3a(nmax),x4a(nmax),x5a(nmax),x6a(nmax)
      real x7a(nmax),x8a(nmax),x9a(nmax),x10a(nmax),x11a(nmax)
      real x12a(nmax),xl(2),yl(2)
      character c2*40

      call qc1('Input datafile ','plotxy.def',c2)
      open(unit=1,file=c2,status='old')
      
      call qr2('Xmin and Xmax ','plotxy.def',xmin,xmax)
      call qr2('Ymin and Ymax ','plotxy.def',ymin,ymax)
      call savdef
      
      call pgbegin(0,'?',3,3)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.5)

      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12
         n=n+1
         avg1=(x1+x2+x3)/3.
         avg2=(x4+x5+x6)/3.
         avg3=(x7+x8+x9)/3.
         avg4=(x10+x11+x12)/3.
         avg34=(x7+x8+x9+x10+x11+x12)/6.
         x1a(n)=x1/avg1
         x2a(n)=x2/avg1
         x3a(n)=x3/avg1
         x4a(n)=x4/avg2
         x5a(n)=x5/avg2
         x6a(n)=x6/avg2
         x7a(n)=x7/avg3
         x8a(n)=x8/avg3
         x9a(n)=x9/avg3
         x7a(n)=(x7+x10)/avg34/2.
         x8a(n)=(x8+x11)/avg34/2.
         x9a(n)=(x9+x12)/avg34/2.
      enddo
 666  continue
      close(1)

      xl(1)=xmin
      xl(2)=xmax
      yl(1)=ymin
      yl(2)=ymax
      call pgsls(1)
      call pgsch(1.5)

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgline(2,xl,yl)
      call pgpt(n,x1a,x7a,17)
      call pglabel("d1_illum","d1_gp","")

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgline(2,xl,yl)
      call pgpt(n,x2a,x8a,17)
      call pglabel("d2_illum","d2_gp","")

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgline(2,xl,yl)
      call pgpt(n,x3a,x9a,17)
      call pglabel("d3_illum","d3_gp","")

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgline(2,xl,yl)
      call pgpt(n,x4a,x7a,17)
      call pglabel("d1_star","d1_gp","")

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgline(2,xl,yl)
      call pgpt(n,x5a,x8a,17)
      call pglabel("d2_star","d2_gp","")

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgline(2,xl,yl)
      call pgpt(n,x6a,x9a,17)
      call pglabel("d3_star","d3_gp","")

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgline(2,xl,yl)
      call pgpt(n,x1a,x4a,17)
      call pglabel("d1_illum","d1_star","")

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgline(2,xl,yl)
      call pgpt(n,x2a,x5a,17)
      call pglabel("d2_illum","d2_star","")

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgline(2,xl,yl)
      call pgpt(n,x3a,x6a,17)
      call pglabel("d3_illum","d3_star","")

 866  close(11)
      call pgend
      end
