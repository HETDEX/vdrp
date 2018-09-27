
      parameter(nmax=200000)
      real x(nmax),y(nmax),sig(nmax),xsig(nmax),ysig(nmax)
      real xin(nmax),yin(nmax),xl(2),yl(2)
      integer ic(10000)
      character c1*40,c2*40

      ifit=0
      call qc1('Input datafile ','plotxy.def',c2)
      open(unit=1,file=c2,status='old')
      
      ilog=0
      call qr2('Xmin and Xmax ','plotxy.def',xmin,xmax)
      call qr2('Ymin and Ymax ','plotxy.def',ymin,ymax)
      call savdef
      
      call pgbegin(0,'?',2,2)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.5)

      xoff=0.
      yoff=0.
      do iall=1,4
         read(1,*,end=667) c1
         open(unit=2,file=c1,status='old')
         n=0
         do i=1,nmax
            read(2,*,end=666) x1,x2,i3
c            read(2,*,end=666) x1,x2,i3,x4,x5
c            if(x5.gt.380.and.x5.lt.430) then
               n=n+1
               x(n)=x1
               y(n)=x2
               ic(n)=i3
c            endif
         enddo
 666     continue
         close(2)

         call pgsci(1)
         call pgsch(1.4)
         call pgenv(xmin,xmax,ymin,ymax,0,0)
         call pglabel("Days since June 1","Data-Overscan","")
         call pgsch(1.7)
         call pgmtxt('T',-1.3,0.5,0.5,c1(1:5))
         call pgsls(1)
         call pgsch(2.2)
         do i=1,n
            call pgsci(ic(i))
            xp=x(i)
            call pgpt1(xp,y(i),17)
         enddo
      enddo
 667  continue
 866  close(1)
      call pgend
      end
