
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
      
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.5)
      call pgslw(2)

      xoff=8.5
      xoff=0.
      yoff=0.
      n=0
c      ymax=-1e10
c      ymin=1e10
      do i=1,nmax
         read(1,*,end=666) x1,x2,i3
c         print *,x1,x2,i3
c         if(x2.gt.ymax) x2=ymax
c         ymin=min(ymin,x2)
c         ymax=max(ymax,x2)
c         read(1,*,end=666) x1,x2,x3,x4
c         read(1,*,end=666) c1,x2,x3,x4,x5,x6,x7
         n=n+1
c         x(n)=x2
c         xin(n)=x(n)-xoff
c         y(n)=x5
c         yin(n)=y(n)-yoff
c         xsig(n)=(x4-x3)/2.
c         ysig(n)=(x7-x6)/2.
         x(n)=x1
         xin(n)=x(n)-xoff
         y(n)=x2
         yin(n)=y(n)
         ic(n)=i3
c         xsig(n)=x2/4.
c         ysig(n)=x4
         xsig(n)=0.05
         ysig(n)=0.05
c         call pgerry(1,x(n),y(n)+ysig(n),y(n)-ysig(n),1.)
c         call pgerrx(1,x(n)-xsig(n),x(n)+xsig(n),y(n),1.)
      enddo
 666  continue
      close(1)

      ybit=(ymax-ymin)/10.
c      ymax=ymax+ybit
c      ymin=ymin-ybit
      if(ilog.eq.0) then
         call pgenv(xmin,xmax,ymin,ymax,0,0)
      else
         xmin=log10(xmin)
         xmax=log10(xmax)
         call pgenv(xmin,xmax,ymin,ymax,0,10)
      endif
      call pgsls(1)
c      call pgsch(0.8)
      call pgsch(1.9)
      do i=1,n
c         call pgsci(2)
         call pgsci(ic(i))
c         xp=x(i)-0.1
         xp=x(i)
         if(ic(i).eq.0) then
            call pgsci(4)
c            xp=x(i)+0.1
            xp=x(i)
         endif
         call pgpt1(xp,y(i),17)
      enddo

      call pgsci(1)

      goto 865
      if(ifit.eq.1) then
         call fitexy(xin,yin,n,xsig,ysig,a,b,siga,sigb,chi2,q)

         print *,a,b,siga,sigb
         print *,chi2,q

c      a=-0.015
c      b=0.53

         x(1)=xmin
         x(2)=xmax
         y(1)=a+b*(x(1)-xoff)+yoff
         y(2)=a+b*(x(2)-xoff)+yoff
         call pgline(2,x,y)

         call pgsch(1.2)
 1001    format(a3,f6.3,a4,f6.3,a8)
 1002    format(a3,f6.3,a4,f6.3)
         write(c1,1001) "a= ",a," +- ",siga,"  at -22"
c         call pgmtxt('B',-2.5,0.1,0.,c1)
         write(c1,1002) "b= ",b," +- ",sigb
c         call pgmtxt('B',-1.3,0.1,0.,c1)
      else
         call biwgt(yin,n,a,siga)
         siga=siga/sqrt(float(n-1))
         x(1)=xmin
         x(2)=xmax
         y(1)=a
         y(2)=a
         call pgline(2,x,y)
         call getidisp(n,yin,ysig,dexp,dmeas)
         call pgsch(1.2)
         write(c1,1002) "a= ",a," +- ",siga
         call pgmtxt('B',-3.0,0.1,0.,c1)
         write(c1,1003) "Expected Dispersion= ",dexp
         call pgmtxt('B',-2.0,0.1,0.,c1)
         write(c1,1003) "Measured Dispersion= ",dmeas
         call pgmtxt('B',-0.9,0.1,0.,c1)
 1003    format(a21,f5.2)
      endif

 865  continue

      xl(1)=xmin
      xl(2)=xmax
      yl(1)=ymin
      yl(2)=ymax
      call pgsci(2)
c      call pgline(2,xl,yl)
      call pgsci(1)
c      call pgpt1(0.,0.,5)
      call pgsci(2)
      scale=4.5
      ang=24.
      x1=1.27*scale
      x2=-0.73*scale
      x1n= x1*cos(ang/57.3)+x2*sin(ang/57.3)
      x2n=-x1*sin(ang/57.3)+x2*cos(ang/57.3)
c      call pgpt1(x1n,x2n,5)
      call pgsci(4)
      x1=1.27*scale
      x2=0.73*scale
      x1n= x1*cos(ang/57.3)+x2*sin(ang/57.3)
      x2n=-x1*sin(ang/57.3)+x2*cos(ang/57.3)
c      call pgpt1(x1n,x2n,5)
      call pgsci(2)

      xl(1)=xmin
      xl(2)=xmax
      yl(1)=4.2
      yl(2)=4.2
      call pgslw(3)
      call pgsci(1)
      call pgsls(4)
      call pgline(2,xl,yl)
      call pgsci(1)
      call pgslw(2)
      call pgsls(1)

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

      subroutine getidisp(n,x,sig,sum,siga)
      real x(n),sig(n),xr(200000)
      call biwgt(x,n,a,siga)
      idum=-1
      sum=0.
      do i=1,100
         do j=1,n
            xr(j)=a+gasdev(idum)*sig(j)
         enddo
         call biwgt(xr,n,an,sigan)
         sum=sum+sigan
      enddo
      sum=sum/float(100)
      return
      end
