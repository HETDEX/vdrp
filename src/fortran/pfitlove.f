
      parameter(nmax=10000)

      real v(nmax),y(nmax),a(5),yherm(nmax),covar(6,6),siga(nmax)
      real yl(nmax),yh(nmax)
      integer ia(6)
      character file1*40
      data big/1.e30/

      parameter(pi=3.1415926539)

 1    call qc1('File of fits ','pallmc.def',file1)
      call savdef

      open(unit=1,file=file1,status='old',err=1)
      open(unit=11,file='pfitlov.out',status='unknown')

 1001 format(7x,i3)

      call pgbegin(0,'?',2,2)
c      call pgbegin(0,'?',1,1)
c      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.2)
c      call pgenv(-200.,1700.,0.,0.14,0,0)

      do if=1,nmax
         read(1,*,end=666) file1
         do i=1,40
            if(file1(i:i).eq.' ') then
               nfile=i-1
               goto 966
            endif
         enddo
 966     continue
         open(unit=2,file=file1(1:nfile),status='old')
         vref=0.
         nl=0
         sum=0.
         do i=1,nmax
            read(2,*,end=667) x1,x2,x3,x4
            nl=nl+1
            v(nl)=x1+vref
            y(nl)=x2
            yl(nl)=x3
            yh(nl)=x4
            siga(nl)=1.
            sum=sum+y(nl)
         enddo
 667     continue
         close(2) 
c         if(if.eq.1) sum=sum*29./1000./1.03
c         print *,sum
         do i=1,nl
            y(i)=y(i)/sum
            yl(i)=yl(i)/sum
            yh(i)=yh(i)/sum
         enddo
         call getfwhm(nl,v,y,0.5,fwhm,vmax,ymax)
         a(1)=ymax*sqrt(2.*pi)*fwhm/2.35
         a(2)=vmax
         a(3)=fwhm/2.35
c         print *,a(3)
         a(4)=0.
         a(5)=0.
         ia(1)=1
         ia(2)=1
         ia(3)=1
         ia(4)=0
         ia(5)=0
         ia(4)=1
         ia(5)=1
         ia(6)=0
         call fithermec(nl,v,y,siga,a,ia,6,covar)
         sherm=a(3)
         vherm=a(2)
         eh1=sqrt(covar(2,2))
         eh2=sqrt(covar(3,3))
         eh3=sqrt(covar(4,4))
         eh4=sqrt(covar(5,5))
c         print *,file1(1:nfile),vherm,eh1,sherm,eh2,a(4),eh3,a(5),eh4
         print *,file1(1:nfile),vherm,sherm,a(4),a(5)
c         if(if.eq.1) call kgenv(nl,v,yh)
         call kgenv(nl,v,yh)
         call pglabel('','',file1(1:nfile))
c         call pgsci(if)
         call pgline(nl,v,y)
         call pgsls(4)
c         if(if.eq.1) then
            call pgline(nl,v,yl)
            call pgline(nl,v,yh)
c         endif
         call pgsls(1)
         amp=a(1)
         vel=a(2)
         sig=a(3)
         h3=a(4)
         h4=a(5)
         do i=1,nl
            w=(v(i)-vel)/sig
            gaus=exp(-w*w/2.)/sqrt(2.*pi)
            yherm(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))
         enddo
         call pgsci(2)
c         call pgline(nl,v,yherm)
         call pgsci(1)
         write(11,*) file1(1:nfile),vherm,sherm,h3,h4
      enddo
 666  continue
      close(1)

      call pgend

      end

      subroutine getfwhm(n,x,y,frac,fwhm,xmax2,ymax2)
      real x(n),y(n),y2(10000)

      data big /1.e20/

c      call spline(x,y,n,0.,0.,y2)

      ymax=-big
      ymax2=-big
      do i=1,n-1
         do ia=1,9
            xp=x(i)+float(ia-1)/9.*(x(i+1)-x(i))
c            call splint(x,y,y2,n,xp,yp)
            if(yp.gt.ymax2) then
               ymax2=yp
               xmax2=xp
            endif
         enddo
         if(y(i).gt.ymax) then
            ymax=y(i)
            imax=i
         endif
      enddo

      ymax2=ymax
      xmax2=x(imax)
      yhalf=ymax2*frac

      diff=big
      x1=x(1)
      do i=1,imax-1
         if(yhalf.ge.y(i).and.yhalf.lt.y(i+1)) then
            x1=x(i)+(yhalf-y(i))/(y(i+1)-y(i))*(x(i+1)-x(i))
         endif
      enddo

      diff=big
      x2=x(n)
      do i=imax,n-1
         if(yhalf.ge.y(i+1).and.yhalf.lt.y(i)) then
            x2=x(i+1)+(yhalf-y(i+1))/(y(i)-y(i+1))*(x(i)-x(i+1))
         endif
      enddo

      fwhm=x2-x1

      return
      end

      subroutine getvell(n,x,y,xmax,ymax,xlow,xhigh)
      real x(n),y(n)

      do i=1,n-1
         if(ymax.gt.y(i).and.ymax.le.y(i+1)) then
            xlow=x(i)+(x(i+1)-x(i))*(ymax-y(i))/(y(i+1)-y(i))
         endif
         if(ymax.le.y(i).and.ymax.gt.y(i+1)) then
            xhigh=x(i)+(x(i+1)-x(i))*(ymax-y(i))/(y(i+1)-y(i))
         endif
      enddo
      return
      end

      subroutine kgenv(n,x,y)
      real x(n),y(n)
 
      data big/1.e20/
 
      xmin=big
      xmax=-big
      ymin=big
      ymax=-big
      do i=1,n
         xmin=min(xmin,x(i))
         xmax=max(xmax,x(i))
         ymin=min(ymin,y(i))
         ymax=max(ymax,y(i))
      enddo
 
      xbit=(xmax-xmin)/15.
      ybit=(ymax-ymin)/15.
      if(ymax.eq.ymin) ybit=1.e-3
c      call pgenv(xmin-xbit,xmax+xbit,ymin-ybit,ymax+ybit,0,0)
      call pgenv(xmin-xbit,xmax+xbit,-0.001,ymax+ybit,0,0)
 
      return
      end
