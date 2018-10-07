
      parameter(nmax=100000)

      real v(nmax),y(nmax),sigy(nmax),a(6),yherm(nmax),covar(6,6)
      integer ia(6)
      character file1*50
      data big/1.e30/

      parameter(pi=3.1415926539)

 1    call qc1('File of fits ','pallmc.def',file1)
      call savdef

      open(unit=1,file=file1,status='old',err=1)
      open(unit=11,file='pfitlov.out',status='unknown')

 1001 format(7x,i3)

      call pgbegin(0,'?',2,2)
c      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.2)
c      call pgenv(-200.,1700.,0.,0.14,0,0)

      do i=1,6
         ia(i)=1
      enddo
      ia(6)=0

c      ia(4)=0
c      ia(5)=0

      do if=1,nmax
         read(1,*,end=666) file1
         do i=1,40
            if(file1(i:i).eq.' ') then
               nfile=i-1
               goto 966
            endif
         enddo
 966     continue
         vref=0
         open(unit=2,file=file1(1:nfile),status='old')
         read(2,*) vref,x2,x3,ntot
         nl=0
         do i=1,nmax
            read(2,*,end=667) i1,x2,x3
c            read(2,*,end=667) x2,x3
            nl=nl+1
            v(nl)=x2+vref
            y(nl)=x3
            sigy(nl)=.01
         enddo
 667     continue
         close(2)
         call getfwhm(nl,v,y,0.5,fwhm,vmax,ymax,v1,v2)
         a(1)=ymax*sqrt(2.*pi)*fwhm/2.35
         a(2)=vmax
         a(3)=fwhm/2.35
         a(4)=0.
         a(5)=0.
         a(6)=0.
         call fithermec(nl,v,y,sigy,a,ia,6,covar)
         sherm=a(3)
         vherm=a(2)
         eh1=sqrt(covar(2,2))
         eh2=sqrt(covar(3,3))
         eh3=sqrt(covar(4,4))
         eh4=sqrt(covar(5,5))
c         if(if.eq.1) call kgenv(nl,v,y)
c         call pgsci(if)
         call kgenv(nl,v,y)
         call pgline(nl,v,y)
         amp=a(1)
         vel=a(2)
         sig=a(3)
         h3=a(4)
         h4=a(5)
         do i=1,nl
            w=(v(i)-vel)/sig
            gaus=exp(-w*w/2.)/sqrt(2.*pi)
            yherm(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))
c            print *,v(i),y(i),yherm(i)
         enddo
         call pgsci(3)
         call pgline(nl,v,yherm)
         call pgsci(1)
         write(*,1102) file1(1:nfile),vherm,sherm,fwhm/2.35,h3,h4,v1,v2
     $        ,ntot
         write(11,1102) file1(1:nfile),vherm,sherm,fwhm/2.35,h3,h4,v1,v2
     $        ,ntot
c         write(11,1101) file1(1:nfile),vherm,sherm,h3,h4
      enddo
 666  continue
      close(1)

 1101 format(1x,a25,4(1x,f10.3))
 1102 format(1x,a25,7(1x,f9.3),1x,i5)

      call pgend

      end

      subroutine getfwhm(n,x,y,frac,fwhm,xmax2,ymax2,v1,v2)
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

c - get 1st and second moment:

      sum1=0.
      sum2=0.
      do i=1,n
         sum1=sum1+x(i)*y(i)
         sum2=sum2+y(i)
      enddo
      v1=sum1/sum2
      sum1=0.
      sum2=0.
      do i=1,n
         sum1=sum1+y(i)*(x(i)-v1)**2
         sum2=sum2+y(i)
      enddo
      v2=sqrt(sum1/sum2)

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
      call pgenv(xmin-xbit,xmax+xbit,ymin-ybit,ymax+ybit,0,0)
 
      return
      end
