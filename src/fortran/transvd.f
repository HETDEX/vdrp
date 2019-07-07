
      parameter(nmax=10000)
      real v(nmax),y(nmax),yl(nmax),yh(nmax)
      real y2(nmax),yl2(nmax),yh2(nmax),yout(nmax)
      character file1*40

c      fac=sqrt(25./13.)
c      fac=sqrt(85./54.)
c      fac=1.

      write(*,"('km/s of data and model bins : '$)")
      read *,x1,x2
      fac=sqrt(x2/x1)

 1    write(*,"('Vel data file : '$)")
      read *,file1
      open(unit=1,file=file1,status='old',err=1)
      write(*,"('Vel to add : '$)")
      read *,voff
      write(*,"('Symmetrize? (1-yes) : '$)")
      read *,isym
      write(*,"('Invert? (1-yes) : '$)")
      read *,ivert

      n=0
      sum=0.
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4
c         read(1,*,end=666) x1,x2
         n=n+1
         v(n)=x1+voff
         y(n)=x2
         yl(n)=(x2-x3)/fac
         yh(n)=(x4-x2)/fac
         sum=sum+y(n)
      enddo
 666  continue
      close(1)

      do i=1,n
         y(i)=y(i)/sum
         yl(i)=yl(i)/sum
         yh(i)=yh(i)/sum
      enddo

      if(isym.eq.1) then
         call getfwhm(n,v,y,0.5,fwhm,vmax,ymax)
         do i=1,n
            v(i)=v(i)-vmax
         enddo
      endif

      if(ivert.eq.1) then
         call invert(v,y,n,0.,yout)
         do i=1,n
            y(i)=yout(i)
         enddo
         call invert(v,yl,n,0.,yout)
         do i=1,n
            yl(i)=yout(i)
         enddo
         call invert(v,yh,n,0.,yout)
         do i=1,n
            yh(i)=yout(i)
         enddo
      endif

      open(unit=1,file='transvd.out',status='unknown')
      
      ntot=1000
      yhpold=1.e-1
      do i=1,ntot
         vp=v(1)-2.*v(1)/float(ntot-1)*(i-1)
         if(vp.ge.v(1).and.vp.le.v(n)) then
            call xlinint(vp,n,v,y,yp)
            call xlinint(vp,n,v,yl,ylp)
            call xlinint(vp,n,v,yh,yhp)
         else
            yp=0.
            ylp=0.
            yhp=yhpold
         endif
c         yn=max(0.,yp)
c         ynl=max(0.,yn-abs(ylp))
c         ynh=max(0.,yn+abs(yhp))
         yn=yp
         ynl=yn-abs(ylp)
         ynh=yn+abs(yhp)
         if(isym.eq.1) then
            if(-vp.ge.v(1).and.-vp.le.v(n)) then
               call xlinint(-vp,n,v,y,yp2)
               call xlinint(-vp,n,v,yl,ylp2)
               call xlinint(-vp,n,v,yh,yhp2)
            else
               yp2=0.
               ylp2=0.
               yhp2=yhpold
            endif
c            yn2=max(0.,yp2)
c            ynl2=max(0.,yn2-abs(ylp2))
c            ynh2=max(0.,yn2+abs(yhp2))
            yn2=yp2
            ynl2=yn2-abs(ylp2)
            ynh2=yn2+abs(yhp2)
            yn=(yn+yn2)/2.
            ynl=(ynl+ynl2)/2.
            ynh=(ynh+ynh2)/2.
         endif
         yhpold=yhp
         write(1,*) vp,yn,ynl,ynh
      enddo

      close(1)

      end

      subroutine invert(x,y,n,xinv,yout)
      real x(n),y(n),yout(n)

      xmin=xinv-x(n)
      xmax=xinv-x(1)
      do i=1,n
         xp=x(i)
         if(xp.lt.xmin.or.xp.gt.xmax) then
c            yout(i)=-666.
            yout(i)=0.
            goto 666
         endif
         xp=-xp
         do j=1,n-1
            if(xp.ge.x(j).and.xp.le.x(j+1)) then
               yout(i)=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
            endif
         enddo
 666     continue
      enddo
      return
      end
      subroutine getfwhm(n,x,y,frac,fwhm,xmax2,ymax2)
      real x(n),y(n),y2(10000)

      data big /1.e20/

      nbig=500
      if(n.le.nbig) call spline(x,y,n,0.,0.,y2)

      ymax=-big
      ymax2=-big
      do i=1,n-1
         if(n.le.nbig) then
            do ia=1,9
               xp=x(i)+float(ia-1)/9.*(x(i+1)-x(i))
               call splint(x,y,y2,n,xp,yp)
               if(yp.gt.ymax2) then
                  ymax2=yp
                  xmax2=xp
               endif
            enddo
         endif
         if(y(i).gt.ymax) then
            ymax=y(i)
            xmax=x(i)
            imax=i
         endif
      enddo

      if(n.gt.nbig) then
         xmax2=xmax
         ymax2=ymax
      endif
      
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

      subroutine xlinint(xp,n,x,y,yp)
      real x(n),y(n)
      do j=1,n-1
         if(xp.ge.x(j).and.xp.lt.x(j+1)) then
            yp=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
            return
         endif
      enddo
      if(xp.lt.x(1)) yp=y(1)
      if(xp.gt.x(n)) yp=y(n)
      return
      end
