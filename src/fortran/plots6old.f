      parameter(nmax=10000)
      real x(nmax),y(nmax),s(nmax),rad(nmax),rg(nmax)
      real sg(nmax)
      parameter(pi=3.14159)

      fwhm=2.1
      sig=fwhm/2.35

      xmin=1e10
      xmax=-1e10
      ymin=1e10
      ymax=-1e10
      smax=0.
      open(unit=1,file='s6',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3
         n=n+1
         x(n)=x1
         y(n)=x2
         s(n)=x3
         xmin=min(xmin,x(n))
         xmax=max(xmax,x(n))
         ymin=min(ymin,y(n))
         ymax=max(ymax,y(n))
         smax=max(smax,s(n))
      enddo
 666  continue
      close(1)

c      call pgbegin(0,'?',3,3)
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      namp=10
      ampmin=0.
      ampmax=smax*2.
      nx=30
      ny=30
      chiall=1.e10
      xmin=xmin*0.8
      xmax=xmax*0.8
      ymin=ymin*0.8
      ymax=ymax*0.8
      radmax=(xmax-xmin)*0.8
      ng=100
      do i=1,ng
         rg(i)=float(i-1)/float(ng-1)*radmax
      enddo

      do i=1,nx
         xt=xmin+float(i-1)/float(nx-1)*(xmax-xmin)
         do j=1,ny
            yt=ymin+float(j-1)/float(ny-1)*(ymax-ymin)
c            call pgenv(0.,radmax,0.,smax,0,0)
            do ia=1,n
               rad(ia)=sqrt((x(ia)-xt)**2+(y(ia)-yt)**2)
c               call pgpt1(rad(ia),s(ia),17)
            enddo
            chimin=1.e10
            do iamp=1,namp
               amp=ampmin+float(iamp-1)/float(namp-1)*(ampmax-ampmin)
               do ia=1,ng
                  w=rg(ia)/sig
                  gaus=amp*exp(-w*w/2)/sqrt(2.*pi*sig**2)
                  sg(ia)=gaus
               enddo
               chi=0.
               do ia=1,n
                  call xlinint(rad(ia),ng,rg,sg,sgp)
                  diff=abs(sgp-s(ia))
                  chi=chi+(diff)**3
c                  chi=chi+diff
               enddo
               if(chi.lt.chimin) then
                  chimin=chi
                  ampbest=amp
               endif
            enddo
            amp=ampbest
            do ia=1,ng
               w=rg(ia)/sig
               gaus=amp*exp(-w*w/2)/sqrt(2.*pi*sig**2)
               sg(ia)=gaus
            enddo
c            call pgline(ng,rg,sg)
            if(chimin.lt.chiall) then
               chiall=chimin
               xtbest=xt
               ytbest=yt
               ampall=ampbest
            endif
         enddo
      enddo
      print *,xtbest,ytbest,chiall
      xt=xtbest
      yt=ytbest
c      call pgenv(0.,radmax,0.,smax,0,0)
      call pgenv(0.,radmax,-30.,smax,0,0)
      do ia=1,n
         rad(ia)=sqrt((x(ia)-xt)**2+(y(ia)-yt)**2)
         call pgpt1(rad(ia),s(ia),17)
      enddo
      amp=ampall
      do ia=1,ng
         w=rg(ia)/sig
         gaus=amp*exp(-w*w/2)/sqrt(2.*pi*sig**2)
         sg(ia)=gaus
      enddo
      call pgline(ng,rg,sg)
      open(unit=11,file='out',status='unknown')
      do ia=1,n
         call xlinint(rad(ia),ng,rg,sg,sgp)
         write(11,*) rad(ia),sgp-s(ia)
      enddo      
      close(11)
      
      call pgend

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
