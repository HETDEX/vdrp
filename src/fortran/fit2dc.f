
      parameter(nmax=3000)
      real xr(nmax),xd(nmax),xf(nmax),xw(nmax)
      real xr0(nmax),xd0(nmax)
      real xfa(nmax),da(nmax),gausa(nmax)
      integer iflag(nmax)
      character an(nmax,4)*20,a5*20,a6*20,a7*20,a8*20
      parameter(pi=3.141593e0)      

      read *,xrs0,xds0
      rfw=1.4
      rsig=rfw/2.35
      open(unit=1,file="in",status="old")
      xmax=-1e10
      xmin=1e10
      ymax=-1e10
      ymin=1e10
      xfmax=0.
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,a5,a6,a7,a8
         n=n+1
         xr0(n)=x1
         xd0(n)=x2
         xf(n)=x3
         xw(n)=x4
         an(n,1)=a5
         an(n,2)=a6
         an(n,3)=a7
         an(n,4)=a8
         iflag(n)=0
         if(xf(n).eq.0) iflag(n)=1
         xmin=min(xmin,xr(i))
         xmax=max(xmax,xr(i))
         ymin=min(xmin,xd(i))
         ymax=max(ymax,xd(i))
         xfmax=max(xfmax,xf(i))
      enddo
 666  continue
      close(1)
      
      do i=1,n
         xr(i)=xr0(i)-xrs0
         xd(i)=xd0(i)-xds0
         xr(i)=xr(i)*3600.*cos(xds0/57.3)
         xd(i)=xd(i)*3600.
      enddo
      xrs=0.
      xds=0.

      as=40.
      ae=2000.
      ae=xfmax*2.
      print *,ae
      chimin=1e10
      do ia=1,100
         at=as+(ae-as)/float(100-1)*float(ia-1)
         call getchifib(0.,0.,at,n,xr,xd,xf,xw,iflag,da,gausa,an,chi,0)
         if(chi.lt.chimin) then
            chimin=chi
            atb=at
         endif
      enddo
      amps=atb
      print *,"Begin:"
      call getchifib(0.,0.,amps,n,xr,xd,xf,xw,iflag,da,gausa,an,chi,1)

      nstep=13
      xs=-.7
      xe=0.7
      ys=-0.7
      ye=0.7
      as=0.7*amps
      ae=1.3*amps
      chimin=1e10
      do i=1,nstep
         xt=xs+(xe-xs)/float(nstep-1)*float(i-1)
         do j=1,nstep
            yt=ys+(ye-ys)/float(nstep-1)*float(j-1)
            do ia=1,nstep
               at=as+(ae-as)/float(nstep-1)*float(ia-1)
               call getchifib(xt,yt,at,n,xr,xd,xf,xw,iflag,da,
     $              gausa,an,chi,0)
               if(chi.lt.chimin) then
                  chimin=chi
                  xtb=xt
                  ytb=yt
                  atb=at
               endif
            enddo
         enddo
      enddo

      print *,"End:"
      call getchifib(xtb,ytb,atb,n,xr,xd,xf,xw,iflag,da,gausa,an,chi,1)
      print *,xtb,ytb,xrs0,xds0
      xnew=xrs0+xtb/3600./cos(xds0/57.3)
      ynew=xds0+ytb/3600.
      print *,xnew,ynew,atb,chimin
      open(unit=11,file='out2',status='unknown')
      write(11,*) xnew,ynew,atb,xtb,ytb,chimin
      close(11)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(2)

      xmin=0.
      xmax=2.5
      ymin=-20.
      ymax=0.

      open(unit=11,file='out',status='unknown')
      do i=1,n
         xr(i)=sqrt(xr(i)**2+xd(i)**2)
         ymax=max(ymax,xf(i))
         write(11,1001) xr0(i),xd0(i),xf(i),xw(i),
     $        an(i,1),an(i,2),an(i,3),an(i,4),iflag(i),gausa(i)
      enddo
      close(11)

      ymax=ymax*1.05
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel("Radius to fiber center","Counts","")
      do i=1,n
         if(iflag(i).eq.0) then
            call pgsci(1)
            call pgpt1(xr(i),xf(i),17)
            call pgsci(2)
            call pgpt1(xr(i),gausa(i),22)
         endif
      enddo

 1001 format(2(2x,f10.5),1x,f10.2,1x,f5.3,2x,a17,1x,a8,1x,a3,1x,a5,
     $     x,i1,1x,f11.2)

      end

      subroutine getchifib(xrs,xds,amps,n,xr,xd,xf,xw,iflag,da,
     $     gausa,an,chi,ip)
      real xr(n),xd(n),xf(n),xw(n),da(n),chia(10000)
      real gausa(n)
      integer iflag(n)
      character an(3000,4)*20
      parameter(pi=3.141593e0)      

      rsig=1.4/2.35
      rfib=0.8
      nstep=100
      xstep=2.*rfib/float(nstep-1)
c      deltx=2.*rfib/float(nstep)
      deltx=2.*rfib
      area=amps*deltx**2
      chi=0.
      do i=1,n
         xs=xr(i)-rfib
         ys=xd(i)-rfib
         gaus=0.
         nsum=0
         do ix=1,nstep
            xp=xs+xstep*float(ix-1)
            do iy=1,nstep
               yp=ys+xstep*float(iy-1)
               dist0=sqrt((xp-xr(i))**2+(yp-xd(i))**2)
               if(dist0.lt.rfib) then
                  dist=sqrt((xp-xrs)**2+(yp-xds)**2)
                  g=dist/rsig
                  gaus=gaus+exp(-g*g/2.)/sqrt(2.*rsig*rsig*pi)*area
                  nsum=nsum+1
               endif
            enddo
         enddo
         gaus=gaus/float(nsum)
         gausa(i)=gaus
         chi1=xw(i)*(gaus-xf(i))**2/(15.)**2
         da(i)=xf(i)-gaus
         chia(i)=da(i)
         if(iflag(i).eq.0) chi=chi+chi1
         rad=sqrt((xr(i)-xrs)**2+(xd(i)-xds)**2)
         if(ip.eq.1) write(*,1001) i,xf(i),gaus,rad,da(i),chi1,
     $        an(i,1),an(i,2),an(i,3),an(i,4),iflag(i)
      enddo

      if(ip.eq.1) then
         chimin=1.e10
         do i=1,n
            if(iflag(i).eq.0) then
               if(chia(i).lt.chimin) then
                  chimin=chia(i)
                  imin=i
               endif
            endif
         enddo
c         iflag(imin)=1
      endif

 1001 format(i3,1x,f12.2,2(2x,f9.2),2(2x,f12.2),
     $     2x,a17,1x,a8,1x,a3,1x,a5,1x,i1)
      return
      end

