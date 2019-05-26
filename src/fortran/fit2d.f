
      parameter(nmax=3000)
      real xr(nmax),xd(nmax),xf(nmax),xw(nmax),az(nmax)
      real xr0(nmax),xd0(nmax),wadc(10),adc(10),fadcw(10)
      real xfa(nmax),da(nmax),gausa(nmax),fadc(3000,10)
      real ferr(nmax),xrel(nmax),relnorm(3)
      integer iflag(nmax)
      character an(nmax,4)*20,a5*20,a6*20,a7*20,a8*20,aname(3)*5
      parameter(pi=3.141593e0)      
      common/csigma/ rsig,fmof,bmof,imoff

      imoff=0
      imoff=1

      read *,xrs0,xds0
      open(unit=1,file='fwhm.use',status='old',err=955)
      read(1,*) rfw
      close(1)
      goto 956
 955  continue
      close(1)
      rfw=1.55
      print *,"No fwhm.use file, using: ",rfw
 956  continue
      open(unit=1,file='fwhm.fix',status='old',err=957)
      read(1,*) rfw0
      close(1)
      if(rfw0.lt.0) rfw=-rfw0
      goto 958
 957  continue
      close(1)
 958  continue
      print *,"Using FWHM: ",rfw
      fmof=rfw
      bmof=3.5

c - get relative frame normalizations                                                                                                             
      open(unit=1,file='normexp.out',status='old',err=333)
      ne=0
      do i=1,3
         read(1,*,end=334) a5,x2,x3
         ne=ne+1
         aname(ne)=a5
         relnorm(ne)=x3
      enddo
 334  continue
      goto 335
 333  continue
      ne=3
      aname(1)='exp01'
      aname(2)='exp02'
      aname(3)='exp03'
      relnorm(1)=1.
      relnorm(2)=1.
      relnorm(3)=1.
 335  continue
      close(1)

      rsig=rfw/2.35
      open(unit=1,file="in",status="old",err=866)
      xmax=-1e10
      xmin=1e10
      ymax=-1e10
      ymin=1e10
      xfmax=0.
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,a5,a6,a7,a8,x9,x10
         n=n+1
         xr0(n)=x1
         xd0(n)=x2
         xf(n)=x3
         xw(n)=x4
         if(x3.eq.0) xw(n)=0.
         an(n,1)=a5
         an(n,2)=a6
         an(n,3)=a7
         an(n,4)=a8
         az(n)=x9
         ferr(n)=x10
         iflag(n)=0
         if(xf(n).eq.0) iflag(n)=1
         do j=1,ne
            if(a8(1:5).eq.aname(j)) xrel(n)=relnorm(j)
         enddo
         xmin=min(xmin,xr(i))
         xmax=max(xmax,xr(i))
         ymin=min(xmin,xd(i))
         ymax=max(ymax,xd(i))
         xfmax=max(xfmax,xf(i))
      enddo
 666  continue
      close(1)
      if(n.eq.0) goto 866
      
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
      ae=xfmax*5.
      if(ae.lt.as) then
         print *,"Problem with negative flux"
         goto 866
      endif
      print *,ae
      chimin=1e10
      do ia=1,100
         at=as+(ae-as)/float(100-1)*float(ia-1)
         call getchifib(0.,0.,at,n,xr,xd,xf,xw,ferr,xrel,
     $        iflag,da,gausa,an,chi,0,sumrat)
         if(chi.lt.chimin) then
            chimin=chi
            atb=at
         endif
      enddo
      amps=atb
      print *,"Begin:"
      call getchifib(0.,0.,amps,n,xr,xd,xf,xw,ferr,xrel,
     $     iflag,da,gausa,an,chi,1,sumrat)

      nstepa=30
      nstepc=21
c      xs=-1.0
c      xe=1.0
c      ys=-1.0
c      ye=1.0
      xs=-0.2
      xe=0.2
      ys=-0.2
      ye=0.2
      as=0.7*amps
      ae=1.4*amps
      chimin=1e10
      do i=1,nstepc
         xt=xs+(xe-xs)/float(nstepc-1)*float(i-1)
         do j=1,nstepc
            yt=ys+(ye-ys)/float(nstepc-1)*float(j-1)
            do ia=1,nstepa
               at=as+(ae-as)/float(nstepa-1)*float(ia-1)
               call getchifib(xt,yt,at,n,xr,xd,xf,xw,ferr,xrel,
     $              iflag,da,gausa,an,chi,0,sumrat)
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
      call getchifib(xtb,ytb,atb,n,xr,xd,xf,xw,ferr,xrel,
     $     iflag,da,gausa,an,chi,1,sumrat)
      print *,xtb,ytb,xrs0,xds0,atb/amps
      xnew=xrs0+xtb/3600./cos(xds0/57.3)
      ynew=xds0+ytb/3600.
      print *,"RAnew, DECnew, Amp, chimin, FW"
      print *,xnew,ynew,atb,chimin,rfw
      sumg=0.
      sumd=0.
      do i=1,n
         sumg=sumg+gausa(i)
         sumd=sumd+xf(i)
      enddo
      print *,atb
      sumrat=sumg/atb
      open(unit=11,file='out2',status='unknown')
      write(11,1103) xnew,ynew,atb,xtb,ytb,chimin,rfw,sumrat
      close(11)
 1103 format(2(1x,f9.5),1x,f10.2,2(1x,f8.3),1x,f10.2,2(1x,f6.3))

c- now get the atmospheric distortion correction to each fiber

      nw=5
      wadc(1)=3500.
      wadc(2)=4000.
      wadc(3)=4500.
      wadc(4)=5000.
      wadc(5)=5500.
      adc(1)=-0.74
      adc(2)=-0.4
      adc(3)=0.0
      adc(4)=0.08
      adc(5)=0.20
      call adcor(nw,wadc,adc,fadc,0.,0.,n,xr,xd,az)
      do i=1,n
         do j=1,nw
            fadcw(j)=fadc(i,j)/fadc(i,3)
         enddo
c         write(*,1002) i,fadcw(1),fadcw(2),fadcw(3),fadcw(4),fadcw(5)
      enddo
 1002 format(i4,5(1x,f5.3))

      xmin=0.
      xmax=3.5
      ymin=-20.
      ymax=0.

      sumd=0.
      open(unit=11,file='out',status='unknown')
      do i=1,n
         xr(i)=sqrt(xr(i)**2+xd(i)**2)
         ymax=max(ymax,xf(i))
         do j=1,nw
            fadcw(j)=fadc(i,j)/fadc(i,3)
         enddo
         write(11,1001) xr0(i),xd0(i),xf(i),xw(i),
     $        an(i,1),an(i,2),an(i,3),an(i,4),iflag(i),gausa(i),
     $        fadcw(1),fadcw(2),fadcw(3),fadcw(4),fadcw(5)
         sumd=sumd+xf(i)
      enddo
      close(11)
      open(unit=11,file="fac2.out",status="unknown")
      write(11,*) sumrat,atb/sumd
      close(11)
      print *,"Fiber coverage: ",sumrat,atb/sumd

      ymax=ymax*1.5
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(2)

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
 866  continue
 1001 format(2(2x,f10.5),1x,f10.2,1x,f5.3,2x,a17,1x,a8,1x,a3,1x,a5,
     $     x,i1,1x,f11.2,5(1x,f5.3))

      end

      subroutine getchifib(xrs,xds,amps,n,xr,xd,xf,xw,fe,xrel,iflag,da,
     $     gausa,an,chi,ip,sumrat)
      real xr(n),xd(n),xf(n),xw(n),da(n),chia(10000),fe(n)
      real gausa(n),xrel(n)
      integer iflag(n)
      character an(3000,4)*20
      parameter(pi=3.141593e0)      
      common/csigma/ rsig,fmof,bmof,imoff

      if(ip.eq.1) write(*,*) "Ifib     Counts      Fit",
     $     "       Distance        C-F           Chi"

      rfib=0.75
c      nstep=50
      nstep=30
      xstep=2.*rfib/float(nstep-1)
      deltx=pi*rfib*rfib
      area=amps*deltx/(2.*rsig*rsig*pi)
      areamoff=4.*(2.**(1./bmof)-1.)*(bmof-1.)/pi/fmof/fmof
      areamoff=amps*deltx*areamoff
      chi=0.
      do i=1,n
         xs=xr(i)-rfib
         ys=xd(i)-rfib
         gaus=0.
         xmoff=0.
         nsum=0
         do ix=1,nstep
            xp=xs+xstep*float(ix-1)
            do iy=1,nstep
               yp=ys+xstep*float(iy-1)
               dist0=sqrt((xp-xr(i))**2+(yp-xd(i))**2)
               if(dist0.lt.rfib) then
                  dist=sqrt((xp-xrs)**2+(yp-xds)**2)
                  g=dist/rsig
                  gaus=gaus+exp(-g*g/2.)*area
                  xmoff=xmoff+areamoff*((1.+4.*(2.**(1./bmof)-1.)*
     $                 (dist/fmof)**2)**(-bmof))
                  nsum=nsum+1
               endif
            enddo
         enddo
         gaus=gaus/float(nsum)
         xmoff=xmoff/float(nsum)
         if(imoff.eq.1) gaus=xmoff
         gausa(i)=gaus
         if(fe(i).gt.0) then
            chi1=xw(i)*(gaus-xf(i))**2/(fe(i))**2
         else
            chi1=0.
         endif
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
         iflag(imin)=1
         iflag(imin)=0
      endif

c- now get area covered with fibers
      sumf1=0.
      sumf2=0.
      nfull=100
      sigfull=5.
      xs=xrs-sigfull*rsig
      xe=xrs+sigfull*rsig
      ys=xds-sigfull*rsig
      ye=xds+sigfull*rsig
      do ix=1,nfull
         xp=xs+float(ix-1)*(xe-xs)/float(nfull-1)
         do jx=1,nfull
            yp=ys+float(jx-1)*(ye-ys)/float(nfull-1)
            do i=1,n
               dist0=sqrt((xp-xr(i))**2+(yp-xd(i))**2)
               if(dist0.lt.rfib) then
                  sumf1=sumf1+1
                  goto 866
               endif
            enddo
 866        continue
            sumf2=sumf2+1
         enddo
      enddo
      sumrat=sumf1/sumf2

 1001 format(i3,1x,f12.2,2(2x,f9.2),2(2x,f12.2),
     $     2x,a17,1x,a8,1x,a3,1x,a5,1x,i1)
      return
      end

      subroutine adcor(nw,wadc,adc,fadc,xrs0,xds0,n,xr,xd,az)
      real wadc(nw),adc(nw),fadc(3000,10),xr(n),xd(n),az(n)
      parameter(pi=3.141593e0)      
      common/csigma/ rsig,fmof,bmof,imoff
      
      dtr=180./pi
      xrs=xrs0
      xds=xds0
      rfib=0.75
      nstep=50
      xstep=2.*rfib/float(nstep-1)
      deltx=pi*rfib*rfib
      area=1.*deltx/(2.*rsig*rsig*pi)
      areamoff=4.*(2.**(1./bmof)-1.)*(bmof-1.)/pi/fmof/fmof
      areamoff=deltx*areamoff
      do ia=1,nw
         xaoff=adc(ia)*sin(az(1)/dtr)
         yaoff=adc(ia)*cos(az(1)/dtr)
         do i=1,n
            xaoff=adc(ia)*sin(az(i)/dtr)
            yaoff=adc(ia)*cos(az(i)/dtr)
            xs=xr(i)-rfib+xaoff
            ys=xd(i)-rfib+yaoff
            gaus=0.
            xmoff=0.
            nsum=0
            do ix=1,nstep
               xp=xs+xstep*float(ix-1)
               do iy=1,nstep
                  yp=ys+xstep*float(iy-1)
                  dist0=sqrt((xp-xr(i))**2+(yp-xd(i))**2)
                  if(dist0.lt.rfib) then
                     dist=sqrt((xp-xrs)**2+(yp-xds)**2)
                     g=dist/rsig
                     gaus=gaus+exp(-g*g/2.)*area
                     xmoff=xmoff+areamoff*((1.+4.*(2.**(1./bmof)-1.)*
     $                    (dist/fmof)**2)**(-bmof))
                     nsum=nsum+1
                  endif
               enddo
            enddo
            gaus=gaus/float(nsum)
            xmoff=xmoff/float(nsum)
            if(imoff.eq.1) gaus=xmoff
            fadc(i,ia)=gaus
         enddo

c- now get area covered with fibers
         sumf1=0.
         sumf2=0.
         nfull=100
         sigfull=5.
         xs=xrs-sigfull*rsig
         xe=xrs+sigfull*rsig
         ys=xds-sigfull*rsig
         ye=xds+sigfull*rsig
         xs=xs+xaoff
         ys=ys+yaoff
         do ix=1,nfull
            xp=xs+float(ix-1)*(xe-xs)/float(nfull-1)
            do jx=1,nfull
               yp=ys+float(jx-1)*(ye-ys)/float(nfull-1)
               do i=1,n
                  dist0=sqrt((xp-xr(i))**2+(yp-xd(i))**2)
                  if(dist0.lt.rfib) then
                     sumf1=sumf1+1
                     goto 866
                  endif
               enddo
 866           continue
               sumf2=sumf2+1
            enddo
         enddo
         sumrat=sumf1/sumf2
         do i=1,n
c            fadc(i,ia)=fadc(i,ia)/sumrat
         enddo
      enddo

      return
      end
