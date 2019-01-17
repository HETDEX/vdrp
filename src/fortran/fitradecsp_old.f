
      parameter(nmax=3000,nfmax=3000)
      real ara(nmax),adec(nmax),sflux(nmax),sfluxe(nmax)
      real wa(nfmax,nmax),fa(nfmax,nmax),fea(nfmax,nmax)
      real fa2(nfmax,nmax),fea2(nfmax,nmax),fitout(nmax*5,6)
      real weight(nfmax),raf(nfmax),decf(nfmax),xw(nmax),fitcon(nmax)
      real spec(nmax,9),az(nfmax),fadcw(nfmax,5),fitw(nmax),fitsn(nmax)
      real rat(nmax*5),dect(nmax*5),fitwt(nmax*5),fitsnt(nmax*5)
      real fweight(nfmax,nmax),fitp(nmax,8),fitchi(nmax),fitchit(nmax*5)
      real fitamp(nmax),fitsig(nmax),fitampt(nmax),fitsigt(nmax)
      real fitcont(nmax),fweight2(nfmax,nmax),sumrata(nmax)
      integer iflag(nfmax)

      rd=3.
      wd=4.

      print *,"ra,dec,step,nstep,wcen,wrange,ifit1"
      read *,ra,dec,step,nstep,wcen,wrange,ifit1

      sigin=-3.0
      if(ifit1.eq.1) sigin=2.5
      
      call getspec(wa,fa,fea,fa2,fea2,fweight,fweight2,
     $     naf,ntf,raf,decf,az)
      call mkraster(ra,dec,step,nstep,ara,adec,na)
      call getfl2d(wa,fa,fea,fweight,naf,ntf,wcen,wrange,
     $     sflux,sfluxe,xw)
      if(ifit1.ge.0) wrange=50.

      nt=0
      do i=1,na
         call fit2d(ara(i),adec(i),ntf,sflux,sfluxe,raf,decf,
     $        xw,weight,iflag,fadcw,az,chi,amps,sumrata)
         if(ifit1.ge.0) then
            call sumspec(naf,ntf,wa,fa,fea,fa2,fea2,weight,
     $           iflag,fadcw,az,spec,sumrata,fweight2)
            call fitspec(naf,spec,sigin,wcen,wrange,ifit1,nout,fitout)
            call getbestw(nout,fitout,wd,nw,fitw,fitsn,fitchi,fitcon,
     $           fitamp,fitsig)
            do j=1,nw
               nt=nt+1
               fitwt(nt)=fitw(j)
               fitsnt(nt)=fitsn(j)
               fitchit(nt)=fitchi(j)
               fitampt(nt)=fitamp(j)
               fitsigt(nt)=fitsig(j)
               fitcont(nt)=fitcon(j)
               rat(nt)=ara(i)
               dect(nt)=adec(i)
               if(nt.eq.nmax*5) goto 766
            enddo
         else
            nt=nt+1
            fitchit(nt)=chi
            fitcont(nt)=amps
            rat(nt)=ara(i)
            dect(nt)=adec(i)
         endif
      enddo
 766  continue

      if(ifit1.ge.0) then
         open(unit=11,file='outbest',status='unknown')
         call getbestp(nt,rat,dect,fitwt,fitsnt,fitchit,fitampt,fitsigt,
     $        fitcont,rd,wd,np,fitp)
         do i=1,np
            write(11,1101) fitp(i,1),fitp(i,2),fitp(i,3),
     $           fitp(i,4),fitp(i,5),fitp(i,6),fitp(i,7),fitp(i,8)
         enddo
      else
         open(unit=11,file='outbestc',status='unknown')
         do i=1,nt
            write(11,1102) rat(i),dect(i),fitchit(i),fitcont(i)
         enddo
      endif
      close(11)

      if(ifit1.eq.1) call writespec(naf,spec)

 1101 format(2(1x,f10.6),3(1x,f7.2),1x,f9.1,1x,f7.2,1x,f10.2)
 1102 format(2(1x,f10.6),2(1x,f11.3))
      end

      subroutine writespec(n,spec)
      parameter(nmax=3000)
      real spec(nmax,9)
      
      open(unit=13,file='spec.out',status='unknown')
      do i=1,n
         write(13,1301) spec(i,1),spec(i,3),spec(i,5),
     $        spec(i,2),spec(i,4),spec(i,6),spec(i,7),
     $        spec(i,8),spec(i,9)
      enddo
 1301 format(1x,f7.2,8(1x,f11.3))

      return
      end

      subroutine getbestp(nt,rat,dect,fitwt,fitsnt,fitchit,
     $     fitampt,fitsigt,fitcont,rd,wd,nw,fitp)
      parameter(nmax=3000)
      real rat(nt),dect(nt),fitwt(nt),fitsnt(nt),fitchit(nt)
      real fitp(nmax,8),fitampt(nt),fitsigt(nt),fitcont(nt)
      integer icheck(nmax)
      parameter(radtodeg=57.29578)

      cosd=cos(dect(1)/radtodeg)
      nw=0
      do j=1,nt
         icheck(j)=0
      enddo
      do k=1,nt
         if(icheck(k).eq.0) then
            snmax=0.
            do j=1,nt
               if(icheck(j).eq.0) then
                  if(fitsnt(j).gt.snmax) then
                     snmax=fitsnt(j)
                     imax=j
                  endif
               endif
            enddo
            icheck(imax)=1
            nw=nw+1
            fitp(nw,1)=rat(imax)
            fitp(nw,2)=dect(imax)
            fitp(nw,3)=fitwt(imax)
            fitp(nw,4)=fitsnt(imax)
            fitp(nw,5)=fitchit(imax)
            fitp(nw,6)=fitampt(imax)
            fitp(nw,7)=fitsigt(imax)
            fitp(nw,8)=fitcont(imax)
            do j=1,nt
               wd0=abs(fitp(nw,3)-fitwt(j))
               rd0=cosd*(rat(imax)-rat(j))**2+(dect(imax)-dect(j))**2
               rd0=3600.d0*sqrt(rd0)
               if(wd0.lt.wd.and.rd0.lt.rd) icheck(j)=1
            enddo
         endif
      enddo

      return
      end

      subroutine getbestw(nout,fitout,wd,nw,fitw,fitsn,fitchi,
     $     fitcon,fitamp,fitsig)
      parameter(nmax=3000)
      real fitout(nmax*5,6),fitw(nmax),fitsn(nmax),fitchi(nmax)
      real fitcon(nmax),fitamp(nmax),fitsig(nmax)
      integer icheck(nmax)

c - this routine finds the wavelength with the highest S/N within wd
      nw=0
      do j=1,nout
         icheck(j)=0
      enddo
      do k=1,nout
         if(icheck(k).eq.0) then
            snmax=0.
            do j=1,nout
               if(icheck(j).eq.0) then
                  if(fitout(j,4).gt.snmax) then
                     snmax=fitout(j,4)
                     imax=j
                  endif
               endif
            enddo
            icheck(imax)=1
            nw=nw+1
            fitw(nw)=fitout(imax,1)
            fitamp(nw)=fitout(imax,2)
            fitsig(nw)=fitout(imax,3)
            fitsn(nw)=fitout(imax,4)
            fitcon(nw)=fitout(imax,5)
            fitchi(nw)=fitout(imax,6)
            do j=1,nout
               if(abs(fitout(j,1)-fitout(imax,1)).lt.wd) icheck(j)=1
            enddo
         endif
      enddo

      return
      end

      subroutine getfl2d(wa,fa,fea,fweight,na,ntf,wcen,wrange,
     $     sflux,sfluxe,fw)
      parameter(nmax=3000,nfmax=3000)
      real wa(nfmax,nmax),fa(nfmax,nmax),fea(nfmax,nmax)
      real fweight(nfmax,nmax),xin3(nmax),fw(nmax)
      real sflux(nmax),sfluxe(nmax),xin(nmax),xin2(nmax)
      w1=wcen-wrange
      w2=wcen+wrange
      do i=1,ntf
         nin=0
         do j=1,na
            if(wa(i,j).gt.w1.and.wa(i,j).lt.w2.and.fa(i,j).ne.0) then
               nin=nin+1
               xin(nin)=fa(i,j)
               xin2(nin)=fea(i,j)*fea(i,j)
               xin3(nin)=fweight(i,j)
            endif
         enddo
         if(nin.ge.1) then
            call biwgt(xin,nin,xb,xs)
            call biwgt(xin2,nin,xb2,xs2)
            call biwgt(xin3,nin,xb3,xs3)
         else
            xb=0.
            xb2=0.
            xb3=0.
         endif
         sflux(i)=xb
         fw(i)=xb3
         if(xb2.ge.0) then
            sfluxe(i)=sqrt(xb2)
         else
            sfluxe(i)=0
         endif   
      enddo

      return
      end

      subroutine getspec(wa,fa,fea,fa2,fea2,fweight,fweight2,
     $     na,ntf,ra,dec,az)
      parameter(nmax=3000,nfmax=3000)
      real wa(nfmax,nmax),fa(nfmax,nmax),fea(nfmax,nmax)
      real fa2(nfmax,nmax),fea2(nfmax,nmax),fweight(nfmax,nmax)
      real ra(nfmax),dec(nfmax),az(nfmax),fweight2(nfmax,nmax)
      character file2*80,file3*120

      open(unit=1,file='list',status='old')
      ntf=0
      do i=1,nfmax
         read(1,*,end=666) file2,x1,x2,x3
         ntf=ntf+1
         ra(ntf)=x1
         dec(ntf)=x2
         az(ntf)=x3
         open(unit=2,file=file2,status='old')
         na=0
         do j=1,nmax
            read(2,*,end=667) x1,x2,x3,x4,x5,x6,x7,x8,x9
            na=na+1
            wa(i,na)=x1
            fa(i,na)=x2
            fea(i,na)=x8
            fa2(i,na)=x3
            fea2(i,na)=x9
            fweight(i,na)=x4*x5
c            fweight2(i,na)=x4*x6
            fweight2(i,na)=1.
         enddo
 667     continue
         close(2)
      enddo
 666  continue
      close(1)

      return
      end

      subroutine mkraster(ra,dec,step,nstep,ara,adec,ntot)
      parameter(nmax=3000)
      real ara(nmax),adec(nmax)

      if(nstep.lt.2) then
         ara(1)=ra
         adec(1)=dec
         ntot=1
         return
      endif

      atot=step*float(nstep-1)
      ahalf=atot/2.

      cdec=cos(dec/57.29)
      ras=ra-ahalf/cdec/3600.
      decs=dec-ahalf/3600.

      ia=0
      do i=1,nstep
         r=ras+float(i-1)*step/cdec/3600.
         do j=1,nstep
            d=decs+float(j-1)*step/3600.
            ia=ia+1
            ara(ia)=r
            adec(ia)=d
         enddo
      enddo
      ntot=ia

      return
      end

      subroutine fit2d(ra,dec,ntf,sflux,sfluxe,raf,decf,xw,gausa,
     $     iflag,fadcw,az,chi,amps,sumrata)

      parameter(nmax=3000,nfmax=3000)
      real raf(ntf),decf(ntf),sflux(ntf),sfluxe(ntf),gaussa(ntf)
      real xr(nmax),xd(nmax),wadc(5),adc(5),fadcw(nfmax,5),az(ntf)
      real da(nfmax),gausa(nfmax),xw(ntf),fadc(nfmax,5),sumrata(nmax)
      integer iflag(nfmax)
      parameter(pi=3.141593e0)      
      common/csigma/ rsig

      open(unit=1,file='fwhm.use',status='old',err=955)
      read(1,*) rfw
      close(1)
      goto 956
 955  continue
      close(1)
      rfw=1.55
 956  continue
      rsig=rfw/2.35
      do i=1,ntf
         iflag(i)=0
         if(sflux(i).eq.0) iflag(i)=1
      enddo

      xfmax=0.
      do i=1,ntf
         xr(i)=raf(i)-ra
         xd(i)=decf(i)-dec
         xr(i)=xr(i)*3600.*cos(dec/57.3)
         xd(i)=xd(i)*3600.
         xfmax=max(xfmax,sflux(i))
      enddo

      as=10.
      ae=2000.
      ae=xfmax*2.
      chimin=1e10
      do ia=1,100
         at=as+(ae-as)/float(100-1)*float(ia-1)
         call getchifib(0.,0.,at,ntf,xr,xd,sflux,xw,sfluxe,
     $        iflag,da,gausa,chi,0,sumrat)
         if(chi.lt.chimin) then
            chimin=chi
            atb=at
         endif
      enddo
      amps=atb
      call getchifib(0.,0.,amps,ntf,xr,xd,sflux,xw,sfluxe,
     $     iflag,da,gausa,chi,1,sumrat)

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
      call adcor(nw,wadc,adc,fadc,0.,0.,ntf,xr,xd,az,sumrata)
      do i=1,ntf
         do j=1,nw
            fadcw(i,j)=fadc(i,j)/fadc(i,3)
         enddo
      enddo

      return
      end

      subroutine getchifib(xrs,xds,amps,n,xr,xd,xf,xw,fe,iflag,da,
     $     gausa,chi,ip,sumrat)
      real xr(n),xd(n),xf(n),xw(n),da(n),fe(n)
      real gausa(n)
      integer iflag(n)
      parameter(pi=3.141593e0)      
      common/csigma/ rsig

c      if(ip.eq.1) write(*,*) "Ifib     Counts      Fit",
c     $     "       Distance        C-F           Chi"

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
         if(fe(i).gt.0) then
            chi1=xw(i)*(gaus-xf(i))**2/(fe(i))**2
         else
            chi1=0.
         endif
         da(i)=xf(i)-gaus
c         chia(i)=da(i)
         if(iflag(i).eq.0) chi=chi+chi1
         rad=sqrt((xr(i)-xrs)**2+(xd(i)-xds)**2)
c         if(ip.eq.1) write(*,1001) i,xf(i),gaus,rad,da(i),chi1,
c     $        an(i,1),an(i,2),an(i,3),an(i,4),iflag(i)
      enddo

 1001 format(i3,1x,f12.2,2(2x,f9.2),2(2x,f12.2),
     $     2x,a17,1x,a8,1x,a3,1x,a5,1x,i1)
      return
      end

      subroutine adcor(nw,wadc,adc,fadc,xrs0,xds0,n,xr,xd,az,sumrata)
      real wadc(nw),adc(nw),fadc(3000,5),xr(n),xd(n),az(n),sumrata(nw)
      parameter(pi=3.141593e0)      
      common/csigma/ rsig
      
      dtr=180./pi
      xrs=xrs0
      xds=xds0
      rfib=0.8
      nstep=100
      xstep=2.*rfib/float(nstep-1)
      deltx=2.*rfib
      area=1.*deltx**2
      do ia=1,nw
         xaoff=adc(ia)*sin(az(1)/dtr)
         yaoff=adc(ia)*cos(az(1)/dtr)
         do i=1,n
            xaoff=adc(ia)*sin(az(i)/dtr)
            yaoff=adc(ia)*cos(az(i)/dtr)
            xs=xr(i)-rfib+xaoff
            ys=xd(i)-rfib+yaoff
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
            fadc(i,ia)=gaus
         enddo

c- now get area covered with fibers
         sumf1=0.
         sumf2=0.
         nfull=100
         sigfull=6.
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
         sumrata(ia)=sumrat
         do i=1,n
            fadc(i,ia)=fadc(i,ia)/sumrat
         enddo
      enddo

      return
      end

      subroutine sumspec(naf,ntf,wa,fa,fea,fa2,fea2,gna0,
     $     iflag,fadcw,az,spec,sumrata,fweight2)
      parameter(nmax=3000,nfmax=3000)
      real wa(nfmax,nmax),fa(nfmax,nmax),fea(nfmax,nmax)
      real fa2(nfmax,nmax),fea2(nfmax,nmax)
      real gna(ntf),gna0(nfmax),az(ntf),spec(nmax,9),wn(5)
      real x(nmax),y(nmax),y2(nmax),ye(nmax),ye2(nmax)
      real wv(5),fadcw(nfmax,5),ysum3(nmax),ysum3e(nmax),gsum(nmax)
      real ysum(nmax),ysum2(nmax),ysume(nmax),ysum2e(nmax),fw2(nmax)
      real yout1(nmax),yout2(nmax),yout3(nmax),yout4(nmax),yout5(nmax)
      real yout6(nmax),yout7(nmax),sumrata(nmax),fweight2(nfmax,nmax)
      integer iflag(ntf)

      fcut=0.03

      do i=1,nmax
         ysum(i)=0.
         ysum2(i)=0.
         ysume(i)=0.
         ysum2e(i)=0.
         ysum3(i)=0.
         ysum3e(i)=0.
         fw2(i)=0.
         gsum(i)=0.
      enddo
      sumg=0.
      ng=0
      do i=1,ntf
         if(iflag(i).eq.0) then
            ng=i
            gna(ng)=gna0(i)
            sumg=sumg+gna0(i)
         endif
      enddo
      do i=1,ng
         gna(i)=gna(i)/sumg
      enddo
      sumgall=sumg
c      fcut=fcut/float(ng)*7.

      nsum=0
      sumg=0.
      nw=5
      wv(1)=3500.
      wv(2)=4000.
      wv(3)=4500.
      wv(4)=5000.
      wv(5)=5500.
c - get the weighted sum
      do il=1,ntf
         gn0=gna0(il)
         if(gna(il).lt.fcut) iflag(il)=1
         if(iflag(il).eq.0) then
            nsum=nsum+1
            sumg=sumg+gn0
            n=0
            do i=1,nw
               wn(i)=fadcw(il,i)
            enddo
            do i=1,naf
               x1=wa(il,i)
               call xlinint(x1,nw,wv,wn,fadc)
               gn=gn0*fadc
               n=n+1
               x(n)=x1
               y(n)=fa(il,i)
               y2(n)=fa2(il,i)
               ye(n)=fea(il,i)
               ye2(n)=fea2(il,i)
               ysum(n)=ysum(n)+y(n)*gn
               ysum2(n)=ysum2(n)+y2(n)*gn
               ysume(n)=ysume(n)+ye(n)*ye(n)*gn
               ysum2e(n)=ysum2e(n)+ye2(n)*ye2(n)*gn
               fw2(n)=fw2(n)+fweight2(il,i)*gn
               if(y(n).ne.0) gsum(n)=gsum(n)+gn
            enddo
         endif
      enddo

c - get the straight sum
      do il=1,ntf
         if(iflag(il).eq.0) then
            n=0
            do i=1,naf
               n=n+1
               ysum3(n)=ysum3(n)+fa(il,i)
               ysum3e(n)=ysum3e(n)+fea(il,i)*fea(il,i)
            enddo
         endif
      enddo

      if(nsum.eq.0.or.sumg.le.0.) then
         fac=0.
      else
         sumg=sumg/float(nsum)
         fac=1./sumg/1.2
      endif

      do i=1,naf
         fac2=0.
         if(gsum(i).gt.0) fac2=sumgall/gsum(i)
         facu=fac*fac2
         xs1=sqrt(ysume(i)*facu)
         xs2=sqrt(ysum2e(i)*facu)
         xs3=sqrt(ysum3e(i))
         spec(i,1)=wa(1,i)
         spec(i,2)=ysum(i)*facu
         spec(i,3)=ysum2(i)*facu*1.0e17
         spec(i,4)=xs1
         spec(i,5)=xs2*1.0e17
         spec(i,6)=ysum3(i)
         spec(i,7)=xs3
c         spec(i,8)=fac2
         xdum=0.
         if(gsum(i).gt.0) xdum=fw2(i)/gsum(i)
         if(xdum.gt.0) then
            spec(i,8)=1./xdum
            spec(i,8)=1.
         else
            spec(i,8)=0.
         endif
         call xlinint(wa(1,i),nw,wv,sumrata,wgeom)
         spec(i,9)=wgeom
      enddo
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

      subroutine fitspec(n0,spec,signsl0,wcen,wrange,ifit1,nout,fitout)

      parameter(nmax=3000,nca=17)
      real spec(3000,9)
      real wave(nmax),flux(nmax),x(nmax),y(nmax),alpha(nca,nca)
      real a(nca),yin(nmax),covar(nca,nca)
      real fluxe(nmax),ye(nmax),fitout(nmax*5,6)
      parameter(pi=3.141592e0,cee=2.99792458e5)
      data big/1.e20/
      common/aval/ np,ifitsig

      n=n0
      ifit=0
      sncut=3.5
      if(ifit1.eq.1) sncut=0.
      sncuthi=200.
      ampcut=1.e30
      sigcut=50.
c      wavecut=5.
      wavecut=20.
      pixsize=2.

      wavec=50.

      if(signsl0.lt.0) then
         signsl=-signsl0
         ifitsig=0
      else
         signsl=signsl0
         ifitsig=1
      endif

c      ws=3504.
c      we=5496.
c      nw=600
      ws=spec(1,1)
      we=spec(n,1)
      dwave=0.5
      dwave=1.0
      nw=nint((we-ws)/dwave)+1

      nadd=5

      do i=1,n
         wave(i)=spec(i,1)
         flux(i)=spec(i,3)
         fluxe(i)=spec(i,5)
         if(flux(i).eq.0) flux(i)=-666
      enddo

      do iall=1,nw
         wave0=ws+float(iall-1)*(we-ws)/float(nw-1)
         if(ifit1.eq.1) then
            wave0=wcen
            wavec=wrange
         endif
         wlo=wave0-wavec
         wup=wave0+wavec
         ymax=-big
         nt=0
         do i=1,nca
            do j=1,nca
               covar(i,j)=0.
               alpha(i,j)=0.
            enddo
         enddo
         do i=1,n
            if(wave(i).gt.wlo.and.wave(i).lt.wup.and.
     $           nint(flux(i)).ne.-666) then
               nt=nt+1
               x(nt)=wave(i)
               y(nt)=flux(i)
               yin(nt)=y(nt)
               ye(nt)=fluxe(i)
               ymax=max(ymax,y(nt))
            endif
         enddo
         call biwgt(yin,nt,xb,xs)
         call biwgt(yin,12,xb,xs)

         amp=(ymax-xb)*3.5
         a(1)=signsl
         a(2)=0.0
         a(3)=0.
         a(4)=xb
         a(5)=0.
         a(6)=wave0
         a(7)=amp
         a(8)=a(1)
         a(9)=0.
         na=8
         np=1

         call fitherms(nt,x,y,ye,a,na,covar,alpha,nca)

         h3=a(2)
         h4=a(3)
         con=a(4)
         xoff=a(5)
         rms=0.
         chi=0.
         do ia=1,nt
            yfit=con
            do i=1,np
               sigg=a(i+nadd+np+np)
               amp=a(i+nadd+np)
               vel=a(i+nadd)+xoff
               w=(x(ia)-vel)/sigg
               gaus=exp(-w*w/2.)/sqrt(2.*pi*sigg**2)
               yfit=yfit+amp*gaus*(1.+h3*fh3(w)+h4*fh4(w))
            enddo
            rms=rms+(y(ia)-yfit)**2
            if(ye(ia).gt.0) chi=chi+((y(ia)-yfit)/ye(ia))**2
            y(ia)=yfit
         enddo
         rms=sqrt(rms/float(nt))
         chi=chi/float(nt)

         wfit=a(6)+a(5)

c - check if enough data under the fit
         wfit0=wfit
         nfit=0
         do ii=1,nt
            if(x(ii).gt.(wfit0-6).and.x(ii).lt.(wfit0+6)) nfit=nfit+1
         enddo
         if(nfit.le.5) goto 766

         znew=(a(6)+a(5))/wave0-1
         zerr=(a(6)+a(5)+sqrt(covar(5,5)))/wave0-1
         zerr=zerr-znew
c         ampe=covar(7,7)

         if(sigg.gt.sigcut) goto 766
         if(chi.gt.99) goto 766
         if(con.gt.1000) goto 766
         if(amp.le.0) goto 766
         if(abs(wfit-wave0).gt.wavecut) goto 766
         if((wfit-x(1)).lt.6) goto 766
         if((x(nt)-wfit).lt.6) goto 766

         igood=0
c         if(ampe.gt.0.and.ampe.lt.ampcut) then
            igood=1
c         endif
         if(igood.eq.0) goto 766
c         if((wfit-x(1)).lt.8) goto 766
c         if((x(nt)-wfit).lt.8) goto 766
c         ampe=sqrt(ampe)
         sigg=sqrt(sigg*sigg)
         xnp=4.*sigg
         xnp=xnp/pixsize

c - get noise from the rms
         xnoise=rms*sqrt(xnp)
         ston=0.95*amp/xnoise/pixsize

c - get noise from the errors
         w1=wfit-xnp
         w2=wfit+xnp
         nerr=0
         xnoise2=0.
         xmaxn=0.
         do i=1,nt
            if(x(i).ge.w1.and.x(i).le.w2) then
               nerr=nerr+1
               xnoise2=xnoise2+ye(i)*ye(i)
               xmaxn=max(xmaxn,ye(i))
            endif
         enddo
         if(nerr.gt.0) then
            xnoise2=sqrt(xnoise2)
            xnoise2=xmaxn*sqrt(float(nerr))
         else
            xnoise2=0.
         endif
         ston2=0.95*amp/xnoise2/pixsize
c         print *,'RMS, dAMP, S/N = ',rms,ampe,ston,ston2
         ston=ston2

         if(ston.lt.sncut) goto 766
         if(ston.gt.sncuthi) goto 766

         ifit=ifit+1
         nout=ifit
         fitout(ifit,1)=wfit
         fitout(ifit,2)=amp/pixsize
         fitout(ifit,3)=abs(a(8))
         fitout(ifit,4)=ston
         fitout(ifit,5)=con
         fitout(ifit,6)=chi
 766     continue
         if(ifit1.eq.1) goto 966
      enddo
 966  continue

      return
      end

      subroutine fitherms(n,x,y,sig,a,na,covar,alpha,nca)

      parameter(ncaf=17)

      real x(n),y(n),a(na),covar(nca,nca)
      real alpha(nca,nca),sig(n)
      integer ia(ncaf)
      common/aval/ np,ifitsig

      data tol,itermax/1.e-4,1000/

      nadd=5
      do i=1,na
         ia(i)=1
      enddo
      do i=nadd+1,np+nadd
         ia(i)=0
      enddo
      ia(1)=0
      ia(2)=0
      ia(3)=0
      ia(4)=1
c     this is wavelength: 0 fix, 1 fit
c      ia(5)=0
c      ia(6)=0
c     this is sigma: 0 fix, 1 fit
      ia(8)=0
      if(ifitsig.eq.1) ia(8)=1

      alamda=-1
      alamo=1.e10
      cold=1.e10
      do iter=1,itermax
         call mrqminb(x,y,sig,n,a,ia,na,covar,alpha,nca,
     $        chisq,alamda)
         chirel=abs(cold-chisq)/chisq
         cold=chisq
         if(alamda.lt.alamo.and.chirel.lt.tol) goto 666
         if(alamda.gt.1.e9) goto 666
         alamo=alamda
      enddo
c      print *,'Hit max iteration'
 666  continue

      call mrqminb(x,y,sig,n,a,ia,na,covar,alpha,nca,
     $     chisq,0.)

      return
      end

      subroutine funcs(x,a,yfit,dyda,na)
      real a(na),dyda(na)
      parameter(pi=3.141593e0)
      common/aval/ np,ifitsig

      h3=a(2)
      h4=a(3)
      con=a(4)
      xoff=a(5)
      nadd=5
      yfit=con
      do i=1,na
         dyda(i)=0.
      enddo
      dyda(4)=1.
      do i=1,np
         sig=a(i+nadd+np+np)
         amp=a(i+nadd+np)
         vel=a(i+nadd)+xoff
         w=(x-vel)/sig
         gaus=exp(-w*w/2.)/sqrt(2.*pi*sig*sig)
         yadd=amp*gaus*(1.+h3*fh3(w)+h4*fh4(w))
         yfit=yfit+yadd
         dyda(1)=dyda(1)+
     $        (-yadd)/sig+yadd*w*(x-vel)/sig/sig+amp*gaus*(
     $        -h3*dfh3(w)*(x-vel)/sig/sig-h4*dfh4(w)*(x-vel)/sig/sig)
         dyda(2)=dyda(2)+amp*gaus*fh3(w)
         dyda(3)=dyda(3)+amp*gaus*fh4(w)
         dyda(5)=dyda(5)+
     $        yadd*w/sig+amp*gaus*(-h3*dfh3(w)/sig-h4*dfh4(w)/sig)
         dyda(i+nadd+np)=yadd/amp
         dyda(i+nadd)=1.
         dyda(i+nadd+np+np)=
     $        (-yadd)/sig+yadd*w*(x-vel)/sig/sig+amp*gaus*(
     $        -h3*dfh3(w)*(x-vel)/sig/sig-h4*dfh4(w)*(x-vel)/sig/sig)
c         print *,i,sig,dyda(i+nadd+np+np),gaus,w,x,vel
c         read *
      enddo

      return
      end

      function fh3(x)
      fh3=1./sqrt(6.)*(2.*sqrt(2.)*x*x*x-3.*sqrt(2.)*x)
      return
      end
      function fh4(x)
      fh4=1./sqrt(24.)*(4.*x*x*x*x-12.*x*x+3.)
      return
      end
      function dfh3(x)
      dfh3=1./sqrt(6.)*(6.*sqrt(2.)*x*x-3.*sqrt(2.))
      return
      end
      function dfh4(x)
      dfh4=1./sqrt(24.)*(16.*x*x*x-24.*x)
      return
      end

      SUBROUTINE mrqminb(x,y,sig,ndata,a,ia,ma,covar,alpha,nca,chisq,
     *alamda)
      INTEGER ma,nca,ndata,ia(ma),MMAX
      REAL alamda,chisq,a(ma),alpha(nca,nca),covar(nca,nca),
     *sig(ndata),x(ndata),y(ndata)
      PARAMETER (MMAX=100)
CU    USES covsrt,gaussj,mrqcof
      INTEGER j,k,l,m,mfit
      REAL ochisq,atry(MMAX),beta(MMAX),da(MMAX)
      SAVE ochisq,atry,beta,da,mfit
      if(alamda.lt.0.)then
        mfit=0
        do 11 j=1,ma
          if (ia(j).ne.0) mfit=mfit+1
11      continue
        alamda=0.001
        call mrqcofb(x,y,sig,ndata,a,ia,ma,alpha,beta,nca,chisq)
        ochisq=chisq
        do 12 j=1,ma
          atry(j)=a(j)
12      continue
      endif
      j=0
      do 14 l=1,ma
        if(ia(l).ne.0) then
          j=j+1
          k=0
          do 13 m=1,ma
            if(ia(m).ne.0) then
              k=k+1
              covar(j,k)=alpha(j,k)
            endif
13        continue
          covar(j,j)=alpha(j,j)*(1.+alamda)
          da(j)=beta(j)
        endif
14    continue
      call gaussjb(covar,mfit,nca,da,1,1)
      if(alamda.eq.0.)then
        call covsrt(covar,nca,ma,ia,mfit)
        return
      endif
      j=0
      do 15 l=1,ma
        if(ia(l).ne.0) then
          j=j+1
          atry(l)=a(l)+da(j)
        endif
15    continue
      call mrqcofb(x,y,sig,ndata,atry,ia,ma,covar,da,nca,chisq)
      if(chisq.lt.ochisq)then
        alamda=0.1*alamda
        ochisq=chisq
        j=0
        do 17 l=1,ma
          if(ia(l).ne.0) then
            j=j+1
            k=0
            do 16 m=1,ma
              if(ia(m).ne.0) then
                k=k+1
                alpha(j,k)=covar(j,k)
              endif
16          continue
            beta(j)=da(j)
            a(l)=atry(l)
          endif
17      continue
      else
        alamda=10.*alamda
        chisq=ochisq
      endif
      return
      END
      SUBROUTINE mrqcofb(x,y,sig,ndata,a,ia,ma,alpha,beta,nalp,chisq)
      INTEGER ma,nalp,ndata,ia(ma),MMAX
      REAL chisq,a(ma),alpha(nalp,nalp),beta(ma),sig(ndata),x(ndata),
     *y(ndata)
      PARAMETER (MMAX=100)
      INTEGER mfit,i,j,k,l,m
      REAL dy,sig2i,wt,ymod,dyda(MMAX)
      mfit=0
      do 11 j=1,ma
        if (ia(j).ne.0) mfit=mfit+1
11    continue
      do 13 j=1,mfit
        do 12 k=1,j
          alpha(j,k)=0.
12      continue
        beta(j)=0.
13    continue
      chisq=0.
      do 16 i=1,ndata
        call funcs(x(i),a,ymod,dyda,ma)
        sig2i=1./(sig(i)*sig(i))
        dy=y(i)-ymod
        j=0
        do 15 l=1,ma
          if(ia(l).ne.0) then
            j=j+1
            wt=dyda(l)*sig2i
            k=0
            do 14 m=1,l
              if(ia(m).ne.0) then
                k=k+1
                alpha(j,k)=alpha(j,k)+wt*dyda(m)
              endif
14          continue
            beta(j)=beta(j)+dy*wt
          endif
15      continue
        chisq=chisq+dy*dy*sig2i
16    continue
      do 18 j=2,mfit
        do 17 k=1,j-1
          alpha(k,j)=alpha(j,k)
17      continue
18    continue
      return
      END
      SUBROUTINE gaussjb(a,n,np,b,m,mp)
      INTEGER m,mp,n,np,NMAX
      REAL a(np,np),b(np,mp)
      PARAMETER (NMAX=100)
      INTEGER i,icol,irow,j,k,l,ll,indxc(NMAX),indxr(NMAX),ipiv(NMAX)
      REAL big,dum,pivinv
      do 11 j=1,n
        ipiv(j)=0
11    continue
      do 22 i=1,n
        big=0.
        do 13 j=1,n
          if(ipiv(j).ne.1)then
            do 12 k=1,n
              if (ipiv(k).eq.0) then
                if (abs(a(j,k)).ge.big)then
                  big=abs(a(j,k))
                  irow=j
                  icol=k
                endif
              else if (ipiv(k).gt.1) then
              
c                pause 'singular matrix in gaussj'
c                print *,'singular matrix in gaussj'
              endif
12          continue
          endif
13      continue
        ipiv(icol)=ipiv(icol)+1
        if (irow.ne.icol) then
          do 14 l=1,n
            dum=a(irow,l)
            a(irow,l)=a(icol,l)
            a(icol,l)=dum
14        continue
          do 15 l=1,m
            dum=b(irow,l)
            b(irow,l)=b(icol,l)
            b(icol,l)=dum
15        continue
        endif
        indxr(i)=irow
        indxc(i)=icol
c        if (a(icol,icol).eq.0.) pause 'singular matrix in gaussj'
c        if (a(icol,icol).eq.0.) print *,'singular matrix in gaussj'
        pivinv=1./a(icol,icol)
        a(icol,icol)=1.
        do 16 l=1,n
          a(icol,l)=a(icol,l)*pivinv
16      continue
        do 17 l=1,m
          b(icol,l)=b(icol,l)*pivinv
17      continue
        do 21 ll=1,n
          if(ll.ne.icol)then
            dum=a(ll,icol)
            a(ll,icol)=0.
            do 18 l=1,n
              a(ll,l)=a(ll,l)-a(icol,l)*dum
18          continue
            do 19 l=1,m
              b(ll,l)=b(ll,l)-b(icol,l)*dum
19          continue
          endif
21      continue
22    continue
      do 24 l=n,1,-1
        if(indxr(l).ne.indxc(l))then
          do 23 k=1,n
            dum=a(k,indxr(l))
            a(k,indxr(l))=a(k,indxc(l))
            a(k,indxc(l))=dum
23        continue
        endif
24    continue
      return
      END
      SUBROUTINE covsrt(covar,npc,ma,ia,mfit)
      INTEGER ma,mfit,npc,ia(ma)
      REAL covar(npc,npc)
      INTEGER i,j,k
      REAL swap
      do 12 i=mfit+1,ma
        do 11 j=1,i
          covar(i,j)=0.
          covar(j,i)=0.
11      continue
12    continue
      k=mfit
      do 15 j=ma,1,-1
        if(ia(j).ne.0)then
          do 13 i=1,ma
            swap=covar(i,k)
            covar(i,k)=covar(i,j)
            covar(i,j)=swap
13        continue
          do 14 i=1,ma
            swap=covar(k,i)
            covar(k,i)=covar(j,i)
            covar(j,i)=swap
14        continue
          k=k-1
        endif
15    continue
      return
      END
