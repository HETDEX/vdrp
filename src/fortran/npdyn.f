
      parameter(nmax=1000,npm=100,mm=2,nwk=nmax+6*(nmax*mm+1),mm2=mm*2)
      parameter(limit=1000,lenw=limit*4,nptm=400)

      real f(nmax),rfit(npm),xnew(npm),rl(nmax),fn(npm),fn1(npm)
      real xnewl(npm),dn(npm),dn1(npm),sigpsf(nmax)
      real work(lenw),mass(npm),nuv2(npm),isig(npm),sig(npm)
      real vr(npm),vsig(npm),yt(2),ieff,ydiff(nmax)
      real*8 y(nmax),wk(nwk),q(mm2),wx(nmax),drr(nmax)
      real*8 cf(nmax),dr(nmax),cfr(nmax),splder,val,deps
      real*8 cfm(nmax),cfv(nmax),cfis(nmax)
      integer iwork(limit)
      character filein*40,title*70,cband*1
      external funcf,funcm,funcv,funcs,funcs2,funci,derivs,bsstep

      common /ffunc/ dr,cf,xrmaxi,apowi,m,n
      common /mfunc/ drr,cfr,cfm,cfv,rp,xrmax,rmax,apow,np
      COMMON /path/ kmax,kount,dxsav,xp,yp
      common/cpsf/see

      parameter (pi=3.141593e0,gee=4.3e-3)

      data big,eps,istepmax /1.e30,1.e-5,6/
      data deps /1.d-7/
      data epsabs,epsrel,epsode /1.e-10,1.e-5,1.e-5/
      data apow,apowi /4.,3./
      data radd,nadd /100.,15/
c      data radd,nadd /1500.,15/
c      data radd,nadd /2000.,15/
c      data radd,nadd /500.,15/
      kmax=0
      ntot=0

      ibox=0
      ipsf=1
c      see=1.0
      see=0.15
      see=see/2.35

      isurf=0

 1069 format(a)

      np=npm

c -- Let's get the data

 1    call qc1('File with surface brightness ','npdyn.def',filein)
      open(unit=1,file=filein,status='old',err=1)

      call qr1('Distance ','npdyn.def',dist)
      call qd1('Enter val for f ','npdyn.def',val)
      call qr1('Lower radius cutoff ','npdyn.def',rlow)
      call qr1('R_e ','npdyn.def',reff)
      call qr2('M/L and BH ','npdyn.def',xmtol,xbh)
      call qc1('Which Band (UBVRIJHK) ','npdyn.def',cband)
      call qc1('Title ','npdyn.def',title)
      call savdef

      open(unit=10,file='npdyn.out',status='unknown')

      md=3
      if(val.eq.0.) md=2
      m=2
      mv=2
      
      if(isurf.ne.1) then
         call pgbegin(0,'?',3,2)
      else
         call pgbegin(0,'?',1,1)
         call pgpap(0.,1.)
      endif
      call pgscf(2)
      call pgsch(1.4)

      n=0
      rmin=big
      rmax=-big
      fmin=big
      fmax=-big
      idum=-1

      astopc=dist*4.83
      rdiff=big

c - get the surface brightness points

      if(cband.eq.'U') sun=5.61
      if(cband.eq.'B') sun=5.48
      if(cband.eq.'V') sun=4.83
      if(cband.eq.'R') sun=4.42
      if(cband.eq.'I') sun=4.08
      if(cband.eq.'J') sun=3.64
      if(cband.eq.'H') sun=3.30
      if(cband.eq.'K') sun=3.28
      print *,sun

c      fcon=5.6
      fcon=0.
      do i=1,nmax
         read(1,*,end=666) x1,x2
         if(x1.gt.rlow) then
         fcheck=10**(-0.4*(x2-sun))*(20626.5)**2 - fcon
         if(fcheck.le.0) then
            n=n+1
            rl(n)=log10(x1)
            f(n)=f(n-1)-1.5
            print *,"hi"
            goto 666
         endif
         n=n+1
         rl(n)=log10(x1)
c         rl(n)=x1
c         f(n)=log10(10**(-0.4*(x2-sun))*(20626.5)**2)
         f(n)=log10(10**(-0.4*(x2-sun))*(20626.5)**2 - fcon)
c         f(n)=log10(x2)
c         f(n)=x2
         endif
      enddo
 666  continue

      call sort2(n,rl,f)

      if(radd.gt.0..and.reff.gt.0..and.radd.gt.10**rl(n)) then
         bit=(log10(radd)-rl(n))/float(nadd)
         bit1=bit/10.
         ieff=10**f(n)/exp(-7.67*((10**rl(n)/reff)**0.25-1.))
         do i=1,nadd
            n=n+1
            if(i.eq.1) then
               rl(n)=rl(n-1)+bit1
            else
               rl(n)=rl(n-1)+bit
            endif
            f(n)=log10(ieff*exp(-7.67*((10**rl(n)/reff)**0.25-1.)))
         enddo
      endif

      do i=1,n
         y(i)=dble(f(i))
         dr(i)=dble(rl(i))
         wx(i)=1.d0
         rmin=min(rmin,rl(i))
         rmax=max(rmax,rl(i))
         fmin=min(fmin,f(i))
         fmax=max(fmax,f(i))
      enddo         

c - plot the Nuker profile and points

      fbit=(fmax-fmin)/10.
      fmin=fmin-fbit
      fmax=fmax+fbit
      rbit=(rmax-rmin)/10.
      rmaxp=rmax+rbit
      rminp=rmin-rbit

      call pgenv(rminp,rmaxp,fmin,fmax,0,30)
      call pgpoint(n,rl,f,21)
      call pglabel('R (arcsec)','\gS (L\D\(2281)\U/pc\U2\D)','')
      call pgmtext('T',-1.6,.5,.5,title)

c -- smoothing the surface brightness

      call gcvspl(dr,y,nmax,wx,1.d0,m,n,1,md,val,cf,nmax,wk,ier)
      if(ier.ne.0) print *,'ier= ',ier,' in f calc'

      print *
      print *,'N           : ',n
      print *,'GCV estimate: ',sngl(wk(1))
      print *,'MSR         : ',sngl(wk(2))
      print *,'dof estimate: ',sngl(wk(3))
      print *,'smoothing p : ',sngl(wk(4))
      print *,'tmse est    : ',sngl(wk(5))
      print *,'GM error    : ',sngl(wk(6))
      sum=0.
      sum2=0.
      den=.01**2
      do i=1,n-1
         sum=sum+(sngl(splder(2,m,n,dble(rl(i)),dr,cf,i,q)))**2*
     $        (rl(i+1)-rl(i))
         sum2=sum2+(f(i)-sngl(splder(0,m,n,dble(rl(i)),dr,cf,i,q)))**2
     $        /den/f(i)**2
      enddo
      sum=sum+(sngl(splder(2,m,n,dble(rl(n)),dr,cf,n,q)))**2*
     $        (rl(n)-rl(n-1))
      sum2=sum2+(f(n)-sngl(splder(0,m,n,dble(rl(n)),dr,cf,n,q)))**2
     $     /den/f(n)**2
      print *,'Sums        : ',sum/(rl(n)-rl(1))**3,sum2/float(n)
      print *,title
      print *

      rdiff=big
      rdiff2=big
      rdiffp=big
      fn2min=big

      do i=1,np
         rfit(i)=rmin+(rmax-rmin)/float(np-1)*float(i-1)
         jin=i
         fn(i)=sngl(splder(0,m,n,dble(rfit(i)),dr,cf,jin,q))
         fn1(i)=sngl(splder(1,m,n,dble(rfit(i)),dr,cf,jin,q))
         fn2=sngl(splder(2,m,n,dble(rfit(i)),dr,cf,jin,q))
         if(fn2.lt.fn2min) then
            fn2min=fn2
            r2min=10**rfit(i)
         endif
         if(abs(10**(rfit(i))-0.1).lt.rdiff) then
            rdiff=abs(10**(rfit(i))-0.1)
            idiff=i
         endif
         if(abs(10**(rfit(i))-0.5).lt.rdiff2) then
            rdiff2=abs(10**(rfit(i))-0.5)
            idiff2=i
         endif
         if(abs(astopc*10**(rfit(i))-17.).lt.rdiffp) then
            rdiffp=abs(astopc*10**(rfit(i))-17.)
            idiffp=i
         endif
      enddo

c      do i=1,n
c         fp=sngl(splder(0,m,n,dble(rl(i)),dr,cf,jin,q))
c         ydiff(i)=10**f(i)-10**fp
c         print *,10**rl(i),(10**f(i)-10**fp),10**f(i)
c      enddo
c      do i=1,n
c         call biwgt(ydiff(i),10,xb,xs)
c         print *,xs/10**f(i)
c      enddo

      open(unit=11,file='r2min.out',status='unknown')
      write(11,*) r2min
      close(11)

      xrmax=rfit(np)+2.
      xrmaxi=xrmax+2.

c - plot the surface brightness
      
      call pgline(np,rfit,fn)

      if(isurf.eq.1) goto 667

c- getting the deprojected density

      rmax=rfit(np)+1.

      factint=2.303/pi/astopc

      do i=1,np
         istepm=istepmax
 978     continue
         s=0.
         do j=1,istepm
            sold=s
            call midsql2(funcf,rfit(i),rfit(i),rmax,s,j)
            if(abs((s-sold)/s).lt.eps) goto 765
         enddo
 765     continue
         xnew(i)=-s*factint
c         print *,i,istepm,rfit(i),xnew(i)
         if(xnew(i).le.0) then
            istepm=istepm-1
            goto 978
         endif
      enddo

      xmin=big
      xmax=-big

      do i=1,np
         xnewl(i)=log10(xnew(i))
         y(i)=dble(xnewl(i))
         drr(i)=dble(rfit(i))
         wx(i)=1.d0
         xmax=max(xmax,xnewl(i))
         xmin=min(xmin,xnewl(i))
         write(10,1001) 10**rfit(i),xnew(i),10**fn(i)
      enddo

c -- smoothing the density and getting the first deriv

      call gcvspl(drr,y,nmax,wx,1.d0,m,np,1,2,val*4.d0,cfr,nmax,wk,ier)
      if(ier.ne.0) print *,'ier= ',ier,' in dens calc'

      open(unit=11,file='slope.out',status='unknown')
      do i=1,np
         jin=i
         dn(i)=sngl(splder(0,m,np,dble(rfit(i)),drr,cfr,jin,q))
         dn1(i)=sngl(splder(1,m,np,dble(rfit(i)),drr,cfr,jin,q))
         fn2=sngl(splder(2,m,n,dble(rfit(i)),dr,cf,jin,q))
         write(11,*) 10**rfit(i),dn1(i),fn2
	 wx(i)=1.d0
      enddo
      close(11)

      xbit=(xmax-xmin)/10.
      xmin=xmin-xbit
      xmax=xmax+xbit

c - plot the density profiles

      call pgenv(rminp,rmaxp,xmin,xmax,0,30)
      call pgline(np,rfit,xnewl)
      call pglabel('r (arcsec)','\gn (L\D\(2281)\U/pc\U3\D)','')
      call pgmtext('T',-1.6,.5,.5,title)

c - plot the derivatives

      call pgenv(rminp,rmaxp,-3.,0.,0,10)
      call pglabel('r (arcsec)','dlog \gn /dlog r','')
      call pgline(np,rfit,dn1)
      call pgmtext('T',-1.6,.5,.5,title)

      sum=0.
      do i=idiff,idiff2
         sum=sum+dn1(i)
      enddo
      sum=sum/float(idiff2-idiff+1)

c - get the mass

      open(unit=11,file='mass.out',status='unknown')
      const=4.*pi*astopc**3*xmtol
      ymin=big
      ymax=-big
      do i=1,np
         call qags(funcm,0.,10**rfit(i),epsabs,epsrel,ss,adserr,
     $           neval,ier,limit,lenw,last,iwork,work)
         if(ier.ne.0) print *,'ier = ',ier
         mass(i)=log10(const*ss+xbh)
c         print *,10**rfit(i),10**(mass(i)),10**rfit(i)*astopc
         y(i)=dble(mass(i))
         wx(i)=1.d0
         ymin=min(ymin,mass(i))
         ymax=max(ymax,mass(i))
         write(11,*) 10**rfit(i),10**(mass(i))
      enddo
      close(11)

c -- smoothing the mass

      call gcvspl(drr,y,nmax,wx,1.d0,m,np,1,md,val,cfm,nmax,wk,ier)
      if(ier.ne.0) print *,'ier= ',ier,' in mass calc'

c - plot the mass

      ybit=(ymax-ymin)/10.
      ymin=ymin-ybit
      ymax=ymax+ybit

      call pgenv(rminp,rmaxp,ymin,ymax,0,30)
      call pglabel('r (arcsec)','Mass (M\D\(2281)\U)','')
      call pgline(np,rfit,mass)
      call pgmtext('T',-1.6,.5,.5,title)

c - get nu x v^2

c - solve the difeq by integrating inward

c     first get the log slope of (m rho / r^2)

c estimate log slope of p(r) at large radii to find starting value of p(r)
      x1=10**rfit(np)
      x2=10**rfit(np-1)
      y1=-10**mass(np)*10**xnewl(np)/(x1*x1)
      y2b=-10**mass(np-1)*10**xnewl(np-1)/(x2*x2)
      gg=-log(y1/y2b)/log(x1/x2)
      if(gg.le.1.2) pause 'warning: outside pressure nearly divergent'
      yt(2)=-y1*x1/(gg-1.)
      vsig(np)=yt(2)
      do j=1,np-1
         x1=10**rfit(np+1-j)
         x2=10**rfit(np-j)
c
c note that although we integrate both p(r) and L(r), the starting value
c of L(r) at each step is taken from the previous integration, which is
c more accurate at small radii
c
         yt(1)=10**mass(np+1-j)
         h1=x2-x1
         call odeint2(yt,2,x1,x2,epsode,h1,hmin,nok,nbad,
     $        derivs,bsstep)
         vsig(np-j)=yt(2)
      enddo
      ymin=big
      ymax=-big
      do j=1,np
c now convert from pressure to rms velocity
         vsig(j)=sqrt(gee/astopc*vsig(j)/10**xnewl(j))
         ymin=min(ymin,vsig(j))
         ymax=max(ymax,vsig(j))
      enddo

c- do it the other way
       
      const=2.303*gee/astopc
      do i=1,np
         call qags(funcv,rfit(i),rmax,epsabs,epsrel,ss,adserr,
     $        neval,ier,limit,lenw,last,iwork,work)
         if(ier.ne.0) print *,'ier = ',ier
         nuv2(i)=log10(const*ss)
         vr(i)=sqrt(10**nuv2(i)/10**(xnewl(i)))
         y(i)=dble(xnewl(i)+2.*log10(vsig(i)))
         wx(i)=1.d0
      enddo

c -- smoothing v

      call gcvspl(drr,y,nmax,wx,1.d0,m,np,1,md,val,cfv,nmax,wk,ier)
      if(ier.ne.0) print *,'ier= ',ier,' in nu v^2 calc'

c - plot v

      ybit=(ymax-ymin)/10.
      ymin=ymin-ybit
      ymax=ymax+ybit

      call pgenv(rminp,rmaxp,ymin,ymax,0,10)
      call pglabel('r (arcsec)','v\Dr\U (km/s)','')
c      call pgline(np,rfit,vr)
      call pgmtext('T',-1.6,.5,.5,title)
      call pgline(np,rfit,vsig)

c - check that I is the same

      const=2.303*2.*astopc
      do i=1,np
         rp=rfit(i)
         s=0.
         do j=1,istepmax
            sold=s
            call midsql2(funci,rfit(i),rfit(i),rmax,s,j)
            if(abs((s-sold)/s).lt.eps) goto 766
         enddo
 766     continue
      enddo

c - get I x sig^2

      const=2.303*2.*astopc
      ymin=big
      ymax=-big
      do i=1,np
         rp=rfit(i)
         s=0.
         do j=1,istepmax
            sold=s
            call midsql2(funcs,rfit(i),rfit(i),rmax,s,j)
            if(abs((s-sold)/s).lt.eps) goto 767
         enddo
 767     continue
c         call qags(funcs2,rfit(i),rmax,epsabs,epsrel,ss,adserr,
c     $        neval,ier,limit,lenw,last,iwork,work)
c         if(ier.ne.0) print *,'ier = ',ier
         isig(i)=log10(const*s)
         sig(i)=sqrt(10**isig(i)/10**fn(i))
         y(i)=dble(isig(i))
         wx(i)=1.d0
c         sig2(i)=sqrt(2.*2.303*gee*ss/10**fn(i))
c         print *,i,sig(i),sig2(i),sig2(i)/sig(i)
c         print *,rfit(i),sig(i)
         ymin=min(ymin,sig(i))
         ymax=max(ymax,sig(i))
      enddo

c -- smoothing I x sig^2

      call gcvspl(drr,y,nmax,wx,1.d0,m,np,1,md,val,cfis,nmax,wk,ier)
      if(ier.ne.0) print *,'ier= ',ier,' in I sig^2 calc'

c - plot sigma

      ybit=(ymax-ymin)/20.
      ymin=ymin-ybit
      ymax=ymax+ybit

      call pgenv(rminp,rmaxp,ymin,ymax,0,10)
      call pglabel('R (arcsec)','\gs (km/s)','')
      call pgline(np,rfit,sig)
      call pgmtext('T',-1.6,.5,.5,title)

c - get the PSF-corrected sigma

      if(ipsf.eq.1) then

         npsf=150
         rpsf=3.
         area=(2.*rpsf/float(npsf))**2
         do i=1,np
            rp=10**rfit(i)
            xbmin=rp-rpsf
            xbmax=rp+rpsf
            ybmin=-rpsf
            ybmax=+rpsf
            xbstep=(xbmax-xbmin)/float(npsf)
            ybstep=xbstep
            sum2=0.
            sum1=0.
            do xb=xbmin,xbmax,xbstep
               do yb=ybmin,ybmax,ybstep
                  rad=sqrt(xb*xb+yb*yb)
                  radl=log10(rad)
                  grad=sqrt((xb-rp)**2+yb*yb)
                  call psf(grad,psfv)
                  if(rad.gt.10**rfit(1)) then
                     radl=log10(rad)
                  else
                     radl=rfit(1)
                  endif
                  surfb=sngl(splder(0,m,n,dble(radl),dr,cf,n/2,q))
                  surfbs=sngl(splder(0,m,np,dble(radl),drr,cfis,np/2,q))
                  sum1=sum1+area*psfv*10**surfb
                  sum2=sum2+area*psfv*10**surfbs
               enddo
            enddo
            sigpsf(i)=sqrt(sum2/sum1)
         enddo
      endif

      call pgsls(4)
      call pgline(np,rfit,sigpsf)
      call pgsls(1)
      open(unit=11,file='sigma.out',status='unknown')
      do i=1,np
         write(11,*) 10**rfit(i),sig(i),sigpsf(i)
      enddo
      close(11)

      if(ibox.eq.1) then

c - get the sigma in a 2" x 2" box

      nbox=25
      dia=2.
      xmin=-dia
      hdia=dia/2.
      dadd=-2.*xmin/float(nbox-1)
      see=1./2.35
      nsee=10
      seetot=2.*3.*see
      seemin=-3.*see
      sadd=seetot/float(nsee-1)
      gc=(6.*see/float(nsee-1))**2/2./pi/see/see
      gden=2.*see*see
      sum1=0.
      sum2=0.
      do i=1,nbox
         xr1=xmin+float(i-1)*dadd
         do j=1,nbox
            yr1=xmin+float(j-1)*dadd
            rad=sqrt(xr1*xr1+yr1*yr1)
            if(rad.gt.10**rfit(1)) then
               rad=log10(rad)
            else
               rad=rfit(1)
            endif
            surfb=sngl(splder(0,m,n,dble(rad),dr,cf,n/2,q))
            surfbs=sngl(splder(0,m,np,dble(rad),drr,cfis,np/2,q))
            do ir=1,nsee
               xr2=xr1+seemin+float(ir-1)*sadd
               if(xr2.lt.hdia) then
                  do jr=1,nsee
                     yr2=yr1+seemin+float(jr-1)*sadd
                     if(yr2.lt.hdia) then
                        radg2=(xr2-xr1)**2+(yr2-yr1)**2
                        gval=gc*exp(-radg2/gden)
                        sum1=sum1+gval*10**surfb
                        sum2=sum2+gval*10**surfbs
                     endif
                  enddo
               endif
            enddo
         enddo
      enddo

      sig0=sqrt(sum2/sum1)
      surfb=sngl(splder(0,m,n,dble(1.),dr,cf,n/2,q))
      surfbs=sngl(splder(0,m,np,dble(1.),drr,cfis,np/2,q))
      sig10=sqrt(10**surfbs/10**surfb)
      print *,sig0/sig10

      endif
         

 667  continue
      call pgend

 1001 format(3(1x,1pe9.3))
 1002 format(a10,2(2x,f7.2),2x,f5.2,2x,f6.2)
 1301 format(a10,2(2x,f7.2))
 1601 format(a10,2x,f10.2,1x,f5.2,2x,f7.2,2x,f5.2)

      end

      subroutine psf(r,v)
      parameter(pi=3.141593e0)
      common/cpsf/see

c -- Gaussian PSF

      gc=0.5/pi/see/see
      gden=2.*see*see
      v=gc*exp(-r*r/gden)

      return
      end
