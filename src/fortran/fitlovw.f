      parameter(narr=50200,nmax=50200,nlovm=600,ncmax=100,nfile=20)

      real xg(narr,1),x(nmax),g(nmax),t(nmax,nfile),temp(nmax)
      real xl(nlovm),yl0(nlovm),yl0p(nlovm),glog(nmax),fac(nlovm)
      real xout(nlovm),xscale(nlovm),rparam(7),xlb(nlovm),xub(nlovm)
      real gerr(nmax),xc(ncmax,2),go(nmax),tfrac(nfile),gp(nmax)
      real tt(nmax),xtemp(nmax),facnew(nlovm*4),ynew(nlovm*4)
      real gtemp(narr),xt(nmax),xdum(nmax),gdum(nmax),xorig(narr)
      real tn(nmax,nfile),xtemp2(nlovm),wave0(narr),flux0(narr)
      REAL RWKSP(19711),xsum(nmax)
      real*8 dw,dxi
      integer naxes(2),iparam(7),pgopen
      character fileg*80,cgal*80,chr
      external fcn

      common /cfunc/ t,glog,gerr,x
      common /cfunc2/ fac,c1,resd,xlam,xl,sigl0,amp1,
     $     np,nl,iskip,nt,ntot,ilog,isym,icoff,nlosvd,igh,igaus,
     $     coffi,coff2i
      COMMON /WORKSP/  RWKSP
      common /cscale/ scale

      parameter(cee=2.99e+5,pi=3.141593,ee=2.71828)
      data big,nl /1.e10,29/

      CALL IWKIN(19711)

      ilog=0
      ibad=1
      igaus=1

 100  call qc1('Galaxy Spectrum ','fitlov.def',fileg)
      call qc1('Galaxy Name ','fitlov.def',cgal)
      call qr2('Wavemin and scale ','fitlov.def',wavem,scale)
      call qr2('2nd and 3rd poly ','fitlov.def',p2,p3)
      call savdef

      open(unit=1,file=fileg,status='old')
      n=0
      do i=1,narr
         read(1,*,end=888) x1,x2
         n=n+1
         x(n)=x1
         go(n)=x2
      enddo
 888  continue
      close(1)

 200  call qc1('File of Templates  ','fitlov.def',fileg)
      call qr2('Wavemin and scale for Templates ',
     $     'fitlov.def',wavemt,scalet)
      call qr2('2nd and 3rd poly for Templates ','fitlov.def',p2,p3)
      call savdef
c      wavemt=log(4802.)
c      scalet=3.33564095e-5

      open(unit=2,file=fileg,status='old',err=200)
      nt=0
      do i=1,nfile
         read(2,*,end=666) fileg
         nt=nt+1
         open(unit=1,file=fileg,status='old')
         ntt=0
         do ip=1,narr
            read(1,*,end=889) x1,x2
            ntt=ntt+1
            xt(ntt)=x1
            t(ntt,nt)=x2
         enddo
 889     continue
         close(1)
      enddo
      if(nl+nt+1.gt.nlovm) print *,'Make nlovm bigger'
 666  continue
      close(2)

      do i=1,nt
         tfrac(i)=1./float(nt)
      enddo

      call qr2('Region ','fitlov.def',xmin,xmax)
      call qr2('Initial z and sigma ','fitlov.def',zgal,sigl)
      call qr1('xlambda ','fitlov.def',xlam)
      call qr1('Tolerance ','fitlov.def',ftol)
      call qr1('Constant offset ','fitlov.def',coff)
      call qi1('Symmetrize (1-yes) ','fitlov.def',isym)
      call savdef
c - icoff=1 implies both float, icoff=0 implies both fixed,
c   icoff=2 implies coff2 floats (offset)
      icoff=1
      coff2i=0.
      ceqwadd=0.8
      if(nint(coff).eq.666) then
         icoff=2
         coffi=0.
         coff=coffi
         coff2i=0.05
      endif
      if(nint(coff).eq.667) then
         call qr1('Constant offset2a ','fitlov.def',coffi)
         call qr1('Constant offset2b ','fitlov.def',coff2i)
         call savdef
         coff=coffi
         icoff=0
      endif
      igh=0
      if(sigl.lt.0) then
         sigl=-sigl
         igh=1
         isym=0
      endif

      if(abs(zgal).gt.4.) zgal=zgal/cee

c - re-bin the galaxy to the local rest frame

      do i=1,n
         x(i)=x(i)/(1.+zgal)
      enddo
      do i=1,n
         xp=x(1)+scale*(i-1)
         do j=1,n-1
            if(xp.ge.x(j).and.xp.lt.x(j+1)) then
               yp=go(j)+(go(j+1)-go(j))*(xp-x(j))/(x(j+1)-x(j))
            endif
         enddo
         if(xp.gt.x(n)) yp=go(n)
         g(i)=yp
         xtemp(i)=xp
      enddo
      do i=1,n
         x(i)=xtemp(i)
         go(i)=g(i)
      enddo

      gmin=big
      gmax=-big
      np=0
      do i=1,n
         if(x(i).gt.xmin.and.x(i).lt.xmax) then
            np=np+1
            x(np)=x(i)
            g(np)=go(i)
            gerr(np)=1.
c            if(g(np).le.0) then
            if(g(np).le.0.or.g(np).gt.1.3) then
               gerr(np)=big
            endif
            if(ilog.eq.1) then
               glog(np)=log(g(np))
            else
               glog(np)=g(np)
            endif
            xp=x(np)
            sum1=0.
            sum2=0.
            do it=1,nt
               diff=1e10
               do j=1,ntt-1
                  if(xp.ge.xt(j).and.xp.lt.xt(j+1)) then
                     tp=t(j,it)+(t(j+1,it)-t(j,it))*
     $                    (xp-xt(j))/(xt(j+1)-xt(j))
                     if(t(j,it).le.0.or.t(j+1,it).le.0) then
                        tp=1
                     endif
                     goto 567
                  endif
               enddo
               gerr(np)=big
               if(xp.lt.t(1,it)) tp=1
               if(xp.gt.t(ntt,it)) tp=1
 567           continue
               sum1=sum1+tfrac(it)*tp
               sum2=sum2+tfrac(it)
               tn(np,it)=tp
            enddo               
            tt(np)=sum1/sum2
            gmin=min(gmin,g(np))
            gmax=max(gmax,g(np))
c            gmin=min(gmin,g(np),tt(np))
c            gmax=max(gmax,g(np),tt(np))
         endif
      enddo
      do j=1,nt
         do i=1,np
            t(i,j)=tn(i,j)
         enddo
      enddo

c      xskip=x(np/2)*4.5*sigl/cee
      xskip=x(np/2)*0.5*sigl/cee
      xskip=0

      istat=pgopen('?')
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.2)
      
      gmin=max(0.,gmin)
      ymin=gmin
      ymax=gmax
      open(unit=11,file='limits.dat',status='old',err=999)
      read(11,*) ymin,ymax
 999  continue
      close(11)
      call pgenv(x(1)+xskip,x(np)-xskip,ymin,ymax,0,0)
      call pglabel('Wavelength (\(2078))','Normalized Flux','')
      call pgsch(1.7)
      call pgmtxt('T',-1.5,0.1,0.0,cgal)
      write(fileg,'(f4.2)') zgal
      call pgsch(0.2)
      do i=1,np
c         gtemp(i)=g(i)+1.
         gtemp(i)=g(i)
      enddo
      call pgslw(1)
      call pgsls(4)
c      call pgline(np,x,tt)
      call pgslw(3)
      call pgsls(1)

      do i=1,np
         xorig(i)=x(i)*(1.+zgal)
      enddo
      if(ibad.eq.1) call setbadreg(np,xorig,gerr,-1)
      if(ibad.eq.1) call setbadreg(np,x,gerr,1)
      call plotline(np,x,gtemp,gerr)
c      call pgsci(2)
c      call pgline(ntt,xt,t(1,1))
      call pgslw(4)

      res=scale
      resd=res**3
      iskip=xskip/res
      c1=1.-x(1)/scale

c - make the initial LOV

      idum=-1
      nlosvd=nint(9.*sigl/scale*x(n/2)/cee)*2
c      nlosvd=nint(9.*sigl/scale*x(n/2)/cee)*1
      if(nlosvd.gt.nlovm*4) then
         print *,nlosvd,nlovm
c         pause 'Make nlovm for nlosvd bigger'
      endif
      xlmin=-4.5*sigl
      xlmax=4.5*sigl
c      xlmin=-3.0*sigl
c      xlmax=3.0*sigl
      sig2l=sigl*sigl
      scl=(xlmax-xlmin)/float(nl)
      ximax=0.
      ymin=big
      ymax=-big
      sum=0.
      do i=1,nl
         xl(i)=xlmin+(xlmax-xlmin)/float(nl-1)*float(i-1)
         yl0(i)=
     $        1./sqrt(2.*3.14)/sigl*exp(-(ximax-xl(i))**2/2./sig2l)
     $        *scl
         yl0(i)=1./float(nl)
c         yl0(i)=max(0.,yl0(i)*(1.+gasdev(idum)/5.))
         yl0p(i)=yl0(i)
         sum=sum+yl0p(i)
         fac(i)=(1.+xl(i)/cee)/scale
         xscale(i)=1.
c         xlb(i)=-1.
         xlb(i)=1.e-6
         xub(i)=1.
         ymin=min(ymin,yl0p(i))
         ymax=max(ymax,yl0p(i))
      enddo
      ymin=0.
      ymax=1./sqrt(2.*3.14)/sigl*scl
      amp1=(xlmax-xlmin)/float(nl)
      print *,iskip,xskip,sum,nlosvd
      iparam(1)=0

c - set weights and the constant offset

      sigl0=sigl
      ibnt=nl
      if(isym.eq.1) ibnt=(nl+1)/2
      if(igh.eq.1) then
         ibnt=5
         call setinitv(1.,ibnt,yl0,xscale,xlb,xub)
         if(igaus.eq.1) ibnt=3
      endif
      do i=1,nt
         yl0(ibnt+i)=tfrac(i)
         xscale(ibnt+i)=.1
         xlb(ibnt+i)=1.e-5
         xub(ibnt+i)=1.
      enddo
      yl0(ibnt+nt+1)=coff
      xscale(ibnt+nt+1)=.2
      xlb(ibnt+nt+1)=coff-ceqwadd
      xub(ibnt+nt+1)=coff+5.*ceqwadd
      coff2=coff2i
      yl0(ibnt+nt+2)=coff2
      xscale(ibnt+nt+2)=.2
      xlb(ibnt+nt+2)=-0.7
      xub(ibnt+nt+2)= 0.7

      nparam=ibnt+nt
      if(icoff.eq.1) nparam=nparam+2
      if(icoff.eq.2) then
         nparam=nparam+1
         yl0(ibnt+nt+1)=yl0(ibnt+nt+2)
         xscale(ibnt+nt+1)=xscale(ibnt+nt+2)
         xlb(ibnt+nt+1)=xlb(ibnt+nt+2)
         xub(ibnt+nt+1)=xub(ibnt+nt+2)
      endif

c - do it

      print *,'Number of bins and size : ',ibnt,xl(2)-xl(1),nparam
 
      call u4inf(iparam,rparam)

      if(ftol.lt.rparam(2)) then
         print *,'tolerance too low'
         print *,ftol,rparam(2)
      endif
      rparam(2)=ftol
      rparam(3)=ftol

      call bconfkg(fcn,nparam,yl0,0,xlb,xub,xscale,1.,iparam,rparam,
     $     xout,fvalue)

      print *
      print *,'Ntot and fvalue = ',ntot,fvalue,fvalue/float(np-2*iskip)

      if(isym.eq.1) then
         imid=(nl+1)/2
         do i=1,nl
            if(i.lt.imid) then
               xtemp(i)=xout(imid+1-i)
            else
               xtemp(i)=xout(i-imid+1)
            endif
         enddo
         do i=1,nt+2
            xtemp(nl+i)=xout((nl+1)/2+i)
         enddo
         ibnt=nl
      else
         do i=1,nparam
            xtemp(i)=xout(i)
         enddo
         if(igh.eq.1) then
            xtemp(2)=xtemp(2)*sigl0
            xtemp(3)=xtemp(3)*sigl0
         endif
      endif

      if(icoff.eq.0) then
         coff=coffi
         coff2=coff2i
      endif
      if(icoff.eq.1) then
         coff=xtemp(ibnt+nt+1)
         coff2=xtemp(ibnt+nt+2)
      endif
      if(icoff.eq.2) then
         coff=coffi
         coff2=xtemp(ibnt+nt+1)
      endif

      open(unit=11,file='fitlov.out',status='unknown')
c      xtemp1=xout(ibnt+nt+1)*500.
      xtemp1=666.
      write(11,*) 0.,fvalue,xtemp1,ntot
c      write(11,*) fvalue,ntot,coff,coff2

c - shift the symmetric profile

c      if(isym.eq.1) then
c         xshift=xout(ibnt+nt+1)*500.
c         do i=1,nl
c            call xlinint(xl(i)-xshift,nl,xl,xtemp,xtemp2(i))
c         enddo
c         do i=1,nl
c            xtemp(i)=xtemp2(i)
c         enddo
c      endif

      if(igh.eq.0) then
         do i=1,nl
            yl0(i)=xtemp(i)
            yl0p(i)=yl0(i)
            write(11,*) i,xl(i),yl0p(i)
            ymin=min(ymin,yl0p(i))
            ymax=max(ymax,yl0p(i))
         enddo
      else
         amp=xtemp(1)*amp1
         vel=xtemp(2)
         sig=xtemp(3)
         if(igaus.eq.0) then
            h3=xtemp(4)
            h4=xtemp(5)
         else
            h3=0.
            h4=0.
         endif
         print *,amp,vel,sig,h3,h4
         den=sqrt(2.*pi)
         do i=1,nl
            w=(xl(i)-vel)/sig
            gaus=exp(-w*w/2.)/den
            yl0p(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))
            write(11,*) i,xl(i),yl0p(i)
            ymax=max(ymax,yl0p(i))
         enddo
      endif
      close(11)

c - plot out both galactic spectra

      do i=1,np
         gp(i)=0.
         xsum(i)=0.
      enddo

      sum2=0.
      do i=1,nt
         tfrac(i)=xtemp(ibnt+i)
         sum2=sum2+tfrac(i)
      enddo

      open(unit=11,file='fitlov.temp',status='unknown')
      suma=0.
      sumf=0.
      sumg=0.
      do i=1,nt
         write(11,*) tfrac(i)/sum2,i
      enddo
      close(11)

      print *,"coff (eqw and offset) = ",coff,coff2
      den=coff+1.
      do i=1,np
         sum1=0.
         do it=1,nt
            sum1=sum1+tfrac(it)*t(i,it)
         enddo
         tval=sum1/sum2
         temp(i)=(tval+coff)/(coff+1.)+coff2
      enddo
      call getnlosvd(nl,xl,yl0p,nlosvd,facnew,ynew)
      do i=1,np
         do j=1,nlosvd
            ip=nint(c1+x(i)*facnew(j))
            if(ip.gt.0.and.ip.le.np) then
               gp(ip)=gp(ip)+temp(i)*ynew(j)
               xsum(ip)=xsum(ip)+ynew(j)
            endif
         enddo
      enddo
      suml=0.
      do i=1,nlosvd
         suml=suml+ynew(i)
      enddo
      do i=1,np
         gp(i)=gp(i)*suml/xsum(i)
         sum1=0.
         do it=1,nt
            sum1=sum1+tfrac(it)*t(i,it)
         enddo
         tt(i)=sum1/sum2
      enddo

c      do i=1,np
c         gp(i)=gp(i)+1.
c      enddo
      call pgsci(2)
      call pgline(np,x,gp)
      call pgsci(1)
c      call pgline(np,x,tt)

      open(unit=13,file='ascii.out',status='unknown')
      rms=0.
      nrms=0
      do i=iskip+1,np-iskip
         ierr=0
         if(gerr(i).ge.1e9) ierr=1
         write(13,1301) x(i),g(i),gp(i),tt(i),ierr
         if(gerr(i).lt.1.e9) then
            rms=rms+(g(i)-gp(i))**2
            nrms=nrms+1
         endif
      enddo
      close(13)
 1301 format(1x,f10.3,3(2x,f7.3),1x,i1)
      open(unit=13,file='rms.out',status='unknown')
      print *,"RMS = ",sqrt(rms/float(nrms)),nrms
      write(13,*) sqrt(rms/float(nrms)),nrms,coff,coff2
      close(13)

      call pgsch(1.7)
      if(nt.eq.19) then
         write(fileg,"(i3)") nint(100.*suma)
c         call pgmtxt('T',-1.5,0.6,0.0,fileg(1:3)//'\% 1-2 Gyr')
         write(fileg,"(i3)") nint(100.*sumf)
c         call pgmtxt('T',-2.5,0.6,0.0,fileg(1:3)//'\% 3-6 Gyr')
         write(fileg,"(i3)") nint(100.*sumg)
c         call pgmtxt('T',-3.5,0.6,0.0,fileg(1:3)//'\% 8-13 Gyr')
      else
         write(fileg,"(i3)") nint(100.*suma)
c         call pgmtxt('T',-1.5,0.6,0.0,fileg(1:3)//'\% A V')
         write(fileg,"(i3)") nint(100.*sumf)
c         call pgmtxt('T',-2.5,0.6,0.0,fileg(1:3)//'\% F-G V')
         write(fileg,"(i3)") nint(100.*sumg)
c         call pgmtxt('T',-3.5,0.6,0.0,fileg(1:3)//'\% G-K III')
      endif

c - plot LOSVD

      call pgclos(istat)
      ymax=ymax*1.05
      istat=pgopen('fit.ps/ps')
      call pgsch(1.2)
      call pgenv(xl(1),xl(nl),0.,ymax,0,0)
      call pglabel('','',cgal)
      call pgslw(3)
      call pgline(nl,xl,yl0p)
      call pgsls(1)

      call pgend

 706  continue

      end
