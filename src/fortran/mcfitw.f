
      parameter(narr=50200,nmax=50200,nlovm=600,ncmax=100,nfile=20)
      parameter(nsimt=1000)

      real xg(narr,1),x(nmax),g(nmax),t(nmax,nfile),temp(nmax)
      real xl(nlovm),yl0(nlovm),yl0p(nlovm),glog(nmax),fac(nlovm)
      real xout(nlovm),xscale(nlovm),rparam(7),xlb(nlovm),xub(nlovm)
      real gerr(nmax),go(nmax),tfrac(nfile),gp(nmax)
      real tt(nmax),tfraco(nfile),yin(nsimt),xsim(nsimt,nlovm)
      real fsim(nsimt),xtemp(nmax),facnew(nlovm*4),ynew(nlovm*4)
      real tn(nmax,nfile),xt(nmax),xorig(narr),xtemp2(nlovm)
      real wave0(narr),flux0(narr),rmsa(narr),xsum(narr)
      REAL RWKSP(19711)
      real*8 dw,dxi
      integer naxes(2),iparam(7)
      character cname*80,fileg*80
      external fcn

      common /cfunc/ t,glog,gerr,x
      common /cfunc2/ fac,c1,resd,xlam,xl,sigl0,amp1,
     $     np,nl,iskip,nt,ntot,ilog,isym,icoff,nlosvd,igh,igaus,
     $     coffi,coff2i
      COMMON /WORKSP/  RWKSP
      common /cscale/ scale

      parameter(cee=2.99e+5,pi=3.141593,ee=2.71828)
      data big,nl /1.e10,29/
c      data ntotval /3000/
      data ntotval /1000/
      data ibin /1/

      CALL IWKIN(19711)
      ilog=0
      ibad=1
      igaus=1
      irmsd=0

 100  call qc1('Galaxy Spectrum ','fitlov.def',cname)
      call qr2('Wavemin and scale ','fitlov.def',wavem,scale)
      call qr2('2nd and 3rd poly ','fitlov.def',p2,p3)
      call savdef

      open(unit=1,file=cname,status='old')
      n=0
      do i=1,narr
         read(1,*,end=888) x1,x2
         n=n+1
         x(n)=x1
         go(n)=x2
      enddo
 888  continue
      close(1)

 200  call qc1('File of Templates  ','fitlov.def',cname)
      call qr2('Wavemin and scale for Templates ','fitlov.def',
     $     wavemt,scalet)
      call qr2('2nd and 3rd poly for Templates ','fitlov.def',p2,p3)
      call savdef

      open(unit=2,file=cname,status='old',err=200)
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
      call qr1('RMS val ','fitlov.def',rmsval)
      call qi1('Symmetrize (1-yes) ','fitlov.def',isym)
      call qi1('Num of Simulations ','fitlov.def',nsim)
      write(*,"('prefix for output: '$)")
      read(*,2001) cname
 2001 format(a80)
      do i=1,40
         if(cname(i:i).eq.' ') then
            nname=i-1
            goto 966
         endif
      enddo
 966  continue
      irms=0
      if(rmsval.lt.0) then
         rmsval=-rmsval
         irms=1
      endif

      if(nsim.gt.nsimt) print *,'too many sims'
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
         ntotval=20
      endif

      if(abs(zgal).gt.9.) zgal=zgal/cee

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
            if(g(np).le.0) then
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
               do j=1,ntt-1
                  if(xp.ge.xt(j).and.xp.lt.xt(j+1)) then
                     tp=t(j,it)+(t(j+1,it)-t(j,it))*
     $                    (xp-xt(j))/(xt(j+1)-xt(j))
                     if(t(j,it).le.0.or.t(j+1,it).le.0) then
                        tp=1
c                        gerr(np)=big
                     endif
                     goto 567
                  endif
               enddo
               gerr(np)=big
               if(xp.lt.t(1,it)) tp=1
               if(xp.gt.t(nr,it)) tp=1
 567           continue
               sum1=sum1+tfrac(it)*tp
               sum2=sum2+tfrac(it)
               tn(np,it)=tp
            enddo               
            tt(np)=sum1/sum2
            gmin=min(gmin,g(np),tt(np))
            gmax=max(gmax,g(np),tt(np))
         endif
      enddo
      do j=1,nt
         do i=1,np
            t(i,j)=tn(i,j)
         enddo
      enddo

      do i=1,np
         go(i)=g(i)
      enddo
      do i=1,nt
         tfraco(i)=tfrac(i)
      enddo

      do i=1,np
         xorig(i)=x(i)*(1.+zgal)
      enddo
      if(ibad.eq.1) call setbadreg(np,xorig,gerr,-1)
      if(ibad.eq.1) call setbadreg(np,x,gerr,1)

      call pgbegin(0,'?',2,2)
      call pgscf(2)
      call pgsch(1.2)
      call pgask(.false.)
      
      sumfv=0.
      rms=0.
      idum=-1
      nskip=0
      do isim=1,nsim
 667     continue
         ntot=0
         do i=1,np
            if(irmsd.eq.1.and.isim.gt.1) rms=rmsa(i)
            g(i)=go(i)+rms*gasdev(idum)
            if(ilog.eq.1) then
               glog(i)=log(g(i))
            else
               glog(i)=g(i)
            endif   
         enddo
      
         res=scale
         resd=res**3
c         xskip=x(np/2)*4.5*sigl/cee
         xskip=x(np/2)*0.5*sigl/cee
         xskip=0.
         iskip=xskip/res
         c1=1.-x(1)/scale

         if(isim.le.4) then
            call pgenv(x(1)+xskip,x(np)-xskip,0.5,1.2,0,0)
            call pgline(np,x,g)
         endif

c - make the initial LOV

         nlosvd=nint(9.*sigl/scale*x(n/2)/cee)*2
         if(nlosvd.gt.nlovm*4) then
            print *,nlosvd,nlovm
            print *,'Make nlovm for nlosvd bigger'
         endif
         xlmin=-4.5*sigl
         xlmax=4.5*sigl
         sig2l=sigl*sigl
         scl=(xlmax-xlmin)/float(nl)
         ximax=0.
         ymin=big
         ymax=-big
         sum=0.
         do i=1,nl
            xl(i)=xlmin+(xlmax-xlmin)/float(nl-1)*float(i-1)
            yl0(i)=
     $           1./sqrt(2.*3.14)/sigl*exp(-(ximax-xl(i))**2/2./sig2l)
     $           *scl
            yl0(i)=1./float(nl)
            if(isim.gt.1) yl0(i)=yl0(i)*(1.+0.2*gasdev(idum))
c            print *,1./float(nl),yl0(i)
            yl0(i)=max(0.,yl0(i))
            yl0p(i)=yl0(i)
            sum=sum+yl0p(i)
            fac(i)=(1.+xl(i)/cee)/scale
            ymin=min(ymin,yl0p(i))
            ymax=max(ymax,yl0p(i))
         enddo
         ymin=0.
         ymax=1./sqrt(2.*3.14)/sigl*scl
         amp1=(xlmax-xlmin)/float(nl)

c - set weights and the constant offset

         do i=1,nl
            xscale(i)=1.
            xlb(i)=1.e-6
            xub(i)=1.
         enddo
         sigl0=sigl
         ibnt=nl
         if(isym.eq.1) ibnt=(nl+1)/2
         if(igh.eq.1) then
            ibnt=5
c            sigl0=sigl*0.75
            call setinitv(1.,ibnt,yl0,xscale,xlb,xub)
            if(igaus.eq.1) ibnt=3
         endif
         do i=1,nt
            yl0(ibnt+i)=tfraco(i)
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

         call u4inf(iparam,rparam)
c         iparam(4)=100
            
         rparam(2)=ftol
         rparam(3)=ftol
         call bconfkg(fcn,nparam,yl0,0,xlb,xub,xscale,1.,
     $        iparam,rparam,xout,fvalue)
         
         if(ntot.lt.ntotval) then
            if(isim.eq.1) then
               print *,ntot,ntotval
               write(*,"('Better check this out')")
               goto 767
            endif
            nskip=nskip+1
c            write(*,"('Skipping this one')")
            goto 667
         endif

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

c - shift the symmetric profile

c         if(isym.eq.1) then
c            xshift=xout(ibnt+nt+1)*500.
c            do i=1,nl
c               call xlinint(xl(i)-xshift,nl,xl,xtemp,xtemp2(i))
c            enddo
c            do i=1,nl
c               xtemp(i)=xtemp2(i)
c            enddo
c            ibnt=nl
c         endif
            
         if(igh.eq.0) then
            do i=1,nl
               xout(i)=xtemp(i)
               yl0p(i)=xout(i)
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
            den=sqrt(2.*pi)
            do i=1,nl
               w=(xl(i)-vel)/sig
               gaus=exp(-w*w/2.)/den
               xout(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))
               yl0p(i)=xout(i)
               ymax=max(ymax,yl0p(i))
            enddo
            xshift=vel
            xout(nl+nt+1)=xshift
         endif

         do i=1,nt
            tfrac(i)=xtemp(ibnt+i)
         enddo

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

         ymax=ymax*1.05

         if(isim.eq.1) then
            coffi=coff
            coff2i=coff2
            icoff=0
         endif

c - plot out both galactic spectra

         sum2=0.
         do i=1,nt
            sum2=sum2+tfrac(i)
         enddo

         do i=1,nt
             xout(nl+i)=tfrac(i)/sum2
         enddo
         
         do i=1,nl+nt+1
            xsim(isim,i)=xout(i)
         enddo
         fsim(isim)=fvalue

         do i=1,np
            gp(i)=0.
            xsum(i)=0.
         enddo

c         print *,"coff (eqw and offset) = ",coff,coff2
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
         enddo
         if(isim.le.4) then
            call pgsci(5)
            call pgline(np,x,gp)
            call pgsci(1)
         endif
          
         sum=0.
         nrms=0
         do i=iskip+1,np-iskip
            sum=sum+(g(i)-gp(i))**2/gerr(i)**2
            nrms=nrms+1
            if(isim.eq.1) then
               go(i)=gp(i)
               rmsa(i)=sqrt((g(i)-gp(i))**2/gerr(i)**2)
            endif
         enddo
         if(isim.eq.1) then
            do i=1,iskip
               rmsa(i)=rmsa(iskip+1)
            enddo
            do i=np-iskip+1,np
               rmsa(i)=rmsa(np-iskip)
            enddo
         endif
         sum=sqrt(sum/float(nrms))
         sumfv=sumfv+fvalue
         avgf=sumfv/float(isim)
         if(isim.eq.1) then
            rms=rmsval
            fvalue1=fvalue
            call asmooth(np,rmsa,ibin)
            sumr=0.
            do i=iskip+1,np-iskip
               sumr=sumr+rmsa(i)
            enddo
            sumr=sumr/float(nrms)
            write(*,*)
            write(*,*)
            write(*,*) 'The Average RMS for ',cname(1:nname),
     $           sum,1./sum
            write(*,*)
      write(*,5001) '            #   Niter  Fvalue  Fvalue1   RMS_Fit',
     $     '  RMS_O    RMS_New   Nskip  Voff'
 5001 format(2a)
c         elseif(abs(fvalue1-avgf)/fvalue1.gt.0.1.and.irms.eq.0) then
c            if(fvalue1.gt.avgf) rms=rms*(1.+0.05)
c            if(fvalue1.lt.avgf) rms=rms*(1.-0.05)
c         endif            
         elseif(abs(rmsval-sum)/fvalue1.gt.0.1.and.irms.eq.0) then
            if(rmsval.gt.sum) rms=rms*(1.+0.05)
            if(rmsval.lt.sum) rms=rms*(1.-0.05)
         endif            
         write(*,1001) isim,ntot,fvalue,fvalue1,sum,rmsval,rms,nskip
     $        ,xshift,sig
      enddo
 1001 format('Iteration: ',i4,1x,i5,5(1x,f8.4),3x,i4
     $     ,1x,f8.2,1x,f8.2)

      print *,'STATISTICS FOR ',cname(1:nname)
      print *,'FVALUE STAT ',sumfv/float(nsim),fvalue1
      print *,'RMS ',rms

      open(unit=12,file=cname(1:nname)//'.sim',status='unknown')
      write(12,*) nsim,nl,nt
      write(12,*) (fsim(i),i=1,nsim)
      open(unit=11,file=cname(1:nname)//'.mcfit',status='unknown')
      i16=max(1,nint(float(nsim)*.16))
      i84=min(nsim,nint(float(nsim)*.84))
      i5=max(1,nint(float(nsim)*.05))
      i95=min(nsim,nint(float(nsim)*.95))
      write(11,*) nl,nt
      do i=1,nl+nt+1
         do j=1,nsim
            yin(j)=xsim(j,i)
         enddo
         write(12,*) (yin(j),j=1,nsim)
         call biwgt(yin,nsim,xb,xs)
         if(i.le.nl) then
            write(11,*) xl(i),xb,yin(i5),yin(i16),yin(i84),yin(i95)
         else
            write(11,*) xb,yin(i5),yin(i16),yin(i84),yin(i95)
         endif
      enddo
      close(11)
      close(12)

 767  continue
      call pgend
 706  continue

      end
      
      subroutine asmooth(n,x,ibin)
      real x(n),xd(100000),xin(10000)
      ib1=(ibin-1)/2
      xib=float(ibin)
      do j=1,n
         istart=max(1,j-ib1)
         iend=istart+ibin-1
         if(iend.gt.n) then
            iend=n
            istart=n-ibin+1
         endif
         ns=0
         do is=istart,iend
            ns=ns+1
            xin(ns)=x(is)
         enddo
         call biwgt(xin,ns,xb,xs)
         xd(j)=xb
      enddo
      do i=1,n
         x(i)=xd(i)
      enddo
      return
      end
