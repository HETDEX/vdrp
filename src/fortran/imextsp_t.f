
      parameter (narrm=5048)
      real xd(narrm,narrm),wave(narrm),xp(narrm),xerr2(narrm)
      real xsp(narrm),waved(narrm),wtp(narrm),tp(narrm)
      real xsky(narrm,narrm),sky(narrm),xspa(narrm,narrm)
      real xtrace(narrm,narrm),trace(narrm),xspab(narrm,narrm)
      real xflag(narrm,narrm),flag(narrm),x2dsub(narrm, narrm)
      real wa(narrm),fa(narrm),f2f(narrm),xerr(narrm),xin(100000)
      integer naxes(2),iwave(narrm)
      integer ixbs(1000),ixbe(1000),iybs(1000),iybe(1000)
      character file1*130,file2*130,file3*130,amp*14,a1*14
      logical simple,extend,anyf

      convfac=(6.626e-27)*(3.e18)/360./5.e5
      read *,file1
      read *,ifib,w0,ww
      read *,file2
      read *,file3
      amp=file3(52:65)

c - get the bad pixel list
      open(unit=1,file='/home/00115/gebhardt/badpix.new',status='old')
      nbad=0
      do i=1,1000
         read(1,*,end=444) a1,i2,i3,i4,i5
         if(a1.eq.amp) then
            nbad=nbad+1
            ixbs(nbad)=i2
            ixbe(nbad)=i3
            iybs(nbad)=i4
            iybe(nbad)=i5
         endif
      enddo
 444  continue
      close(1)

      wmin=w0-ww
      wmax=w0+ww
      wsp=2.
      nw=nint((wmax-wmin)/wsp)+1
      do i=1,nw
         wave(i)=wmin+float(i-1)*wsp
      enddo

      open(unit=1,file=file2,status='old')
      ntp=0
      do i=1,narrm
         read(1,*,end=676) x1,x2
         ntp=ntp+1
         wtp(ntp)=x1
         tp(ntp)=x2
      enddo
 676  continue
      close(1)
      open(unit=1,file=file3,status='old',err=678)
      na=0
      do i=1,narrm
         read(1,*,end=677) x1,x2
         na=na+1
         wa(na)=x1
         fa(na)=x2
      enddo
 677  continue
      close(1)
      goto 679
 678  continue
      na=3
      wa(1)=3500.
      fa(1)=1.
      wa(2)=4500.
      fa(2)=1.
      wa(3)=5500.
      fa(3)=1.
      write(*,*) "Amp Norm does not exist: ",file3
 679  continue

      im1=0
      ier=0
      iread=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the wavelength                                                                                           
      iext=12
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      n=0
      wmin=w0-ww
      wmax=w0+ww
      do i=1,ncol
         w=xd(i,ifib)
         if(w.gt.wmin.and.w.lt.wmax) then
            n=n+1
            waved(n)=w
            iwave(n)=i
         endif
      enddo

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the sky-subtracted spectrum
      iext=16
c      iext=17
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

c - parameters for empirical error estimate
c      nerrsp=51
c      nerrsp2=11
c      nerrf=9
      nerrsp=51
      nerrsp2=3
      nerrf=13
      nerrh=nint(float(nerrsp)/2.)
      nerrh2=nint(float(nerrsp2)/2.)
      nerrfh=nint(float(nerrf)/2.)
      do i=1,n
         xsp(i)=xd(iwave(i),ifib)
c - get the errors both spectrally and across fibers
c - spectrally:
         iemin=max(1,i-nerrh)
         iemax=min(n,iemin+nerrsp)
         nin=0
         do ie=iemin,iemax
            if(ie.lt.(i-1).or.ie.gt.(i+1)) then
               nin=nin+1
               xin(nin)=xd(iwave(ie),ifib)
            endif
         enddo
         call biwgt(xin,nin,xb,xerrsp)
         call moment(xin,nin,ave,adev,sdevs,var,skew,curt)
c - across fibers:
         iemin=max(1,i-nerrh2)
         iemax=min(n,iemin+nerrsp2)
         ifmin=max(1,ifib-nerrfh)
         ifmax=min(112,ifmin+nerrf)
         nin=0
         do ie=iemin,iemax
            do ifc=ifmin,ifmax
               if(ifc.ne.ifib) then
                  nin=nin+1
                  xin(nin)=xd(iwave(ie),ifc)
               endif
            enddo
         enddo
         call biwgt(xin,nin,xb,xerrf) 
         call moment(xin,nin,ave,adev,sdevf,var,skew,curt)
c         xerr(i)=max(xerrsp,xerrf)
         xerr(i)=max(sdevs,sdevf)
      enddo

c - now get the error from the error frame, if it exists

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the error frame
      iext=18
      ierr2=1
      call ftmahd(im1,iext,ihd,ier)
      if(ier.eq.0) then
         call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
         ncol=naxes(1)
         nrow=max(1,naxes(2))
         call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
         do i=1,n
            xval=xd(iwave(i),ifib)
            if(xval.gt.0.and.xval.lt.1e10) then
               xerr2(i)=xd(iwave(i),ifib)
            else
               xerr2(i)=0.
            endif   
         enddo
      else
         ier=0
         ierr2=0
         xerr2(i)=0.
c         print *,"Using local error estimate"
      endif   
      call ftclos(im1,ier)

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the F2F                                                                                                  
      iext=14
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the Skymod
      iext=17
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xsky,anyf,ier)
      call ftclos(im1,ier)

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the trace
      iext=13
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xtrace,anyf,ier)
      call ftclos(im1,ier)

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the 2d sky-subtracted frame
      iext=3
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,x2dsub,anyf,ier)
      call ftclos(im1,ier)

c - get rid of the horizontal streaks
      nrowe=3
      nrt=nrowe*2+1
      nbin=11
      nhalf=(nbin-1)/2
      diffmax=10.
      xsmin=11.
      do i=1,n
         trace(i)=xtrace(iwave(i),ifib)
         ixtr=nint(trace(i))
         do j=1,nrt
            nr=ixtr-nrowe-1+j
            xspa(j,i)=x2dsub(iwave(i),nr)
         enddo
      enddo      
      do j=1,nrt
         do i=1,n
            is=max(1,i-nhalf)
            ie=min(n,i+nhalf)
            xsum=0.
            do ip=is,ie
               xsum=xsum+xspa(j,ip)
            enddo
            xspab(j,i)=xsum
         enddo
      enddo
      do i=1,n
         ixtr=nint(trace(i))
         nin=0
         do j=1,nrt
            nin=nin+1
            xin(nin)=xspab(j,i)
         enddo
         call biwgt(xin,nin,xbcol,xscol)
         xscol=max(xsmin,xscol)
         do j=1,nrt
            diff=(xspab(j,i)-xbcol)/xscol
            if(diff.gt.diffmax) then
               nr=nint(trace(i)-nrowe-1+j)
               nbad=nbad+1
               ixbs(nbad)=i
               ixbe(nbad)=i
               iybs(nbad)=nr
               iybe(nbad)=nr
c               print *,j,i,nr,wave(i),diff,xspab(j,i)
            endif
         enddo
      enddo

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the flagged frame
      iext=2
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xflag,anyf,ier)
      call ftclos(im1,ier)

c - now add the the extra bad pixel list
      if(nbad.gt.0) then
         do ibad=1,nbad
            do ix=ixbs(ibad),ixbe(ibad)
               do iy=iybs(ibad),iybe(ibad)
                  xflag(ix,iy)=0.
               enddo
            enddo
         enddo
      endif

c      radflag=2.0
      radflagx=3.0
      radflagy=4.0
      do i=1,n
         xftf=xd(iwave(i),ifib)
         if(xftf.gt.0.1) then
            xsp(i)=xsp(i)/xd(iwave(i),ifib)
            xerr(i)=xerr(i)/xd(iwave(i),ifib)
            f2f(i)=xd(iwave(i),ifib)
            sky(i)=xsky(iwave(i),ifib)

c - flag if anything nearby is below 0
            trace(i)=xtrace(iwave(i),ifib)
            xfpos=float(iwave(i))
            yfpos=trace(i)
            ixmin=max(1,nint(xfpos-radflagx))
            ixmax=min(n,nint(xfpos+radflagx))
            iymin=max(1,nint(yfpos-radflagy))
            iymax=min(n,nint(yfpos+radflagy))
            flag(i)=xflag(iwave(i),nint(trace(i)))
            do ix=ixmin,ixmax
               do iy=iymin,iymax
                  if(xflag(ix,iy).le.0) flag(i)=-1
               enddo
            enddo
         else
            xftf=1.
            xsp(i)=xsp(i)
            xerr(i)=100.
            f2f(i)=xftf
            sky(i)=0.
            flag(i)=-1
         endif
      enddo

      open(unit=11,file='out.sp',status='unknown')
      do i=1,nw
         call xlinint(wave(i),n,waved,xsp,yp)
         call xlinint(wave(i),n,waved,xerr,yerr)
         call xlinint(wave(i),n,waved,xerr2,yerr2)
         call xlinint(wave(i),n,waved,f2f,yf2f)
         call xlinint(wave(i),n,waved,sky,ysky)
         call xlinint(wave(i),n,waved,trace,ytrace)
         call xlinint(wave(i),n,waved,flag,yflag)
         call xlinint(wave(i),ntp,wtp,tp,ytp)
         call xlinint(wave(i),na,wa,fa,yfp)
         yf2f=max(yf2f,0.)
         yfrac=yfp*ytp
         if(yfp.gt.0.and.yflag.gt.0.and.yp.gt.-1e5.and.
     $        yp.lt.1e5.and.yp.gt.-1e3) then
c - yerr2 is propagation of error
c   ysky is the sky value
c   xrs is 5% of the sky value
c   yerrn is the max of yerr2 or (yerr2+0.05*sky)/1.3
c   or yerrn is the quad sum
c   yerr is the empirical local estimate
            xph=yerr2
            xct=ysky
c            xrs=0.05*xct
c            yerrn=max(xph,(xph+xrs)/1.3)
            xrs=0.04*xct
            yerrn=sqrt(xph*xph+xrs*xrs)
c            yerr2=yerrn
            yerrm=yerr
            if(xph.gt.1) then
               yerr=yerrn
               yerrm=yerrn
            endif
            yerrm=yerrm*convfac/wave(i)/ytp/yfp
            write(11,1101) wave(i),yp/yfp,yp*convfac/wave(i)/ytp/yfp,
     $           yfp,ytp,yf2f,yerr2/yfp,yerr/yfp,yerrm
         else
            write(11,1101) wave(i),0.,0.,0.,ytp,yf2f,0.,0.,0.
         endif
      enddo
      close(11)

 1101 format(1x,f8.2,1x,f9.2,1x,1pe11.2,3(1x,0pf7.3),2(1x,f8.2),
     $     1x,1pe11.2)
 706  continue
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
c      if(xp.lt.x(1)) yp=0.
c      if(xp.gt.x(n)) yp=0.
      return
      end

      SUBROUTINE moment(data,n,ave,adev,sdev,var,skew,curt)
      INTEGER n
      REAL adev,ave,curt,sdev,skew,var,data(n)
      INTEGER j
      REAL p,s,ep
c      if(n.le.1)pause 'n must be at least 2 in moment'
      s=0.
      do 11 j=1,n
        s=s+data(j)
11    continue
      ave=s/n
      adev=0.
      var=0.
      skew=0.
      curt=0.
      ep=0.
      do 12 j=1,n
        s=data(j)-ave
        ep=ep+s
        adev=adev+abs(s)
        p=s*s
        var=var+p
        p=p*s
        skew=skew+p
        p=p*s
        curt=curt+p
12    continue
      adev=adev/n
      var=(var-ep**2/n)/(n-1)
      sdev=sqrt(var)
      if(var.ne.0.)then
        skew=skew/(n*sdev**3)
        curt=curt/(n*var**2)-3.
      else
c        pause 'no skew or kurtosis when zero variance in moment'
      endif
      return
      END
