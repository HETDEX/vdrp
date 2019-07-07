
      parameter (narrm1=3100,narrm2=1100,nmax=1100)
      real xspec(narrm1,narrm2),xwave(narrm1,narrm2)
      real xsub(narrm1,narrm2),xwaven(narrm1,narrm2)
      real xd(narrm1,narrm2),xd2(narrm1,narrm2),xd3(narrm1,narrm2)
      real xsub2(narrm1,narrm2),xresid(narrm1,narrm2)
      real xf2f(narrm1,narrm2)
      real xin(narrm1),xfibi(narrm2),xfibb(narrm2)
      real wt(narrm1*narrm2),st(narrm1*narrm2),woffa(nmax)
      real xa(narrm1*narrm2),xn(narrm1*narrm2),pfib(nmax)
      real ya(narrm1*narrm2),yn(narrm1*narrm2)
      real xsm(narrm1*narrm2),ysm(narrm1*narrm2)
      real y3(narrm1*narrm2),wlo(nmax),whi(nmax)
      real xsm2(nmax),ysm2(nmax)
      real*8 val
      integer naxes(2),iflag(narrm2),ibadfib(112)
      character file1*180,file1o*180
      common/smoothv/ val
      logical simple,extend,anyf

      xwlo=4500.
      xwhi=5300.
      xcut0=3.
      nrcut=10
      ibin=11
      val=0.05d0

      print *,"Image"
      read *,file1
      file1o=file1

c - this is the spectrum
      iext=11
      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         print *,"Nothing here ",file1
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xspec,anyf,ier)

c - this is the wavelength
      iext=12
      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xwave,anyf,ier)

c - this is the f2f
      iext=14
      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xf2f,anyf,ier)
      call ftclos(im1,ier)

c- find the average value for each fiber
      do j=1,nrow
         nin=0
         do i=1,ncol
            wave=xwave(i,j)
            if(wave.gt.xwlo.and.wave.lt.xwhi) then
               if(xf2f(i,j).gt.0) then
                  nin=nin+1
                  xin(nin)=xspec(i,j)/xf2f(i,j)
               endif
            endif
         enddo
         call biwgt(xin,nin,xb,xs)
         xfibi(j)=float(j)
         xfibb(j)=xb
      enddo
      nfib=nrow
      nwave=ncol

c- not cut the highest out and then n-sigma
      call sort2(nrow,xfibb,xfibi)
      do i=1,nrow
         xin(i)=xfibb(i)
      enddo
      call biwgt(xin,nrow-nrcut,xb,xs)
      xcut=xb+xcut0*xs
      do i=1,nrow
         itmp=nint(xfibi(i))
         iflag(itmp)=0
         if(xfibb(i).gt.xcut) iflag(itmp)=1
         if(i.ge.nrow-nrcut) iflag(itmp)=1
      enddo

c- make the combined spectra
      nin=0
      do j=1,nrow
         if(iflag(j).eq.0) then
            do i=1,ncol
               if(xf2f(i,j).gt.0.and.xspec(i,j).ne.0) then
                  nin=nin+1
                  wt(nin)=xwave(i,j)
                  st(nin)=xspec(i,j)/xf2f(i,j)
               endif
            enddo
         endif
      enddo
      call sort2(nin,wt,st)
      n=nin

      ymin=1e10
      ymax=-1e10
c- first bin, then smooth
      ib1=(ibin-1)/2
      xib=float(ibin)
      nbb=0
      do j=1,n,ibin
         nbb=nbb+1
         istart=max(0,j-ib1)
         iend=istart+ibin-1
         if(iend.gt.n) then
            iend=n
            istart=n-ibin+1
         endif
         sum=0.
         nb=0
         do is=istart,iend
            sum=sum+st(is)
            nb=nb+1
            ya(nb)=st(is)
            xa(nb)=wt(is)
         enddo
         call biwgt(ya,nb,xbb,xsb)
         yn(nbb)=xbb
         call biwgt(xa,nb,xbb,xsb)
         xn(nbb)=xbb
         ymin=min(ymin,yn(nbb))
         ymax=max(ymax,yn(nbb))
      enddo
      ymin=max(0.,ymin)

c- now smooth and plot

      xmin=xn(1)
      xmax=xn(nbb)
      n2=1000
      n2=nbb*2
      do i=1,n2
         xsm(i)=xmin+(xmax-xmin)*float(i-1)/float(n2-1)
      enddo
      call smooth(nbb,xn,yn,n2,xsm,ysm,y3)

      call pgbegin(0,'?',1,2)
      call pgpap(0.,1.)
      call pgsch(1.4)
      call pgscf(2)
      call pgslw(2)

      xmin=3500.
      xmax=5500.
      call pgsci(1)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Wavelength','Counts','')
      call pgline(nbb,xn,yn)
      call pgsci(2)
      call pgline(n2,xsm,ysm)

c- now get new wavelength offsets for each fiber

      open(unit=1,file=
     $     '/work/00115/gebhardt/maverick/scripts/getwave/wvlist',
     $     status='old')
      nwl=0
      do i=1,nmax
         read(1,*,end=668) x1,x2
         nwl=nwl+1
         wlo(nwl)=x1
         whi(nwl)=x2
      enddo
 668  continue
      close(1)

      na=20
      as=-0.1
      ae=0.1
      nfibg=0
      do ifib=1,nfib
         if(iflag(ifib).eq.0) then
            nfibg=nfibg+1
            rmin=1e10
            do ia=1,na
               woff=as+(ae-as)/float(na-1)*float(ia-1)
               rms=0.
               rms2=0.
               nrms2=0
               nwc=0
               do i=1,nwave
                  if(xf2f(i,ifib).gt.0.and.xspec(i,ifib).ne.0) then
                     nwc=nwc+1
                     wval=xwave(i,ifib)
                     sval=xspec(i,ifib)/xf2f(i,ifib)
                     wnew=wval+woff
                     call xlinint(wnew,n2,xsm,ysm,snew)
                     rms=rms+(snew-sval)**2
                  endif
               enddo
               rms=sqrt(rms/float(nwave))
               if(rms.lt.rmin) then
                  rmin=rms
                  woffa(nfibg)=woff
               endif
            enddo
            pfib(nfibg)=float(ifib)
         endif
      enddo
      call pgsci(1)
      call pgenv(1.,112.,as,ae,0,0)
      call pglabel('Fiber','Wave_offset','')
      call pgline(nfibg,pfib,woffa)
      do i=1,nfibg
         xin(i)=woffa(i)
      enddo
      call biwgt(xin,nfibg,xfbb,xfsb)
      nn=0
      do i=1,nfibg
         x1=xfbb-3.*xfsb
         x2=xfbb+3.*xfsb
         if(woffa(i).ge.x1.and.woffa(i).le.x2) then
            nn=nn+1
            pfib(nn)=pfib(i)
            woffa(nn)=woffa(i)
         endif
      enddo
c- now smooth and plot

      n=112
      do i=1,n
         xsm2(i)=float(i)
      enddo
      val=0.d0
      call smooth(nn,pfib,woffa,n,xsm2,ysm2,y3)

      call pgsci(4)
      call pgline(nn,pfib,woffa)
      call pgsci(2)
      call pgline(n,xsm2,ysm2)

      call pgend

c- write new wavelength array, and subtract new sky
      do j=1,nfib
         do i=1,nwave
            xwaven(i,j)=xwave(i,j)+ysm2(j)
            if(xf2f(i,j).gt.0.and.xspec(i,j).ne.0) then
               wval=xwaven(i,j)
               call xlinint(wval,n2,xsm,ysm,yval)
               sval=xspec(i,j)-yval*xf2f(i,j)
               sval2=yval*xf2f(i,j)
            else
               sval=0.
               sval2=0.
            endif
            xsub(i,j)=sval
            xsub2(i,j)=sval2
         enddo
      enddo

c- get a boxcar of sky subtracted image for background
      nbi=201
      nbj=41
c      nbi=101
c      nbj=21
      call boxcar(ncol,nrow,xsub,xd,nbi,nbj,iflag,1.0)
      do j=1,nfib
         do i=1,nwave
            xsub(i,j)=xsub(i,j)-xd(i,j)
         enddo
      enddo

c- get sqrt(sky) error: get sky_sub, boxcor to get smooth continuum sources, add into 2dsky, take sqrt
      nbi=15
      nbj=1
      readn=2.9*sqrt(5.)
      readn=0.
      call boxcar(ncol,nrow,xsub,xd2,nbi,nbj,iflag,0.0)
      do j=1,nfib
         do i=1,nwave
            xval=xsub2(i,j)+xd2(i,j)
            xdum=xval+readn*readn
            xdum=max(0.,xdum)
            xd3(i,j)=sqrt(xdum)
         enddo
      enddo

c- apply residuals
      call getresidual(file1,xresid,iout)
      if(iout.eq.0) then
         do j=1,nfib
            do i=1,nwave
               xres=xsub2(i,j)*xresid(i,j)
               xsub(i,j)=xsub(i,j)-xres
            enddo
         enddo
      endif

c- get list of bad fibers
      call getbadfib(file1,nbadf,ibadfib)
      do j=1,nbadf
         ifib=ibadfib(j)
         do i=1,nwave
            xsub(i,ifib)=0.
            xd3(i,ifib)=0.
         enddo
      enddo

c- write it out to the multifits
      im1=0
      ier=0
      iread=0
      call ftclos(51,ier)
      call ftclos(50,ier)
      call ftopen(51,file1o,iread,iblock,ier)
      call ftinit(50,'out.fits',iblock,ier)
      do iext=1,17
         ier=0
         call ftmahd(51,iext,ihd,ier)
         call ftghpr(51,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
         if(naxis.eq.1) naxes(2)=1
         ncol=naxes(1)
         nrow=naxes(2)
         call ftg2de(51,igc,0.,narrm1,ncol,nrow,xspec,anyf,ier)
         call ftcopy(51,50,0,ier)
         if(iext.eq.12) then
            call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xwaven,ier)
         elseif(iext.eq.16) then
            call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xsub,ier)
         else
            call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xspec,ier)
         endif
      enddo
      call ftiimg(50,-32,2,naxes,ier)
      call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xd3,ier)
      call ftpkls(50,'EXTNAME','error1Dfib',"Label",ier)
      call ftclos(51,ier)
      call ftclos(50,ier)

c- skip for now
      goto 706
c- write out the images
      naxis=2
      naxes(1)=nwave
      naxes(2)=nfib
      iblock=1
      igc=0
      ier=0
c- the wavelength
      call ftgiou(im1,ier)
      call ftinit(im1,'image1.fits',iblock,ier)
      call ftphps(im1,-32,naxis,naxes,ier)
      call ftp2de(im1,igc,narrm1,naxes(1),naxes(2),xwaven,ier)
      call ftclos(im1,ier)

c- the sky subtracted
      call ftgiou(im1,ier)
      call ftinit(im1,'image2.fits',iblock,ier)
      call ftphps(im1,-32,naxis,naxes,ier)
      call ftp2de(im1,igc,narrm1,naxes(1),naxes(2),xsub,ier)
      call ftclos(im1,ier)

c- the background
      call ftgiou(im1,ier)
      call ftinit(im1,'image3.fits',iblock,ier)
      call ftphps(im1,-32,naxis,naxes,ier)
      call ftp2de(im1,igc,narrm1,naxes(1),naxes(2),xd,ier)
      call ftclos(im1,ier)

c- the error frame
      call ftgiou(im1,ier)
      call ftinit(im1,'image4.fits',iblock,ier)
      call ftphps(im1,-32,naxis,naxes,ier)
      call ftp2de(im1,igc,narrm1,naxes(1),naxes(2),xd3,ier)
      call ftclos(im1,ier)

 706  continue
      end

      subroutine smooth(n,x,y,n2,x2,y2,y3)
      parameter(nmax=20000,mm=2,nwk=nmax+6*(nmax*mm+1),mm2=mm*2)
      real x(n),y(n),x2(n2),y2(n2),y3(n)
      real*8 dx(nmax),dy(nmax),wx(nmax),cf(nmax),wk(nwk),val,splder
      real*8 q(mm2)
      common/smoothv/ val

      if(n.gt.nmax) print *,'make nmax bigger in smooth'

      if(val.ge.0) then
         md=3
         if(val.eq.0.) md=2
         m=2
         
         do i=1,n
            dx(i)=dble(x(i))
            dy(i)=dble(y(i))
            wx(i)=1.d0
         enddo
         
         ier=0
         call gcvspl(dx,dy,nmax,wx,1.d0,m,n,1,md,val,cf,nmax,wk,ier)
         if(ier.ne.0) print *,'ier= ',ier
         
         do i=1,n2
            in=i
            y2(i)=sngl(splder(0,m,n,dble(x2(i)),dx,cf,in,q))
         enddo
         
         do i=1,n
            in=i
            y3(i)=sngl(splder(0,m,n,dble(x(i)),dx,cf,in,q))
         enddo
      else
         do i=1,n2
            call xlinint(x2(i),n,x,y,y2(i))
         enddo
         do i=1,n
            y3(i)=y(i)
         enddo
      endif
         
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

      subroutine boxcar(ncol,nrow,xd,xd2,nbi,nbj,iflaga,rad)
      parameter(narrm1=3100,narrm2=1100,nmax=1100)
      real xd(narrm1,narrm2),xd2(narrm1,narrm2),xin(100000)
      integer iflaga(1000)

      iflag=0
      ibi=1
      ibox=nbi
      jbox=nbj
      ihalf=(ibox-1)/2
      jhalf=(jbox-1)/2
      do j=1,nrow
         jlo=max(1,j-jhalf)
         jup=min(nrow,j+jhalf)
         do i=1,ncol
            ilo=max(1,i-ihalf)
            iup=min(ncol,i+ihalf)
            nin=0
            do j1=jlo,jup
               if(iflaga(j1).eq.0) then
                  do i1=ilo,iup
                     radp=sqrt(float(i1-i)**2+float(j1-j)**2)
                     if(radp.gt.rad.and.xd(i1,j1).ne.0) then
                        nin=nin+1
                        xin(nin)=xd(i1,j1)
                     endif
                  enddo
               endif
            enddo
            if(nin.gt.0) then
               if(ibi.eq.0) then
                  if(nin.gt.1) then
                     call sort(nin,xin)
                     nuse=nint(float(nin)*0.7)
                     xb=xin(nuse)
                  else
                     xb=xin(1)
                  endif
               else
                  call biwgt(xin,nin,xb,xs)
               endif
            else
               xb=float(iflag)
            endif
            xd2(i,j)=xb
         enddo
      enddo

      return
      end

      subroutine getresidual(file1,xresid,iout)
      parameter (narrm1=3100,narrm2=1100,nmax=1100)
      real xresid(narrm1,narrm2)
      integer naxes(2)
      character file1*180,file2*180
      logical simple,extend,anyf

      nc=0
      do i=1,180
         nc=nc+1
         if(file1(i:i).eq.".") goto 666
      enddo
 666  continue
      nc=nc-1
      file2="/work/03946/hetdex/maverick/virus_config/rescor/"
     $     //file1(1:nc)//"res.fits"

      iext=1
      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file2,iread,iblock,ier)
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xresid,anyf,ier)
      call ftclos(im1,ier)
      if(ier.eq.0) iout=0
      if(ier.ne.0) iout=1

      return
      end

      subroutine getbadfib(file1,nbadf,ibadfib)
      integer ibadfib(112)
      character file1*120,c1*50

      open(unit=1,file=
     $     '/work/00115/gebhardt/maverick/scripts/getwave/badfib.list',
     $     status='old')
      nbadf=0
      do i=1,10000
         read(1,*,end=666) c1,i2
         if(c1(1:20).eq.file1(1:20)) then
            nbadf=nbadf+1
            ibadfib(nbadf)=i2
         endif
      enddo
 666  continue
      close(1)

      return
      end
