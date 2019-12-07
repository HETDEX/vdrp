
      parameter (narrm=2000)
      real xd(narrm,narrm),arr(narrm,narrm),xin(narrm*narrm)
      real xov(narrm),xbias(narrm,narrm),xback(10),xiback(10)
      real xouta(3),xsum(narrm,narrm),arr2(narrm,narrm)
      real xtrace(narrm,narrm),xprof(1032,112,15),x(15),y(15)
      real xspec(narrm,112),xpeaks(10),xspecr(1032,112),xspecc(1032,112)
      real wtrace(narrm,112),xftf(narrm,112),xmaster(narrm*112)
      real wmaster(narrm*112),arrs(1032,1032),xspecs(narrm,112)
      real wfito(narrm),ofito(narrm),wavesol(1036),wavesolo(1036)
      real wtrace0(narrm,112),xflat(narrm,narrm),xspecsm(1032,112)
      real xifupos(2,112),xflat2(1032,1032),arrm(1032,1032)
      real arrerr(1032,1032),specerr(1032,112)
      integer naxes(2),ifib(112),ipeaks(10)
      character file1*40,cspec*100,cback(10)*60,cspecid*3
      character cpos*1,chalf*1,cspec0*3,cifu*3,cifupos*3
      character amp*2,campname*2,cmth*6,string*80,fileo*80
      logical simple,extend,anyf

      ioutall=1
      iplot=0

      if(iplot.eq.1) then
         call pgbegin(0,'?',2,2)
c         call pgask(.false.)
         call pgpap(0.,1.)
         call pgscf(2)
         call pgsch(1.5)
         call pgslw(2)
      endif

c- set iuse to fit calibration from frame (0), or read in calibration (1)
      open(unit=1,file='vred.in',status='old')
      read(1,*) imth,iuse
      close(1)
      write(cmth,2001) imth
 2001 format(i6)

c- set the output wavelength calibration
      nwo=1036
      do i=1,nwo
         wfito(i)=3470.+2*float(i-1)
      enddo

c- set the normalization value and the 3 background arrays
      xnorm=25.
      xback(1)=1.0
      xback(2)=5.0
      xback(3)=25.0
      xiback(1)=1.
      xiback(2)=2.
      xiback(3)=3.
      cback(1)=
     $"/work/03946/hetdex/maverick/virus_config/lib_back/"//cmth//"sci/"
      cback(2)=
     $"/work/03946/hetdex/maverick/virus_config/lib_back/"//cmth//"bri/"
      cback(3)=
     $"/work/03946/hetdex/maverick/virus_config/lib_back/"//cmth//"twi/"

c- get bias
      do i=1,narrm
         do j=1,narrm
            xbias(i,j)=0.
         enddo
      enddo
      open(unit=1,file='spec.use',status='old',err=444)
      read(1,'(a100)') cspec
      close(1)
      call getifits(cspec,xbias)
 444  continue
      close(1)

c- get data frame
      file1='in.fits'
      iext=1

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftgkye(im1,'IFUSLOT',pifu,file1,ier)
      ifupos=nint(pifu)
      cifupos="000"
      if(ifupos.lt.10) write(cifupos(3:3),1001) ifupos
      if(ifupos.ge.10.and.ifupos.lt.100) write(cifupos(2:3),1002) ifupos
      if(ifupos.ge.100) write(cifupos(1:3),1003) ifupos
 1001 format(i1)
 1002 format(i2)
 1003 format(i3)
      call ftgkys(im1,'CCDPOS',cpos,file1,ier)
      call ftgkys(im1,'CCDHALF',chalf,file1,ier)
      call ftgkys(im1,'IFUID',cifu,file1,ier)
      call ftgkys(im1,'AMPNAME',campname,file1,ier)
      if(ier.ne.0) then
         ier=0
         call ftgkys(im1,'AMPLIFIE',campname,file1,ier)
      endif
      call ftgkye(im1,'SPECID',specid,file1,ier)
      ispec=nint(specid)
      cspecid="000"
      if(ispec.lt.10) write(cspecid(3:3),1001) ispec
      if(ispec.ge.10.and.ispec.lt.100) write(cspecid(2:3),1002) ispec
      if(ispec.ge.100) write(cspecid(1:3),1003) ispec
      call ftclos(im1,ier)

c- get pixel flat
      do i=1,1032
         do j=1,1032
            xflat(i,j)=1.0
         enddo
      enddo
      cspec="/work/03946/hetdex/maverick/virus_config/lib_pflat/"//
     $     "pixelflat_cam"//cspecid//"_"//cpos//chalf//".fits"
      print *,"Reading: ",cspec
      call getifits(cspec,xflat)
      close(1)

c- set overscan, trim, and range for normalization 
      ix1=1034
      ix2=1060
      iy1=10
      iy2=1032

      ixt1=1
      ixt2=1032
      iyt1=1
      iyt2=1032

      ixn1=300
      ixn2=700
      iyn1=300
      iyn2=700

c- remove overscan and bias, and divide by flat
      n=0
      nov=0
      do j=iy1,iy2
         nov=nov+1
         do i=ix1,ix2
            n=n+1
            xin(n)=xd(i,j)
         enddo
      enddo
      call biwgt(xin,n,xb,xs)
      xov(nov)=xb
      xbo=xb

      ny=0
      do j=iyt1,iyt2
         ny=ny+1
         nx=0
         do i=ixt1,ixt2
            nx=nx+1
            arr(nx,ny)=xd(i,j)-xb-xbias(i,j)
            if(xflat(i,j).gt.0) then
               arr(nx,ny)=arr(nx,ny)/xflat(i,j)
            else
               arr(nx,ny)=0.
            endif
         enddo
      enddo      

c- get background number and build background frame
      naxes(1)=ixt2-ixt1+1
      naxes(2)=iyt2-iyt1+1

      n=0
      do j=iyn1,iyn2
         do i=ixn1,ixn2
            n=n+1
            xin(n)=arr(i,j)
         enddo
      enddo      

      call biwgt(xin,n,xb,xs)
      xbn=xb/xnorm

      call xlinint(xbn,3,xback,xiback,xib)
      xout1=0.
      xout2=0.
      xout3=0.
      if(xib.ge.1.and.xib.lt.2) then
         xout1=1.-(xib-1.)
         xout2=1.-xout1
      endif
      if(xib.ge.2.and.xib.le.3) then
         xout2=1.-(xib-2.)
         xout3=1.-xout2
      endif
      open(unit=11,file='out',status='unknown')
      write(11,*) xbn,xout1,xout2,xout3
      close(11)
      xouta(1)=xout1
      xouta(2)=xout2
      xouta(3)=xout3

c- get background data
      do i=1,ncol
         do j=1,nrow
            xsum(i,j)=0.
         enddo
      enddo
      do ia=1,3
         cspec=cback(ia)//"bk"//cmth//"i"//cifupos//cpos//chalf//".fits"
         call getifits(cspec,xbias)
         do i=1,ncol
            do j=1,nrow
               xsum(i,j)=xsum(i,j)+xouta(ia)*xbias(i,j)
            enddo
         enddo
      enddo
c- subtract background
      do i=1,ncol
         do j=1,nrow
            if(arr(i,j).ne.0) arr(i,j)=arr(i,j)-xbn*xsum(i,j)
         enddo
      enddo

c- flip amps for data
      amp=cpos//chalf
      if(amp.eq."LU".or.amp.eq."RL") then
         do i=1,1032
            do j=1,1032
               arr2(i,j)=arr(1033-i,1033-j)
            enddo
         enddo
         do i=1,1032
            do j=1,1032
               arr(i,j)=arr2(i,j)
            enddo
         enddo
      endif
      if(campname.eq."LR".or.campname.eq."UL") then
         do i=1,1032
            do j=1,1032
               arr2(i,j)=arr(1033-i,j)
            enddo
         enddo
         do i=1,1032
            do j=1,1032
               arr(i,j)=arr2(i,j)
            enddo
         enddo
      endif

c- flip amps for flat
      amp=cpos//chalf
      if(amp.eq."LU".or.amp.eq."RL") then
         do i=1,1032
            do j=1,1032
               arr2(i,j)=xflat(1033-i,1033-j)
            enddo
         enddo
         do i=1,1032
            do j=1,1032
               xflat2(i,j)=arr2(i,j)
            enddo
         enddo
      endif
      if(campname.eq."LR".or.campname.eq."UL") then
         do i=1,1032
            do j=1,1032
               arr2(i,j)=xflat(1033-i,j)
            enddo
         enddo
         do i=1,1032
            do j=1,1032
               xflat2(i,j)=arr2(i,j)
            enddo
         enddo
      endif

c- get initial trace
      call getfloc(arr,cspecid,cpos,chalf,cifupos,cifu,xtrace,ifib,
     $     iuse,0)

c- get the binning for the fiber profile
      call getfbin(nfbin,x)

c- read in fiber profile and f2f if iuse=1
      call readcalib(xprof,xftf,xtrace,wtrace,wavesol,wavesolo,
     $     cifupos,cpos,chalf,cmth,cspecid,iuse)

c- set xftf to 1 if nan (fix later)
      do j=1,112
         do i=1,1032
            if(xftf(i,j).gt.0.and.xftf(i,j).lt.2) then
            else
               xftf(i,j)=1.
            endif
         enddo
      enddo

c- fit profile, refit trace, and iterate
      do iter=1,10
         call getfprof(arr,xtrace,nfbin,x,xprof,iuse,0)
         call getfloc2(arr,xtrace,nfbin,x,xprof,ifib,xlast,iuse,0)
c         print *,iter,xlast
         if(iter.ge.2.and.abs(xlast).lt.0.02) goto 555
      enddo
 555  continue

c- extract the spectra
      call extract(arr,xtrace,nfbin,x,xprof,xspec,xspecr,xspecc,
     $     ifib,iuse)

c- fit the wavelength trace
      call getwtrace(xspec,wtrace,ifib,iuse,0)

c- make master spectrum, get fiber-to-fiber
      call getmaster(xspec,wtrace,nmaster,wmaster,xmaster,xftf,ifib,
     $     iuse,0)
      call fitmaster(nmaster,wmaster,xmaster,nwo,wfito,ofito,iuse,0)
      call writemaster(nmaster,wmaster,xmaster,nwo,wfito,ofito,
     $     wavesol,wavesolo,iuse)

c- get four spectral peaks in master
      call getpeaks(xspec,wtrace,nmaster,wmaster,xmaster,
     $     wavesol,wavesolo,xpeaks,ipeaks,iuse,0)
      do iter=1,4
         call getwtrace3(arr,xspec,xtrace,nfbin,x,xprof,xftf,wtrace,
     $        nmaster,wmaster,xmaster,xspecs,xpeaks,ipeaks,
     $        ifib,iuse,0)
      enddo
      call getmaster(xspec,wtrace,nmaster,wmaster,xmaster,xftf,ifib,
     $     iuse,0)

c- subtract the master spectrum
      call subtract(arr,xspec,xtrace,nfbin,x,xprof,xftf,wtrace,
     $     nmaster,wmaster,xmaster,arrs,arrm,xspecs,xspecsm,ifib,0)

c- subtract a smooth background from the extracted array
      call getback(xspecs,ifib,ifback,iuse,0)

c- get the error frame
      call geterror(arrm,arrs,xflat2,xtrace,nfbin,x,
     $     xprof,ifib,arrerr,specerr)

c- set the wavelength solution
      call getwsol(wavesol,wavesolo,wtrace0,wtrace,
     $     nmaster,wmaster,xmaster,iuse,0)

c- get the ifu positions
      call getifupos(cifu,cpos,chalf,xifupos)

c- write out the profile
      call writeprof(xprof,iuse,1)

c- write out the oriented image
      call ftinit(51,'out0.fits',iblock,ier)
      call ftphps(51,-32,naxis,naxes,ier)
      call ftp2de(51,igc,narrm,naxes(1),naxes(2),arr,ier)
      call ftclos(51,ier)

c- write out the fits files: final image, extracted
      call ftinit(51,'out.fits',iblock,ier)
      call ftphps(51,-32,naxis,naxes,ier)
      call ftp2de(51,igc,1032,naxes(1),naxes(2),arrs,ier)
      call ftclos(51,ier)

      if(iuse.eq.0) then
         naxes(1)=1032
         naxes(2)=112
         call ftinit(51,'out2.fits',iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xftf,ier)
         call ftclos(51,ier)

         call ftinit(51,'out3.fits',iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xtrace,ier)
         call ftclos(51,ier)

         call ftinit(51,'out4.fits',iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),wtrace0,ier)
         call ftclos(51,ier)

         call ftinit(51,'out7.fits',iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xspec,ier)
         call ftpkys(51,"SPECID",cspecid,"spectrograph ID",ier)
         call ftpkys(51,"AMP",cpos//chalf,"AMP",ier)
         call ftpkys(51,"IFUID",cifu,"IFU ID",ier)
         call ftpkys(51,"IFUSLOT",cifupos,"IFU slot",ier)
         call ftclos(51,ier)

         call ftinit(51,'out8.fits',iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),wtrace,ier)
         call ftpkys(51,"SPECID",cspecid,"spectrograph ID",ier)
         call ftpkys(51,"AMP",cpos//chalf,"AMP",ier)
         call ftpkys(51,"IFUID",cifu,"IFU ID",ier)
         call ftpkys(51,"IFUSLOT",cifupos,"IFU slot",ier)
         call ftclos(51,ier)
         
      else

         naxes(1)=1032
         naxes(2)=112
         call ftinit(51,'out5.fits',iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xspecs,ier)
         call ftpkys(51,"SPECID",cspecid,"spectrograph ID",ier)
         call ftpkys(51,"AMP",cpos//chalf,"AMP",ier)
         call ftpkys(51,"IFUID",cifu,"IFU ID",ier)
         call ftpkys(51,"IFUSLOT",cifupos,"IFU slot",ier)
         call ftclos(51,ier)
         
         call ftinit(51,'out6.fits',iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,1032,naxes(1),naxes(2),xspecr,ier)
         call ftpkys(51,"SPECID",cspecid,"spectrograph ID",ier)
         call ftpkys(51,"AMP",cpos//chalf,"AMP",ier)
         call ftpkys(51,"IFUID",cifu,"IFU ID",ier)
         call ftpkys(51,"IFUSLOT",cifupos,"IFU slot",ier)
         call ftclos(51,ier)

         call ftinit(51,'out7.fits',iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xspec,ier)
         call ftpkys(51,"SPECID",cspecid,"spectrograph ID",ier)
         call ftpkys(51,"AMP",cpos//chalf,"AMP",ier)
         call ftpkys(51,"IFUID",cifu,"IFU ID",ier)
         call ftpkys(51,"IFUSLOT",cifupos,"IFU slot",ier)
         call ftclos(51,ier)

         call ftinit(51,'out8.fits',iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),wtrace,ier)
         call ftpkys(51,"SPECID",cspecid,"spectrograph ID",ier)
         call ftpkys(51,"AMP",cpos//chalf,"AMP",ier)
         call ftpkys(51,"IFUID",cifu,"IFU ID",ier)
         call ftpkys(51,"IFUSLOT",cifupos,"IFU slot",ier)
         call ftclos(51,ier)
         
         call ftinit(51,'out9.fits',iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,1032,naxes(1),naxes(2),xspecc,ier)
         call ftpkys(51,"SPECID",cspecid,"spectrograph ID",ier)
         call ftpkys(51,"AMP",cpos//chalf,"AMP",ier)
         call ftpkys(51,"IFUID",cifu,"IFU ID",ier)
         call ftpkys(51,"IFUSLOT",cifupos,"IFU slot",ier)
         call ftclos(51,ier)
      endif

      if(ioutall.eq.1) then
         naxes(1)=1032
         naxes(2)=1032
         fileo="multi_"//cspecid//"_"//cifupos//"_"//cifu//
     $        "_"//cpos//chalf//".fits"
         print *,fileo
         call ftopen(50,'in.fits',iread,iblock,ier)
c         call ftinit(51,'outall.fits',iblock,ier)
         call ftinit(51,fileo,iblock,ier)
         call ftphps(51,-32,naxis,naxes,ier)
c         call ftcopy(50,51,0,ier)
c         call ftcpdt(50,51,ier)

         do i=8,1000
            call ftgrec(50,i,string,ier)
            if(ier.ne.0.or.string(1:3).eq.'END') then
               ier=0
               goto 807
            endif
            call ftprec(51,string,ier)
         enddo
         ier=0
 807     continue                                                                                                  

         call ftclos(50,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),arr,ier)
         call ftpkls(51,'EXTNAME','processed',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,1032,naxes(1),naxes(2),arrerr,ier)
         call ftpkls(51,'EXTNAME','error',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,1032,naxes(1),naxes(2),arrs,ier)
         call ftpkls(51,'EXTNAME','clean_image',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,1032,naxes(1),naxes(2),arrm,ier)
         call ftpkls(51,'EXTNAME','model2d',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,1032,naxes(1),naxes(2),xflat2,ier)
         call ftpkls(51,'EXTNAME','flat',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xflat,ier)
         call ftpkls(51,'EXTNAME','flat_orig',"Label",ier)

         naxes(1)=ncol
         naxes(2)=nrow
         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xd,ier)
         call ftpkls(51,'EXTNAME','raw',"Label",ier)

         naxes(1)=2
         naxes(2)=112
         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,2,naxes(1),naxes(2),xifupos,ier)
         call ftpkls(51,'EXTNAME','ifupos',"Label",ier)

         naxes(1)=1032
         naxes(2)=112
         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,1032,naxes(1),naxes(2),xspecc,ier)
         call ftpkls(51,'EXTNAME','chi2',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,1032,naxes(1),naxes(2),xspecr,ier)
         call ftpkls(51,'EXTNAME','rms',"Label",ier)

         naxes(1)=1
         naxes(2)=1
         call ftiimg(51,-32,naxis,naxes,ier)
c         call ftp2de(51,igc,1032,naxes(1),naxes(2),xspecr,ier)
c         call ftpkls(51,'EXTNAME','error_analysis',"Label",ier)

         naxes(1)=1032
         naxes(2)=112
         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xspec,ier)
         call ftpkls(51,'EXTNAME','spectrum',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),wtrace,ier)
         call ftpkls(51,'EXTNAME','wavelength',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xtrace,ier)
         call ftpkls(51,'EXTNAME','trace',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xftf,ier)
         call ftpkls(51,'EXTNAME','fiber_to_fiber',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,1032,naxes(1),naxes(2),specerr,ier)
         call ftpkls(51,'EXTNAME','error1d',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,narrm,naxes(1),naxes(2),xspecs,ier)
         call ftpkls(51,'EXTNAME','sky_subtracted',"Label",ier)

         call ftiimg(51,-32,naxis,naxes,ier)
         call ftp2de(51,igc,1032,naxes(1),naxes(2),xspecsm,ier)
         call ftpkls(51,'EXTNAME','sky_spectrum',"Label",ier)

         call ftclos(51,ier)
      endif

 706  continue
      end

      subroutine getfbin(n,x)
      real x(15)

      n=15
      xbmin=-4.5
      xbmax=4.5
      do i=1,n
         x(i)=xbmin+float(i-1)*(xbmax-xbmin)/float(n-1)
      enddo
      
      x(1)=-4.5
      x(2)=-4.0
      x(3)=-3.5
      x(4)=-2.9
      x(5)=-2.3
      x(6)=-1.6
      x(7)=-0.7
      x(8)=0.
      x(9)=0.7
      x(10)=1.6
      x(11)=2.3
      x(12)=2.9
      x(13)=3.5
      x(14)=4.0
      x(15)=4.5

      return
      end

      subroutine getifits(cspec,xbias)
      parameter(narrm=2000)
      real xbias(narrm,narrm)
      integer naxes(2)
      character cspec*100
      logical simple,extend,anyf

      im1=0
      ier=0
      iext=1
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,cspec,iread,iblock,ier)
      if(ier.ne.0) then
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         print *,naxes(1),naxes(2)
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xbias,anyf,ier)
      call ftclos(im1,ier)
 706  continue
      if(ier.ne.0) print *,"No image for: ",cspec

      return
      end

      subroutine readcalib(xprof,xftf,xtrace,wtrace,
     $     wavesol,wavesolo,cifupos,cpos,chalf,cmth,cspecid,iuse)

      parameter(narrm=2000)
      real xprof(1032,112,15),xftf(narrm,112),xtrace(narrm,112)
      real wtrace(narrm,112),xd(1032,112*15)
      real wavesol(1036),wavesolo(1036)
      integer naxes(2)
      character file1*100,file2*100
      character cpos*1,chalf*1,cifupos*3,cmth*6,cspecid*3
      logical simple,extend,anyf

      if(iuse.eq.0) goto 444

c- get fiber profile
      file1="/work/03946/hetdex/maverick/virus_config/lib_calib/"//cmth
     $     //"/i"//cifupos//"a"//cpos//chalf//"cmbp.fits"
      print *,"Reading: ",file1

      im1=0
      ier=0
      iext=1
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,1032,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      do i=1,1032
         do j=1,112
            do k=1,15
               jp=(j-1)*15+k
               xprof(i,j,k)=xd(i,jp)
            enddo
         enddo
      enddo

 706  continue
      if(ier.ne.0) print *,"No fiber profile for: ",file1

c- get fiber-to-fiber
      file1="/work/03946/hetdex/maverick/virus_config/lib_calib/"//cmth
     $     //"/i"//cifupos//"a"//cpos//chalf//"cmbf.fits"
      print *,"Reading: ",file1

      im1=0
      ier=0
      iext=1
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 707
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xftf,anyf,ier)
      call ftclos(im1,ier)

 707  continue
      if(ier.ne.0) print *,"No f2f for: ",file1

c- get fiber trace
      file1="/work/03946/hetdex/maverick/virus_config/lib_calib/"//cmth
     $     //"/i"//cifupos//"a"//cpos//chalf//"cbxt.fits"
      print *,"Reading: ",file1

      im1=0
      ier=0
      iext=1
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 708
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xtrace,anyf,ier)
      call ftclos(im1,ier)

 708  continue
      if(ier.ne.0) print *,"No fiber trace for: ",file1

c- get wavelength trace
      file1="/work/03946/hetdex/maverick/virus_config/lib_calib/"//cmth
     $     //"/i"//cifupos//"a"//cpos//chalf//"cbwt.fits"
      print *,"Reading: ",file1

      im1=0
      ier=0
      iext=1
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 709
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,wtrace,anyf,ier)
      call ftclos(im1,ier)

 709  continue
      if(ier.ne.0) print *,"No wavelength trace for: ",file1

 444  continue
c- get wavelength solution; wavesolo is what needs to be subtracted from virus
      file1="/work/03946/hetdex/maverick/virus_config/lib_calib/"//cmth
     $     //"/i"//cifupos//"a"//cpos//chalf//"wave.dat"
      print *,"Reading: ",file1

      open(unit=1,file=file1,status='old',err=710)
      do i=1,1036
         read(1,*) x1,x2
         wavesol(i)=x1
         wavesolo(i)=x2
      enddo
      goto 711
 710  continue
      print *,"No wavelength solution for: ",file1
 711  continue
      close(1)

      return
      end

      
      subroutine writemaster(nmaster,wmaster,xmaster,
     $     nwo,wfito,ofito,wavesol,wavesolo,iuse)
      parameter(narrm=2000)
      real wmaster(narrm*112),xmaster(narrm*112)
      real wfito(narrm),ofito(narrm),wavesol(nwo),wavesolo(nwo)

      if(iuse.eq.0) then
         open(unit=11,file='wave.out',status='new',err=667)
         do i=1,nwo
            write(11,*) wfito(i),ofito(i)
            wavesol(i)=wfito(i)
            wavesolo(i)=ofito(i)
         enddo
 667     continue
         close(11)
      endif      

      return
      end

      subroutine writeprof(x,iuse,iwrite)
      real x(1032,112,15),xout(1032,112*15)
      integer naxes(2)

      if(iuse.eq.1) goto 706
      if(iwrite.eq.0) goto 706

      do i=1,1032
         do j=1,112
            do k=1,15
               jp=(j-1)*15+k
               xout(i,jp)=x(i,j,k)
            enddo
         enddo
      enddo

      naxis=2
      naxes(1)=1032
      naxes(2)=112*15
      ier=0
      iblock=1
      igc=0
      call ftinit(51,'outprof.fits',iblock,ier)
      call ftphps(51,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(51,igc,1032,naxes(1),naxes(2),xout,ier)
      call ftclos(51,ier)

 706  continue

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
      if(xp.ge.x(n)) yp=y(n)
      return
      end

      subroutine xlinint2(xp,n,x,y,yp,jin,jout)
      real x(n),y(n)
      do j=jin,n-1
         if(xp.ge.x(j).and.xp.le.x(j+1)) then
            yp=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
            jout=j
            return
         endif
      enddo
      if(xp.lt.x(1)) yp=y(1)
      if(xp.gt.x(n)) yp=y(n)
      jout=1
      return
      end

      subroutine xlinint0(xp,n,x,y,yp)
      real x(n),y(n)
      do j=1,n-1
         if(xp.ge.x(j).and.xp.lt.x(j+1)) then
            yp=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
            return
         endif
      enddo
      if(xp.lt.x(1)) yp=0.
      if(xp.ge.x(n)) yp=0.
      return
      end

      subroutine getfloc2(arr,xtrace,n,x,xprof,ifib,xlast,iuse,iplot)
      parameter (narrm=2000)
      real arr(narrm,narrm),xtrace(narrm,112),x(n),xo(narrm)
      real xprof(1032,112,15),xin(narrm),yin(narrm),yin2(narrm)
      real xtrace2(1032,112),xin3(narrm*112),yin3(narrm*112)
      real xin4(narrm),yin4(narrm),xin2(narrm),yin2s(narrm)
      integer ifib(112)

      xlast=0
      if(iuse.eq.0) then

      jhalf=4

      noff=35
      xstep=0.02
      xstart=-xstep*float(noff-1)/2.

      do jall=1,112
         ymin=1e10
         ymax=-ymin
         npin=0
         jtry=jall
         do iall=1,1032
            itry=iall
 
            do i=1,n
               yin2(i)=xprof(itry,jtry,i)
            enddo

            rmsmin=1e10
            do ioff=1,noff
               xoff=xstart+float(ioff-1)*xstep
               xtnew=xtrace(itry,jtry)+xoff
               rms=0.

               j0=nint(xtnew)
               jlo=j0-jhalf
               jhi=j0+jhalf
               nin=0
               sum=0.
               do j=jlo,jhi
                  nin=nin+1
                  xin(nin)=float(j)-xtnew
                  yin(nin)=arr(itry,j)
                  sum=sum+yin(nin)
               enddo
               sum2=0.
               jin=1
               do i=1,nin
                  yin(i)=yin(i)/sum
c                  call xlinint(xin(i),n,x,yin2,yv)
                  call xlinint2(xin(i),n,x,yin2,yv,jin,jout)
                  yin3(i)=yv
                  sum2=sum2+yin3(i)
                  jin=jout
               enddo
            
               do i=1,nin
                  yin3(i)=yin3(i)/sum2
                  rms=rms+(yin3(i)-yin(i))**2
               enddo
               if(rms.lt.rmsmin) then
                  rmsmin=rms
                  xoffmin=xoff
               endif
            enddo
            xtrace2(itry,jtry)=xtrace(itry,jtry)+xoffmin
            if(ifib(jtry).eq.1) xtrace2(itry,jtry)=xtrace(itry,jtry)

            npin=npin+1
            xin4(npin)=float(itry)
            yin4(npin)=xtrace2(itry,jtry)
            ymin=min(ymin,yin4(npin))
            ymax=max(ymax,yin4(npin))
         enddo
         ism=9
         call smooth2(npin,xin4,yin4,nin3,xin3,yin3,ism)
         nin2=1032
         do icol=1,nin2
            xin2(icol)=float(icol)
         enddo
         call smooth(nin3,xin3,yin3,nin2,xin2,yin2,yin2s,0.0)
         do icol=1,1032
            xtrace2(icol,jtry)=yin2(icol)
         enddo
         if(iplot.eq.1) then
            call pgenv(1.,1032.,ymin,ymax,0,0)
            call pgline(npin,xin4,yin4)
            call pgsci(2)
            call pgline(nin3,xin3,yin3)
            call pgsci(4)
            call pgline(nin2,xin2,yin2)
            call pgsci(1)
         endif
      enddo

      elseif(iuse.eq.1) then

      jhalf=4

      noff=35
      xstep=0.01
      xstart=-xstep*float(noff-1)/2.

      do jall=1,112
         ymin=1e10
         ymax=-ymin
         npin=0
         jtry=jall
         do iall=1,1032
            itry=iall
 
            do i=1,n
               yin2(i)=xprof(itry,jtry,i)
            enddo

            rmsmin=1e10
            do ioff=1,noff
               xoff=xstart+float(ioff-1)*xstep
               xtnew=xtrace(itry,jtry)+xoff
               rms=0.

               j0=nint(xtnew)
               jlo=j0-jhalf
               jhi=j0+jhalf
               nin=0
               sum=0.
               do j=jlo,jhi
                  nin=nin+1
                  xin(nin)=float(j)-xtnew
                  yin(nin)=arr(itry,j)
                  sum=sum+yin(nin)
               enddo
               sum2=0.
               jin=1
               do i=1,nin
                  yin(i)=yin(i)/sum
c                  call xlinint(xin(i),n,x,yin2,yv)
                  call xlinint2(xin(i),n,x,yin2,yv,jin,jout)
                  yin3(i)=yv
                  sum2=sum2+yin3(i)
                  jin=jout
               enddo
            
               do i=1,nin
                  yin3(i)=yin3(i)/sum2
                  rms=rms+(yin3(i)-yin(i))**2
               enddo
               if(rms.lt.rmsmin) then
                  rmsmin=rms
                  xoffmin=xoff
               endif
            enddo
            xtrace2(itry,jtry)=xtrace(itry,jtry)+xoffmin
            if(ifib(jtry).eq.1) xtrace2(itry,jtry)=xtrace(itry,jtry)

            npin=npin+1
            xin4(npin)=float(itry)
            yin4(npin)=xtrace2(itry,jtry)
            ymin=min(ymin,yin4(npin))
            ymax=max(ymax,yin4(npin))
         enddo
         ism=9
         call smooth2(npin,xin4,yin4,nin3,xin3,yin3,ism)
         nin2=1032
         do icol=1,nin2
            xin2(icol)=float(icol)
         enddo
         call smooth(nin3,xin3,yin3,nin2,xin2,yin2,yin2s,0.0)
         do icol=1,1032
            xtrace2(icol,jtry)=yin2(icol)
            if(ifib(jtry).eq.1) xtrace2(icol,jtry)=xtrace(icol,jtry)
         enddo
         if(iplot.eq.2) then
            call pgenv(1.,1032.,ymin,ymax,0,0)
            call pgline(npin,xin4,yin4)
            call pgsci(2)
            call pgline(nin3,xin3,yin3)
            call pgsci(4)
            call pgline(nin2,xin2,yin2)
            call pgsci(1)
         endif
      enddo

      do j=1,112
         nin=0
         do i=200,900
            nin=nin+1
            xin(nin)=xtrace2(i,j)-xtrace(i,j)
         enddo
         call biwgt(xin,nin,xb,xs)
         xin2(j)=float(j)
         yin(j)=xb
      enddo
      nin3=0
      do j=1,112
         if(ifib(j).eq.0) then
            nin3=nin3+1
            yin3(nin3)=yin(j)
         endif
         do i=1,1032
            xtrace2(i,j)=xtrace2(i,j)+yin(j)/2.
         enddo
      enddo
      call biwgt(yin3,nin3,xb,xs)
      xlast=xb
      if(iplot.eq.1) then
         call pgenv(1.,112.,-0.35,0.35,0,0)
         call pgline(112,xin2,yin)
      endif

      endif

c- copy results over
      do j=1,112
         if(ifib(j).eq.0) then
            do i=1,1032
               xtrace(i,j)=xtrace2(i,j)
            enddo
         endif
      enddo

      return
      end

      subroutine getback(xspecs,ifib,ifback,iuse,iplot)
      parameter(narrm=2000)
      real xspecs(narrm,112),xin(1032),xin2(1032),xin3(1032)
      real xspecn(112)
      integer ifib(112),ifibc(112),ifibc2(112)

      if(iuse.eq.0) return

      ifibh=2

c- find the continuum sources                                                                                                                     
      ilo=300
      ihi=800
      nin2=0
      do j=1,112
         nin=0
         do i=ilo,ihi
            if(xspecs(i,j).ne.0.) then
               nin=nin+1
               xin(nin)=xspecs(i,j)
            endif
         enddo
         call biwgt(xin,nin,xb,xs)
         nin2=nin2+1
         xin2(nin2)=xb
         xin3(nin2)=xb
      enddo
      call biwgt(xin2,nin2,xb,xs)

      xcut=3.0*xs
      xcut2=-6.0*xs
      ncut=0
      do j=1,112
         ifibc(j)=0
         ifibc2(j)=0
c         if(xin3(j).gt.xcut) ifibc(j)=1
         if(abs(xin3(j)).gt.xcut) ifibc(j)=1
         if(xin3(j).lt.xcut2) ifibc2(j)=1
         if(ifibc(j).eq.1) ncut=ncut+1
      enddo

      print *,"Nstars = ",ncut

      do j=1,112
         jlo=max(1,j-ifibh)
         jhi=min(112,j+ifibh)
         nin=0
         do jt=jlo,jhi
            if(ifibc(jt).eq.0) then
               do i=ilo,ihi
                  nin=nin+1
                  xin(nin)=xspecs(i,jt)
               enddo
            endif
         enddo
         if(nin.ge.1) then
            call biwgt(xin,nin,xb,xs)
            xspecn(j)=xb
         else
            xspecn(j)=0.
         endif
      enddo

      do j=1,112
         do i=1,1032
            if(xspecs(i,j).eq.0) then
               xspecs(i,j)=0.
            else
               xspecs(i,j)=xspecs(i,j)-xspecn(j)
            endif
            if(ifibc2(j).eq.1) xspecs(i,j)=0.
         enddo
      enddo

      return
      end

      subroutine geterror(arrm,arrs,xflat2,xtrace,n,x,
     $     xprof,ifib,arrerr,specerr)
      parameter(narrm=2000)
      real arrm(1032,1032),arrs(1032,1032),xflat2(1032,1032)
      real arrerr(1032,1032),specerr(1032,112),xin(narrm)
      real yin(narrm),xtrace(narrm,narrm)
      real xprof(1032,112,15),x(15)
      integer ifib(112)

      readn=3.0
      do j=1,1032
         do i=1,1032
            if(arrs(i,j).ne.0) then
               val=max(0.,xflat2(i,j)*arrm(i,j))
               err=sqrt(readn*readn+val)
               if(xflat2(i,j).gt.0) then
                  arrerr(i,j)=err/xflat2(i,j)
               else
                  arrerr(i,j)=0.
               endif
            else
               arrerr(i,j)=0.
            endif
         enddo
      enddo

      jhalf=4
      nhalf=(n-1)/2
      do j=1,112
         sum4=0.
         do i=1,1032
            j0=nint(xtrace(i,j))
            jlo=max(1,j0-jhalf)
            jhi=min(1032,j0+jhalf)
c- get the fiber profile
            do k=1,n
               xin(k)=xprof(i,j,k)
            enddo
c- rebin the fiber profile
            sum1=0.
            nin=0
            do k=jlo,jhi
               nin=nin+1
               xjp=float(k)-xtrace(i,j)
               call xlinint(xjp,n,x,xin,yjp)
               yin(nin)=yjp
               sum1=sum1+yin(nin)
            enddo
            sum2=0.
            sum5=0.
            nin=0
            nzero=0
            do k=jlo,jhi
               nin=nin+1
               yin(nin)=yin(nin)/sum1
               sum2=sum2+arrerr(i,k)*arrerr(i,k)*yin(nin)
               sum5=sum5+yin(nin)**2
               if(arrs(i,k).eq.0) nzero=nzero+1
            enddo
            sum2=sum2/sum5
            if(sum2.gt.0) then
               specerr(i,j)=sqrt(sum2)
            else
               specerr(i,j)=0.
            endif
            if(ifib(j).eq.1) then
               specerr(i,j)=0.
            endif
c- check if >=3 of the elements are zero, then remove
            if(nzero.ge.3) then
               specerr(i,j)=0.
            endif
         enddo
      enddo

      return
      end

      subroutine subtract(arr,xspec,xtrace,nfbin,x,xprof,xftf,wtrace,
     $     nmaster,wmaster,xmaster,arrs,arrm,xspecs,xspecsm,ifib,iplot)
      parameter(narrm=2000)
      real arr(narrm,narrm),xspec(narrm,112),xtrace(narrm,112)
      real xftf(narrm,112),wtrace(narrm,112),wmaster(narrm*112)
      real xmaster(narrm*112),arrs(1032,1032),xspecs(narrm,112)
      real x(nfbin),xprof(1032,112,15),xin(narrm),yin(narrm)
      real xspecsm(1032,112),arrm(1032,1032)
      integer ifib(112)

      jhalf=4
      n=nfbin
      do j=1,112
         jin=1
         do i=1,1032
            w0=3480.+float(i-1)*2.
            w0=w0+wtrace(i,j)
            call xlinint2(w0,nmaster,wmaster,xmaster,sp0,jin,jout)
            xspecs(i,j)=xspec(i,j)-sp0*xftf(i,j)
            xspecsm(i,j)=sp0*xftf(i,j)
            if(xspec(i,j).eq.0) xspecs(i,j)=0.
            if(ifib(j).eq.1) xspecs(i,j)=0.
            jin=jout
         enddo
      enddo

      aall=0.
      rmsall=0.
      nrall=0
      do j=1,112
         if(ifib(j).eq.0) then
            afib=0.
            rmsfib=0.
            nrfib=0
            jin=1
            do i=1,1032
               w0=3480.+float(i-1)*2.
               w0=w0+wtrace(i,j)
               call xlinint2(w0,nmaster,wmaster,xmaster,xs0,jin,jout)
               jin=jout
               xftf0=xftf(i,j)
               j0=nint(xtrace(i,j))
               jlo=max(1,j0-jhalf)
               jhi=min(1032,j0+jhalf)
               do k=1,n
                  xin(k)=xprof(i,j,k)
               enddo
               sum1=0.
               jin2=1
               do k=jlo,jhi
                  xjp=float(k)-xtrace(i,j)
                  call xlinint2(xjp,n,x,xin,yjp,jin2,jout2)
                  jin2=jout2
                  sum1=sum1+yjp
               enddo
               ares=0.
               rmsres=0.
               nrres=0
               jin2=1
               do k=jlo,jhi
                  xjp=float(k)-xtrace(i,j)
                  call xlinint2(xjp,n,x,xin,yjp,jin2,jout2)
                  jin2=jout2
                  xs=xs0*xftf0*yjp/sum1
                  arrm(i,k)=xs
                  arrs(i,k)=arr(i,k)-xs
                  if(xspec(i,j).eq.0) arrs(i,k)=0.
                  var=arrs(i,k)**2
                  nrres=nrres+1
                  nrfib=nrfib+1
                  nrall=nrall+1
                  rmsres=rmsres+var
                  rmsfib=rmsfib+var
                  rmsall=rmsall+var
                  ares=ares+arr(i,k)
                  afib=afib+arr(i,k)
                  aall=aall+arr(i,k)
               enddo
               rmsres=sqrt(rmsres/float(nrres))
               ares=ares/float(nrres)
            enddo
            rmsfib=sqrt(rmsfib/float(nrfib))
            afib=afib/float(nrfib)
         else
            do i=1,1032
               arrs(i,j)=0.
               arrm(i,j)=0.
            enddo
         endif
      enddo
      rmsall=sqrt(rmsall/float(nrall))
      if(rmsall.gt.-1e10.and.rmsall.lt.1e10) then
         aall=aall/float(nrall)
         rat=rmsall/aall
      else
         rmsall=0.
         aall=0.
         rat=0.
      endif
      print *,"RMS info: ",rmsall,aall,rat

      return
      end

      subroutine getwtrace2(arr,xspec,xtrace,nfbin,x,xprof,xftf,wtrace,
     $     nmaster,wmaster,xmaster,xspecs,xpeaks,ifib,iuse,iplot)
      parameter(narrm=2000)
      real arr(narrm,narrm),xspec(narrm,112),xtrace(narrm,112)
      real xftf(narrm,112),wtrace(narrm,112),wmaster(narrm*112)
      real xmaster(narrm*112),xspecs(narrm,112)
      real x(nfbin),xprof(1032,112,15),xin(narrm),yin(narrm)
      real xpeaks(10),wtrace2(1032,112)
      integer ifib(112)

      if(iuse.eq.0) return

      ifibp=55

      npeaks=3
      do i=1,npeaks
c         print *,xpeaks(i)
      enddo

      wdiff=5.

      nwoff=41
      woffs=-1.1
      woffe=1.1

      nwoff2=25
      woff2s=-4.
      woff2e=0.
      rmstmin=1e10
      do iw2=1,nwoff2
         wtry2=woff2s+float(iw2-1)*(woff2e-woff2s)/float(nwoff2-1)
      do iw=1,nwoff
         wtry=woffs+float(iw-1)*(woffe-woffs)/float(nwoff-1)
         do j=1,112
            wnew=wtry*(float(j-ifibp))/float(112-ifibp)
            do i=1,1032
               wfac=1.+wtry2*float(i-1)/float(1031)
               wtrace2(i,j)=wtrace(i,j)+wnew*wfac
            enddo
         enddo

         rmst=0.
         ntot=0
         do j=1,112
            if(ifib(j).eq.0) then
               do i=1,1032
                  w0=3480.+float(i-1)*2.
                  w0=w0+wtrace2(i,j)
                  do ip=1,npeaks
                     if(abs(w0-xpeaks(ip)).lt.wdiff) then
                        call xlinint(w0,nmaster,wmaster,xmaster,sp0)
                        ntot=ntot+1
                        rmst=rmst+(xspec(i,j)-sp0*xftf(i,j))**2
                     endif
                  enddo
               enddo
            endif
         enddo

         if(rmst.lt.rmstmin) then
            rmstmin=rmst
            wmin=wtry
            wmin2=wtry2
         endif
      enddo
      enddo

      wtry=wmin
      wtry2=wmin2
      print *,"Wavelength offset applied: ",wtry,wtry2,rmstmin
      do j=1,112
         wnew=wtry*(float(j-ifibp))/float(112-ifibp)
         do i=1,1032
            wfac=1.+wtry2*(float(i-1)/float(1031))**2
            wtrace(i,j)=wtrace(i,j)+wnew*wfac
         enddo
      enddo

      return
      end

      subroutine getwtrace3(arr,xspec,xtrace,nfbin,x,xprof,xftf,wtrace,
     $     nmaster,wmaster,xmaster,xspecs,xpeaks,ipeaks,ifib,
     $     iuse,iplot)
      parameter(narrm=2000)
      real arr(narrm,narrm),xspec(narrm,112),xtrace(narrm,112)
      real xftf(narrm,112),wtrace(narrm,112),wmaster(narrm*112)
      real xmaster(narrm*112),xspecs(narrm,112)
      real x(nfbin),xprof(1032,112,15),xin(narrm),yin(narrm)
      real xin2(narrm*112),yin2(narrm*112),yin3(narrm)
      real xpeaks(10),woffa(1032,112),wm2(narrm*112),xm2(narrm*112)
      integer ifib(112),ipeaks(10),ipa(112),iwoffa(1032,112)

      if(iuse.eq.0) return

      call smooth2(nmaster,wmaster,xmaster,nm2,wm2,xm2,5)

      npeaks=6
      do i=1,npeaks
c         print *,"Peaks at: ",xpeaks(i)
      enddo

      do i=1,1032
         do j=1,112
            woffa(i,j)=0.
            iwoffa(i,j)=0
         enddo
      enddo

c      ihalf=3
      ihalf=7

      nwoff=71
c      nwoff=101
c      woffs=-9.1
c      woffe=9.1
      woffs=-1.0
      woffe=1.0

      do ip=1,npeaks
c      do ip=1,1
      ymin=1e10
      ymax=-ymin
      nin=0
      do j=1,112
         if(ifib(j).eq.0) then
            wpmin=1e10
            do i=1,1032
               w0=3480.+float(i-1)*2.
               w0=w0+wtrace(i,j)
               wp=abs(w0-xpeaks(ip))
               if(wp.lt.wpmin) then
                  wpmin=wp
                  ipmin=i
               endif
            enddo
            ipa(j)=ipmin
            iplo=max(1,ipmin-ihalf)
            iphi=min(1032,ipmin+ihalf)
c            if(j.eq.10) print *,ip,j,iplo,iphi

            rmstmin=1e10
            do iw=1,nwoff
               wtry=woffs+float(iw-1)*(woffe-woffs)/float(nwoff-1)
               rmst=0.
               jin=1
               do i=iplo,iphi
                  w0=3480.+float(i-1)*2.
                  w0=w0+wtrace(i,j)+wtry
                  call xlinint2(w0,nmaster,wmaster,xmaster,sp0,jin,jout)
                  jin=jout
                  rms0=(xspec(i,j)-sp0*xftf(i,j))**2
                  rmst=rmst+rms0
c                  print *,i,j,xspec(i,j),sp0*xftf(i,j)
               enddo
c               read *
c               if(j.lt.3) print *,j,iw,wtry,rmst
               if(rmst.lt.rmstmin) then
                  rmstmin=rmst
                  wmin=wtry
               endif
            enddo
            nin=nin+1
            xin(nin)=float(j)
            yin(nin)=wmin
            ymin=min(ymin,yin(nin))
            ymax=max(ymax,yin(nin))
         endif
      enddo

      ymin=-1.1
      ymax=1.1
      if(iplot.eq.1) then
         call pgenv(1.,112.,ymin,ymax,0,0)
         call pgline(nin,xin,yin)
      endif
      call smooth2(nin,xin,yin,nin2,xin2,yin2,13)
      nin=112
      do j=1,nin
         xin(j)=float(j)
      enddo
      call smooth(nin2,xin2,yin2,nin,xin,yin,yin3,0.)
      if(iplot.eq.1) then
         call pgsci(2)
         call pgline(nin2,xin2,yin2)
         call pgsci(4)
         call pgline(nin,xin,yin)
         call pgsci(1)
      endif

      do j=1,112
         if(ifib(j).eq.0) then
            woffa(ipa(j),j)=yin(j)
            iwoffa(ipa(j),j)=1
         endif
      enddo

      enddo

      do j=1,112
         nin=0
         do i=1,1032
            if(iwoffa(i,j).eq.1) then
               nin=nin+1
               xin(nin)=float(i)
               yin(nin)=woffa(i,j)
            endif
         enddo
         jin=1
         do i=1,1032
c            call xlinint(float(i),nin,xin,yin,yv)
            call xlinint2(float(i),nin,xin,yin,yv,jin,jout)
            jin=jout
            wtrace(i,j)=wtrace(i,j)+yv
         enddo
      enddo

      return
      end

      subroutine fitmaster(n,x,y,nwo,wfito,ofito,iuse,iplot)
      parameter(nmax=100000)
      real x(n),y(n),xs(nmax),ys(nmax),ys2(nmax),xtry(nmax)
      real ynew1(nmax),ynew2(nmax),xnew(nmax),xnew2(nmax)
      real yin1(nmax),yin2(nmax),yb1(100),yb2(100),xb1(100)
      real wfit(100),ofit(100),wfito(nwo),ofito(nwo),ofito2(nmax)
      real xin(nmax)

      if(iuse.eq.1) return

      xofflo0=-25.
      xoffhi0=18.
      xofflo1=-5.
      xoffhi1=5.
      xoffmin=0.
c      xofflo=-30.
c      xoffhi=-10.
      noff=400

c- first get the average offset from linear
      open(unit=3,file="/work/03946/hetdex/maverick/"
     $     //"virus_config/lib_calib/woff.in",status="old")

      do i=1,nwo
         read(3,*) x1,x2
c         ofito(i)=0.
         ofito(i)=x2
         ofito2(i)=0.
      enddo
      close(3)

      do iter=1,3
         xoffmin=0.

      do i=1,nwo
         ofito(i)=ofito2(i)+ofito(i)
      enddo

      jin=1
      do i=1,n
         call xlinint2(x(i),nwo,wfito,ofito,yv,jin,jout)
         jin=jout
         xtry(i)=x(i)-yv
      enddo

c- get solar and write into ys. Interpolate virus to solar bins into ys2
      open(unit=1,file='sun_use.dat',status='old')
      ns=0
      jin=1
      do i=1,nmax
         read(1,*,end=666) x1,x2
         ns=ns+1
         xs(ns)=x1
         ys(ns)=x2
         call xlinint2(xs(ns),n,xtry,y,yv,jin,jout)
         jin=jout
         ys2(ns)=yv
      enddo
 666  continue
      close(1)

c- nws is the smoothing for the continuum estimate, nw is for the fitter
      nws=30
      wstart=3510.
      wend=5420.
      wend=5500.
      nw=13
      wbin=(wend-wstart)/float(nw-1)
      wbins=(wend-wstart)/float(nws-1)
c      wbin=(wend-wstart)/float(nw)
c      wbins=(wend-wstart)/float(nws)

      do j=1,nws
         wlos=wstart+float(j-1)*wbins
         whis=wlos+wbins
         n3=0
         do k=1,ns
            if(xs(k).gt.wlos.and.xs(k).lt.whis) then
               n3=n3+1
               yin1(n3)=ys(k)
               yin2(n3)=ys2(k)
            endif
         enddo
         call biwgt(yin1,n3,xbout,xsout)
         yb1(j)=xbout
         call biwgt(yin2,n3,xbout,xsout)
         yb2(j)=xbout
         xb1(j)=(whis+wlos)/2.
      enddo

      do i=1,nw-1
         wlo=wstart+float(i-1)*wbin
         whi=wlo+wbin
         n2=0
         jin=1
         jin2=1
         do j=1,ns
            if(xs(j).gt.wlo-10..and.xs(j).lt.whi+10.) then
               n2=n2+1
               xnew(n2)=xs(j)
               call xlinint2(xnew(n2),nws,xb1,yb1,yv,jin,jout)
               jin=jout
               ynew1(n2)=ys(j)/yv
               call xlinint2(xnew(n2),nws,xb1,yb2,yv,jin2,jout2)
               jin2=jout2
               ynew2(n2)=ys2(j)/yv
            endif
         enddo

c- now get the offset, in terms of adding to the solar wavelengths (or negative virus)
         rmsmin=1e10
         if(i.eq.1) then
            xofflo=xofflo0+xoffmin
            xoffhi=xoffhi0+xoffmin
         else
            xofflo=xofflo1+xoffmin
            xoffhi=xoffhi1+xoffmin
         endif
c         xofflo=xofflo0
c         xoffhi=xoffhi0
         do k=1,noff
            xoff=xofflo+float(k-1)*(xoffhi-xofflo)/float(noff-1)
            rms=0.
            jin=1
            nb=0
            do j=3,n2-3
               call xlinint2(xnew(j)+xoff,n2,xnew,ynew2,yv,jin,jout)
               jin=jout
               rms=rms+(yv-ynew1(j))**2
               nb=nb+1
               xin(nb)=(yv-ynew1(j))**2
            enddo
            call biwgt(xin,nb,xbin,xsin)
c            print *,(whi+wlo)/2.,xoff,xbin
            if(xbin.lt.rmsmin) then
               rmsmin=xbin
               xoffmin=xoff
            endif
         enddo
         wfit(i)=(whi+wlo)/2.
         ofit(i)=xoffmin
         do j=1,n2
            xnew2(j)=xnew(j)-xoffmin
         enddo
         
         if(iplot.eq.1) then
            call pgenv(wlo,whi,0.4,1.3,0,0)
c            call pgenv(wlo-xoffmin,whi-xoffmin,0.4,1.3,0,0)
            call pgline(n2,xnew,ynew1)
            call pgsci(2)
            call pgline(n2,xnew2,ynew2)
            call pgsci(1)
         endif

      enddo
      
      do i=1,nwo
         ofito2(i)=ofito(i)
      enddo

      nfituse=nw-2
      if(iter.ge.2) nfituse=nw-1
c      do j=1,nw-1
c         print *,wfit(j),ofit(j)
c      enddo

      call smooth(nfituse,wfit,ofit,nwo,wfito,ofito,yin2,0.00)

      if(iplot.eq.1) then
         call pgenv(3500.,5500.,-25.,20.,0,0)
         call pgline(nfituse,wfit,ofit)
         call pgsci(2)
         call pgline(nwo,wfito,ofito)
         call pgsci(1)
      endif

      enddo

      do i=1,nwo
         ofito(i)=ofito2(i)+ofito(i)
      enddo

      return
      end

      subroutine getmaster(xspec,wtrace,nout,wout,xmout,xftf,ifib,
     $     iuse,iplot)
      parameter (narrm=2000)
      real xspec(narrm,112),wave(narrm*112),wout(narrm*112)
      real xmout(narrm*112),wmaster(narrm*112)
      real wtrace(narrm,112),xmaster(narrm*112),xftf(narrm,112)
      real xin(narrm),yin(narrm),xin2(narrm*112),yin2(narrm*112)
      real yin3(narrm*112),xp(narrm),yp(narrm)
      integer ifib(112)

      n=0

c- first get a simple ftf
      if(iuse.eq.0) then
         sum=0.
         nsum=0
         do j=1,112
            if(ifib(j).eq.0) then
               nin=0
               do i=400,700
                  nin=nin+1
                  xin(nin)=xspec(i,j)
               enddo
               call biwgt(xin,nin,xb,xs)
               sum=sum+xb
               nsum=nsum+1
               yin(j)=xb
            endif
         enddo
         sum=sum/float(nsum)
         do j=1,112
            if(ifib(j).eq.0) then
               yin(j)=yin(j)/sum
            else
               yin(j)=1.
            endif
         enddo
      endif

      n=0
      do j=1,112
         if(ifib(j).eq.0) then
            do i=1,1032
               if(xspec(i,j).ne.0) then
                  w0=3480.+float(i-1)*2.
                  n=n+1
                  xmaster(n)=xspec(i,j)
                  if(iuse.eq.1) xmaster(n)=xmaster(n)/xftf(i,j)
                  if(iuse.eq.0) xmaster(n)=xmaster(n)/yin(j)
                  wave(n)=w0+wtrace(i,j)
               endif
            enddo
         endif
      enddo

      call sort2(n,wave,xmaster)

      ism=6
      ism=11
      call smooth3(n,wave,xmaster,nout,wout,xmout,ism)

      if(iplot.eq.1) then
         ymin=1e10
         ymax=-ymin
         do i=1,nout
            ymin=min(ymin,xmout(i))
            ymax=max(ymax,xmout(i))
         enddo
         call pgenv(3600.,3800.,ymin,ymax,0,0)
c         call pgenv(4000.,4100.,ymin,ymax,0,0)
c         call pgenv(5450.,5500.,ymin,ymax,0,0)
         call pgline(n,wave,xmaster)
         call pgsci(2)
         call pgline(nout,wout,xmout)
         call pgsci(1)
      endif

c- skip getting new ftf if iuse=1
      if(iuse.eq.1) goto 666

      ilo=40
      ihi=990
      do j=1,112
         if(ifib(j).eq.0) then
            nin=0
            jin=1
            do i=1,1032
               w0=3480.+float(i-1)*2.
               w0=w0+wtrace(i,j)
               call xlinint2(w0,nout,wout,xmout,xm0,jin,jout)
               jin=jout
               yin(i)=xspec(i,j)/xm0
               if(i.gt.ilo.and.i.lt.ihi) then
                  nin=nin+1
                  xp(nin)=float(i)
                  yp(nin)=xspec(i,j)/xm0
               endif
            enddo
            if(iplot.eq.1) then
               call pgenv(1.,1032.,0.85,1.1,0,0)
               call pgline(nin,xp,yp)
            endif
            call smooth2(nin,xp,yp,nin2,xin2,yin2,25)
            nin=1032
            do i=1,nin
               xin(i)=float(i)
            enddo
            call smooth(nin2,xin2,yin2,nin,xin,yin,yin3,0.00)
            if(iplot.eq.1) then
               call pgsci(4)
               call pgline(nin,xin,yin)
               call pgsci(2)
               call pgline(nin2,xin2,yin2)
               call pgsci(1)
            endif
            do i=1,1032
               xftf(i,j)=yin(i)
c               if(i.gt.ilo.and.i.lt.ihi) xftf(i,j)=yin2(i-ilo)
            enddo
         else
            do i=1,1032
               xftf(i,j)=1.
            enddo
         endif
      enddo

 666  continue

      return
      end

      subroutine getpeaks(xspec,wtrace,nm,wm,xm,wavesol,wavesolo,
     $     xp,ipeaks,iuse,iplot)
      parameter (narrm=2000)
      real xspec(narrm,112),wavesol(1036),wavesolo(1036)
      real wtrace(narrm,112),wm(nm),xm(nm),xp(10),xpf(10)
      real xin(narrm*112),yin(narrm*112),yin2(narrm*112)
      integer ipeaks(10)

      if(iuse.eq.0) return

c- if iforce=0 then find peaks, otherwise force peaks
      iforce=1

c- find peaks in master
      npeak=6
      call smooth3(nm,wm,xm,nin,xin,yin,41)
      ymin=1e10
      ymax=-ymin
      do i=1,nm
         call xlinint(wm(i),nin,xin,yin,yv)
         yin2(i)=xm(i)-yv
         ymin=min(ymin,yin2(i))
         ymax=max(ymax,yin2(i))
      enddo

      if(iplot.eq.1) then
c         call pgenv(wm(1),wm(nm),ymin,ymax,0,0)
         call pgenv(5470.,5500.,ymin,ymax,0,0)
         call pgline(nm,wm,yin2)
      endif

      do ip=1,npeak
         ymax=-1e10
         do i=1,nm
            if(ip.gt.1) then
               do ipc=1,ip-1
                  if(abs(xp(ipc)-wm(i)).lt.15) goto 666
               enddo
            endif
            if(yin2(i).gt.ymax) then
               ymax=yin2(i)
               xmax=wm(i)
               imax=i
            endif
 666        continue
         enddo
         xp(ip)=xmax
         ipeaks(ip)=imax
         if(iplot.eq.1) print *,ip,xp(ip)
      enddo

      if(iforce.eq.1) then
         xpf(1)=5466.
         xpf(2)=4359.
         xpf(3)=3831.
         xpf(4)=3544.
         xpf(5)=4313.
         xpf(6)=5169.
         do i=1,npeak
            xpo=xp(i)
            jin=1
c            call xlinint2(xp(i),1036,wavesol,wavesolo,wv,
c     $           jin,jout)
            call xlinint2(xpf(i),1036,wavesol,wavesolo,wv,
     $           jin,jout)
c            jin=jout
            xp(i)=xpf(i)+wv
         enddo
      endif

      return
      end

      subroutine getwtrace(xspec,wout,ifib,iuse,iplot)
      parameter (narrm=2000)
      real xspec(narrm,112),wave(narrm)
      real win1(narrm),win2(narrm),xin1(narrm),xin2(narrm)
      real spec(narrm),woffa(20),soffa(20),xbinc(20),spect(narrm)
      real woffaf(narrm,112),soffaf(narrm,112),xl(1032),yl(1032)
      real wout(narrm,112)
      integer ifib(112)

      if(iuse.eq.1) return

      win=0.
      sin=1.
      iord=1
      nl=1032

      ifibp=55
      ifibh=1
      ifibn=3
      do i=1,1032
         wave(i)=3480.+float(i-1)*2.
         spec(i)=0.
         xl(i)=float(i)
         do j=1,112
            woffaf(i,j)=0.
            soffaf(i,j)=0.
            wout(i,j)=0.
         enddo
      enddo
      nfib=0
      do j=ifibp-ifbh,ifibp+ifibh
         if(ifib(j).eq.0) then
            nfib=nfib+1
            do i=1,1032
               spec(i)=spec(i)+xspec(i,j)
            enddo
         endif
      enddo
      do i=1,1032
         spec(i)=spec(i)/float(nfib)
      enddo

      if(iplot.eq.1) call pgenv(1.,1032.,-24.,24.,0,0)
      nbin=171
      nbin=200
      nfiball=0

c- bottom loop
      do j=ifibp,1,-ifibn
         nfiball=nfiball+1
         jlo=max(1,j-ifibh)
         jhi=min(112,j+ifibh)
         do i=1,1032
            spect(i)=0.
         enddo
         nfib=0
         do jt=jlo,jhi
            if(ifib(jt).eq.0) then
               nfib=nfib+1
               do i=1,1032
                  spect(i)=spect(i)+xspec(i,jt)
               enddo
            endif
         enddo
         do i=1,1032
            spect(i)=spect(i)/float(nfib)
         enddo
         nb=0
         do ibin=1,1032,nbin
            nin=0
            ymax=-1e10
            ymin=-ymax
            ilo=ibin
            ihi=min(1032,ibin+nbin)
            do i=ilo,ihi
               nin=nin+1
               win1(nin)=wave(i)
               win2(nin)=wave(i)
               xin1(nin)=spec(i)
               xin2(nin)=spect(i)
               ymax=max(ymax,xin1(nin),xin2(nin))
               ymin=min(ymin,xin1(nin),xin2(nin))
            enddo
            if(nin.gt.50) then
               nb=nb+1
               if(nfiball.gt.1) then
                  win=woffaf(nb,nfiball-1)
                  sin=soffaf(nb,nfiball-1)
               endif
               call getoffset(nin,win1,xin1,win2,xin2,win,sin,woff,soff)
               woffa(nb)=woff
               soffa(nb)=soff
               xbinc(nb)=float(ilo+ihi)/2.
               ip=nint(xbinc(nb))
               woffaf(nb,nfiball)=woffa(nb)
               soffaf(nb,nfiball)=soffa(nb)
            endif
         enddo
         call fitpoly(5,xbinc,woffa,iord,nl,xl,yl)
         do i=1,1032
            wout(i,j)=yl(i)
         enddo
         if(iplot.eq.1) then
            call pgsci(1)
            call pgline(5,xbinc,woffa)
            call pgsci(2)
            call pgline(nl,xl,yl)
            call pgsci(1)
         endif
      enddo

c- upper loop
      do j=ifibp,112,ifibn
         nfiball=nfiball+1
         jlo=max(1,j-ifibh)
         jhi=min(112,j+ifibh)
         do i=1,1032
            spect(i)=0.
         enddo
         nfib=0
         do jt=jlo,jhi
            if(ifib(jt).eq.0) then
               nfib=nfib+1
               do i=1,1032
                  spect(i)=spect(i)+xspec(i,jt)
               enddo
            endif
         enddo
         do i=1,1032
            spect(i)=spect(i)/float(nfib)
         enddo
         nb=0
         do ibin=1,1032,nbin
            nin=0
            ymax=-1e10
            ymin=-ymax
            ilo=ibin
            ihi=min(1032,ibin+nbin)
            do i=ilo,ihi
               nin=nin+1
               win1(nin)=wave(i)
               win2(nin)=wave(i)
               xin1(nin)=spec(i)
               xin2(nin)=spect(i)
               ymax=max(ymax,xin1(nin),xin2(nin))
               ymin=min(ymin,xin1(nin),xin2(nin))
            enddo
            if(nin.gt.50) then
               nb=nb+1
               if(nfiball.gt.1) then
                  win=woffaf(nb,nfiball-1)
                  sin=soffaf(nb,nfiball-1)
                  if(j.eq.ifibp) then
                     win=0.
                     sin=1.
                  endif
               endif
               call getoffset(nin,win1,xin1,win2,xin2,win,sin,woff,soff)
               woffa(nb)=woff
               soffa(nb)=soff
               xbinc(nb)=float(ilo+ihi)/2.
               ip=nint(xbinc(nb))
               woffaf(nb,nfiball)=woffa(nb)
               soffaf(nb,nfiball)=soffa(nb)
            endif
         enddo
         call fitpoly(5,xbinc,woffa,iord,nl,xl,yl)
         do i=1,1032
            wout(i,j)=yl(i)
         enddo
         if(iplot.eq.1) then
            call pgsci(1)
            call pgline(5,xbinc,woffa)
            call pgsci(2)
            call pgline(nl,xl,yl)
            call pgsci(1)
         endif
      enddo

      do i=1,1032
         nin=0
         do j=1,112
            if(wout(i,j).ne.0) then
               nin=nin+1
               xl(nin)=float(j)
               yl(nin)=wout(i,j)
            endif
         enddo
         do j=1,112
            if(wout(i,j).eq.0) then
               call xlinint(float(j),nin,xl,yl,yv)
               wout(i,j)=yv
            endif
         enddo
      enddo

      if(iplot.eq.1) then
         call pgenv(1.,112.,-24.,24.,0,0)
         do i=1,10
            nin=0
            do j=1,112
               nin=nin+1
               xl(nin)=float(j)
               yl(nin)=wout(i,j)
            enddo
            call pgsci(1)
            call pgline(nin,xl,yl)
         enddo
         do i=102,112
            nin=0
            do j=1,112
               nin=nin+1
               xl(nin)=float(j)
               yl(nin)=wout(i,j)
            enddo
            call pgsci(2)
            call pgline(nin,xl,yl)
         enddo
      endif

      return
      end

      subroutine getoffset(nin,win1,xin1,win2,xin2,win,sin,woff,soff)
      real win1(nin),xin1(nin),win2(nin),xin2(nin)

      nw=25
      nn=17
      w1=win-3.
      w2=win+3.
      x1=sin*0.8
      x2=sin*1.2

      rmsmin=1e10
      do i=1,nw
         wo=w1+float(i-1)*(w2-w1)/float(nw-1)
         do j=1,nn
            xn=x1+float(j-1)*(x2-x1)/float(nn-1)
            rms=0
            do inin=1,nin
               call xlinint(win2(inin)+wo,nin,win1,xin1,xp)
               rms=rms+(xp*xn-xin2(inin))**2
            enddo
            if(rms.lt.rmsmin) then
               rmsmin=rms
               woff=wo
               soff=xn
            endif
         enddo
      enddo
      do i=1,nin
         win2(i)=win2(i)+woff
         xin2(i)=xin2(i)/soff
      enddo

      return
      end

      subroutine extract(arr,xtrace,n,x,xprof,xspec,xspecr,xspecc,
     $     ifib,iuse)
      parameter (narrm=2000)
      real arr(narrm,narrm), xtrace(narrm,narrm),x(n)
      real xprof(1032,112,15),xspec(narrm,112),xin(narrm)
      real yin(narrm),yin2(narrm),xspecr(1032,112),xspecc(1032,112)
      real xspect(1032,112)
      integer ifib(112)

      rnoise=3.0

      if(iuse.eq.1) rmscut=30.
      if(iuse.eq.0) rmscut=1e10
      chicut=5.
      if(iuse.eq.0) chicut=1000.

      jhalf=4
      nhalf=(n-1)/2
      do j=1,112
         sum4=0.
         do i=1,1032
            j0=nint(xtrace(i,j))
            jlo=max(1,j0-jhalf)
            jhi=min(1032,j0+jhalf)
c- get the fiber profile
            do k=1,n
               xin(k)=xprof(i,j,k)
            enddo
c- rebin the fiber profile
            sum1=0.
            nin=0
            do k=jlo,jhi
               nin=nin+1
               xjp=float(k)-xtrace(i,j)
               call xlinint(xjp,n,x,xin,yjp)
               yin(nin)=yjp
               sum1=sum1+yin(nin)
            enddo
            sum2=0.
            sum3=0.
            sum5=0.
            nin=0
            nzero=0
            do k=jlo,jhi
               nin=nin+1
               yin(nin)=yin(nin)/sum1
               sum2=sum2+arr(i,k)*yin(nin)
               sum3=sum3+arr(i,k)
               sum5=sum5+yin(nin)**2
               if(arr(i,k).eq.0) nzero=nzero+1
            enddo
            sum2=sum2/sum5
            xspec(i,j)=sum2
            sum4=sum4+xspec(i,j)
            nin=0
            rms=0.
            chi=0.
            do k=jlo,jhi
               nin=nin+1
               xsq=(arr(i,k)-sum3*yin(nin))**2
               rms=rms+xsq
               xnoise2=rnoise*rnoise+sum3*yin(nin)
               chi=chi+xsq/xnoise2
            enddo
            if(nin.ge.1) then
               rms=sqrt(rms/float(nin))
               chi=sqrt(chi/float(nin))
            else
               rms=0.
               chi=0.
            endif
c            if(rms.gt.rmscut) xspec(i,j)=0.
            if(chi.gt.chicut) xspec(i,j)=0.
            xspecr(i,j)=rms
            xspecc(i,j)=chi
            if(ifib(j).eq.1) then
               xspec(i,j)=0.
               xspecr(i,j)=0.
               xspecc(i,j)=0.
            endif
c- check if >=3 of the elements are zero, then remove
            if(nzero.ge.3) then
               xspec(i,j)=0.
               xspecr(i,j)=0.
               xspecc(i,j)=0.
            endif
            if(xspec(i,j).gt.-1e10.and.xspec(i,j).lt.1e10) then
            else
               xspec(i,j)=0.
               xspecr(i,j)=0.
               xspecc(i,j)=0.
            endif
         enddo
      enddo

c- remove one pixel on either side of a flagged pixel
      do j=1,112
         do i=1,1032
            xspect(i,j)=xspec(i,j)
         enddo
      enddo
      do j=1,112
         do i=1,1032
            if(xspec(i,j).eq.0.) then
               ilo=max(1,i-1)
               ihi=min(1032,i+1)
               xspect(ilo,j)=0.
               xspect(ihi,j)=0.
            endif
         enddo
      enddo
      do j=1,112
         do i=1,1032
            xspec(i,j)=xspect(i,j)
         enddo
      enddo

      return
      end

      subroutine getfprof(arr,xtrace,n,x,yout,iuse,iplot)

      parameter (narrm=2000)
      real arr(narrm,narrm), xtrace(narrm,narrm)
      real xin(narrm*112),yin(narrm*112),xin3(narrm),yin3(narrm*112)
      real xin2(narrm*112),yin2(narrm*112),x(n),y(narrm)
      real yout(1032,112,15),yout1(1032,112,15),xout(1032,112)
      real yin4(narrm*112),xins(narrm*112),yins(narrm*112)
      real xin2s(narrm*112),yin2s(narrm*112)

      if(iuse.eq.1) return

      jhalf=4
      ihalf=9
      istep=2*ihalf+1

      do j=1,112
         ntot=0
         do i=1,1032-4,istep
            ntot=ntot+1
            xout(ntot,j)=float(i)
            j0=nint(xtrace(i,j))
            jlo=max(1,j0-jhalf)
            jhi=min(1032,j0+jhalf)
            ilo=max(1,i-ihalf)
            ihi=min(1032,i+ihalf)
            nin=0
            sum=0.
            do ip=ilo,ihi
               sumj=0.
               do jp=jlo,jhi
                  sumj=sumj+arr(ip,jp)
               enddo
               do jp=jlo,jhi
                  nin=nin+1
                  xin(nin)=float(jp)-xtrace(ip,j)
                  yin(nin)=arr(ip,jp)/sumj
                  sum=sum+yin(nin)
               enddo
            enddo
            do iadd=1,3
               nin=nin+1
               xin(nin)=-float(jhalf)-0.8
               yin(nin)=0.
               nin=nin+1
               xin(nin)=float(jhalf)+0.8
               yin(nin)=0.
            enddo
            call sort2(nin,xin,yin)
            ymax=-1e10
            do ip=1,nin
               yin(ip)=yin(ip)/sum
               ymax=max(ymax,yin(ip))
            enddo
            call smooth2(nin,xin,yin,nin2,xin2,yin2,3)
            sum=0
            do ia=1,n
               call xlinint0(x(ia),nin2,xin2,yin2,yv)
               y(ia)=yv
               sum=sum+y(ia)
               yout1(ntot,j,ia)=yv
            enddo

c- smooth the residuals
            do ia=1,nin
               call xlinint(xin(ia),n,x,y,yv)
               yins(ia)=yin(ia)-yv
            enddo
            call smooth2(nin,xin,yins,nin2s,xin2s,yin2s,7)
c            call smooth2(nin,xin,yins,nin2s,xin2s,yin2s,15)
            call smooth(nin2s,xin2s,yin2s,n,x,yin3,yin4,0.00)
            sum=0.
            do ia=1,n
               y(ia)=y(ia)+yin3(ia)
               sum=sum+y(ia)
            enddo
            do ia=1,n
               yout1(ntot,j,ia)=y(ia)/sum
            enddo

            if(iplot.eq.1) then
               call pgsci(1)
c               call pgenv(-5.,5.,-0.002,ymax,0,0)
               call pgenv(-5.,5.,-0.001,0.001,0,0)
               call pgpt(nin,xin,yin,17)
               call pgsci(2)
               call pgline(nin2,xin2,yin2)
               call pgsci(4)
               call pgline(n,x,y)
               call pgpt(nin,xin,yins,17)
               call pgline(nin2s,xin2s,yin2s)
               call pgsci(2)
               call pgline(n,x,yin3)
c               read *
            endif
         enddo
      enddo

      do j=1,112
         do k=1,n
            do i=1,ntot
               xin(i)=xout(i,j)
               yin(i)=yout1(i,j,k)
            enddo
            ymin=1e10
            ymax=-ymin
            nin2=0
            do i=1,1032
               call xlinint(float(i),ntot,xin,yin,yv)
               yout(i,j,k)=yv
               xin3(i)=float(i)
               yin3(i)=yv
               if(i.gt.40.and.i.lt.992) then
                  nin2=nin2+1
                  xin2(nin2)=float(i)
                  yin2(nin2)=yv
                  ymin=min(ymin,yv)
                  ymax=max(ymax,yv)
               endif
            enddo
            call smooth2(nin2,xin2,yin2,nin,xin,yin,155)
c            call pgenv(1.,1032.,ymin,ymax,0,0)
c            call pgsci(3)
c            call pgline(1032,xin3,yin3)
c            call pgsci(1)
c            call pgline(nin2,xin2,yin2)
c            call pgsci(2)
c            call pgline(nin,xin,yin)
            do i=1,1032
               xin2(i)=float(i)
            enddo
            call smooth(nin,xin,yin,1032,xin2,yin2,yin3,0.01)
            do i=1,1032
               yin2(i)=max(0.,yin2(i))
               yout(i,j,k)=yin2(i)
            enddo
c            call pgsci(4)
c            call pgline(1032,xin2,yin2)
c            call pgsci(1)
         enddo
      enddo

      return
      end

      subroutine getfloc(arr,cspecid,cpos,chalf,cifupos,cifu,
     $     xtrace,ifib,iuse,iplot)

      parameter (narrm=2000)
      real arr(narrm,narrm),xfib(112),xfiba(1032,112)
      real xin(1032),yin(1032),xin2(1032),yin2(1032),yin3(1032)
      real xtrace(narrm,narrm)
      integer ifib(112)
      character cpos*1,chalf*1,cspecid*3,cifu*3,cifupos*3
      character cfib*50

      icen=405
      icen=450

c- get the default fiber positions

      cfib="fiber_loc_"//cspecid//"_"//cifupos//
     $     "_"//cifu//"_"//cpos//chalf//".txt"

      open(unit=1,file=cfib,status='old',err=667)
      read(1,*)
      do i=1,112
         read(1,*,end=667) x1,i2
         xfib(i)=x1
         ifib(i)=i2
      enddo
 667  continue
      close(1)
     
      if(iuse.eq.1) return

      n2=1032
      do j=1,1032
         xin2(j)=float(j)
         do ia=1,112
            xfiba(j,ia)=0.
         enddo
      enddo

c- ihalf is in wavelength direction, jhalf is fiber profile

      ihalf=3
      jhalf=4

      do ia=1,112
         j0=nint(xfib(ia))
         i0=icen
         imin=max(1,i0-ihalf)
         imax=min(1032,i0+ihalf)
         jmin=max(1,j0-jhalf)
         jmax=min(1032,j0+jhalf)
         sum=0.
         sumd=0.
         do i=imin,imax
            do j=jmin,jmax
               sum=sum+arr(i,j)*float(j)
               sumd=sumd+arr(i,j)
            enddo
         enddo
         xold=xfib(ia)
         if(sumd.gt.0) then
            sum=sum/sumd
            xfiba(i0,ia)=sum
         else
            sum=0.
            xfiba(i0,ia)=xold
         endif
         if(ifib(ia).eq.1) xfiba(i0,ia)=xold
c         print *,i0,ia,xold,xfiba(i0,ia)
      enddo
      do itot=1,2
      do ia=1,112
         j0=nint(xfiba(i0,ia))
         i0=icen
         imin=max(1,i0-ihalf)
         imax=min(1032,i0+ihalf)
         jmin=max(1,j0-jhalf)
         jmax=min(1032,j0+jhalf)
         sum=0.
         sumd=0.
         do i=imin,imax
            do j=jmin,jmax
               sum=sum+arr(i,j)*float(j)
               sumd=sumd+arr(i,j)
            enddo
         enddo
         if(sumd.gt.0) then
            sum=sum/sumd
         else
            sum=0.
         endif
         xold=xfiba(i0,ia)
         xfiba(i0,ia)=sum
c         if(ia.eq.112) print *,i0,ia,xold,xfiba(i0,ia)
      enddo      
      enddo

      istep=2*ihalf+1
      do ia=1,112
         nstep=0
c         do iall=icen-istep-2,ihalf,-istep
         do iall=icen,ihalf,-istep
            nstep=nstep+1
            if(nstep.eq.1) then
               j0=nint(xfiba(icen,ia))
            else
               j0=jold
            endif   
            i0=iall
            imin=max(1,i0-ihalf)
            imax=min(1032,i0+ihalf)
            jmin=max(1,j0-jhalf)
            jmax=min(1032,j0+jhalf)
            sum=0.
            sumd=0.
            n=0
            do i=imin,imax
               do j=jmin,jmax
                  sum=sum+arr(i,j)*float(j)
                  sumd=sumd+arr(i,j)
                  n=n+1
               enddo
            enddo
            if(sumd.gt.0) then
               sum=sum/sumd
            else
               sum=0.
            endif
            xfiba(i0,ia)=sum
            jold=nint(xfiba(i0,ia))
         enddo         
      enddo

      do ia=1,112
         nstep=0
c         do iall=icen+istep,1032,istep
         do iall=icen,1032,istep
            nstep=nstep+1
            if(nstep.eq.1) then
               j0=nint(xfiba(icen,ia))
            else
               j0=jold
            endif   
            i0=iall
            imin=max(1,i0-ihalf)
            imax=min(1032,i0+ihalf)
            jmin=max(1,j0-jhalf)
            jmax=min(1032,j0+jhalf)
            sum=0.
            sumd=0.
            n=0
            do i=imin,imax
               do j=jmin,jmax
                  sum=sum+arr(i,j)*float(j)
                  sumd=sumd+arr(i,j)
                  n=n+1
               enddo
            enddo
            if(sumd.gt.0) then
               sum=sum/sumd
            else
               sum=0.
            endif
            xfiba(i0,ia)=sum
            jold=nint(xfiba(i0,ia))
         enddo         
      enddo

c- set missing fibers to adjacent value

      do j=1,112
         if(ifib(j).eq.1) then
            do i=1,1032
               if(j.gt.1.and.j.lt.112) then
                  if(xfiba(i,j-1).gt.0) then
                     xfiba(i,j)=(xfiba(i,j-1)+xfiba(i,j+1))/2.
                  endif
               endif
               if(j.eq.1.and.xfiba(i,j+1).gt.0) 
     $              xfiba(i,j)=max(1.,xfiba(i,j+1)-8.)
               if(j.eq.112.and.xfiba(i,j-1).gt.0) 
     $              xfiba(i,j)=min(1032.,xfiba(i,j-1)+8.)
            enddo
         endif
      enddo

      do j=1,112
         n=0
         ymax=-1e10
         ymin=-ymax
         do i=1,1032
            if(xfiba(i,j).gt.0) then
               n=n+1
               xin(n)=float(i)
               yin(n)=xfiba(i,j)
               ymax=max(ymax,yin(n))
               ymin=min(ymin,yin(n))
            endif
         enddo
         call sort2(n,xin,yin)
         call smooth(n,xin,yin,n2,xin2,yin2,yin3,0.001)
         if(iplot.eq.1) then
            call pgsci(1)
            call pgenv(1.,1032.,ymin,ymax,0,0)
            call pgline(n,xin,yin)
            call pgsci(2)
            call pgline(n2,xin2,yin2)
            call pgsci(1)
         endif
         do i=1,1032
            xtrace(i,j)=yin2(i)
         enddo
      enddo

      return
      end

      subroutine getifupos(cifu,cpos,chalf,xifupos)
      real xifupos(2,112),xt(224),yt(224)
      real xpos(448),ypos(448),xposa(448),yposa(448)
      character cifu*3,cpos*1,chalf*1
      character file1*80

c- order is, from bottom to top: LU,LL,RL,RU
      if(cpos.eq.'L'.and.chalf.eq.'U') istart=1
      if(cpos.eq.'L'.and.chalf.eq.'L') istart=113
      if(cpos.eq.'R'.and.chalf.eq.'L') istart=225
      if(cpos.eq.'R'.and.chalf.eq.'U') istart=337
      
      if(cifu.eq."004") then
         file1="/work/00115/gebhardt/maverick/scripts/back/ifu1.txt"
      else
         file1="/work/00115/gebhardt/maverick/scripts/back/ifu0.txt"
      endif
      open(unit=2,file=file1,status='old')
      do i=1,448
         read(2,*) i1,x2,x3
         xpos(i)=x2
         ypos(i)=x3
      enddo
      close(2)
      if(cifu.eq.'003'.or.cifu.eq.'004'
     $     .or.cifu.eq.'005'.or.cifu.eq.'008') then
         do i=1,224
            xt(i)=xpos(449-i)
            yt(i)=ypos(449-i)
         enddo
         do i=1,224
            xpos(224+i)=xt(i)
            ypos(224+i)=yt(i)
         enddo
      endif
      if(cifu.eq.'007') then
         xp=xpos(38)
         yp=ypos(38)
         xpos(38)=xpos(39)
         ypos(38)=ypos(39)
         xpos(39)=xp
         ypos(39)=yp
      endif
      if(cifu.eq.'025') then
         xp=xpos(209)
         yp=ypos(209)
         xpos(209)=xpos(214)
         ypos(209)=ypos(214)
         xpos(214)=xp
         ypos(214)=yp
      endif
      if(cifu.eq.'030') then
         xp=xpos(446)
         yp=ypos(446)
         xpos(446)=xpos(447)
         ypos(446)=ypos(447)
         xpos(447)=xp
         ypos(447)=yp
      endif
      if(cifu.eq.'038') then
         xp=xpos(303)
         yp=ypos(303)
         xpos(303)=xpos(304)
         ypos(303)=ypos(304)
         xpos(304)=xp
         ypos(304)=yp
      endif
      if(cifu.eq.'041') then
         xp=xpos(252)
         yp=ypos(252)
         xpos(252)=xpos(253)
         ypos(252)=ypos(253)
         xpos(253)=xp
         ypos(253)=yp
      endif

      n=0
      do i=istart,istart+111
         n=n+1
         np=113-n
         xifupos(1,np)=xpos(i)
         xifupos(2,np)=ypos(i)
      enddo

      return
      end

      subroutine getwsol(wavesol,wavesolo,wtrace0,wtrace,
     $     nmaster,wmaster,xmaster,iuse,iplot)
      parameter(nmax=2000)
      real wavesol(1036),wavesolo(1036),wtrace(nmax,112)
      real xin(nmax*112),wmaster(nmax*112),xmaster(nmax*112)
      real yin(nmax*112),xin2(nmax*112),yin2(nmax*112)
      real wtrace0(nmax,112)

      do j=1,112
         jin=1
         do i=1,1032
            w0=3480.+float(i-1)*2.
            call xlinint2(w0+wtrace(i,j),1036,wavesol,wavesolo,wv,
     $           jin,jout)
            jin=jout
            wtrace0(i,j)=wtrace(i,j)
            wtrace(i,j)=wtrace(i,j)+w0-wv
         enddo
      enddo

      if(iuse.eq.0) return

c- now fit to 5461 line
      xmin=5450.
      xmax=5475.
      ymin=1e10
      ymax=-ymin
      nin=0
      do i=1,nmaster
         call xlinint(wmaster(i),1036,wavesol,wavesolo,wv)
         wmaster(i)=wmaster(i)-wv
         if(wmaster(i).gt.xmin.and.wmaster(i).lt.xmax) then
            nin=nin+1
            xin(nin)=wmaster(i)
            yin(nin)=xmaster(i)
            ymin=min(ymin,yin(nin))
            ymax=max(ymax,yin(nin))
         endif
      enddo
      if(nin.eq.0) return

c      call smooth2(nin,xin,yin,nin2,xin2,yin2,13)
c      call smooth3(nin,xin,yin,nin2,xin2,yin2,9)
      call smooth3(nin,xin,yin,nin2,xin2,yin2,17)

      if(iplot.eq.1) then
         call pgenv(xmin,xmax,ymin,ymax,0,0)
         call pgline(nin,xin,yin)
         call pgsci(2)
         call pgline(nin2,xin2,yin2)
         call pgpt(nin2,xin2,yin2,17)
         call pgsci(1)
      endif
      ymax=0.
      do i=1,nin
         if(yin2(i).gt.ymax) then
            ymax=yin2(i)
            wmax=xin2(i)
            imax=i
         endif
      enddo
      x1=xin2(imax-1)
      y1=yin2(imax-1)
      x2=xin2(imax)
      y2=yin2(imax)
      x3=xin2(imax+1)
      y3=yin2(imax+1)
      p=0.5*(y1-y3)/(y1-2.*y2+y3)
      if(p.le.0) wv=x2+p*(x2-x1)
      if(p.gt.0) wv=x2+p*(x3-x2)

      woff=wmax-5461.
      woffq=wv-5461
      print *,"Wavelength offset: ",woff,woffq
      do j=1,112
         do i=1,1032
c            wtrace(i,j)=wtrace(i,j)-woffq
         enddo
      enddo

      return
      end

      subroutine fitpoly(n,x,y,iord,nl,xl,yl)
      parameter(nmax=10000,npm=10)
      real x(n),y(n),xl(nl),yl(nl),a(npm),sig(nmax)
      real covar(npm,npm),alpha(npm,npm)
      integer ia(npm)
      external funcs

      data itermax,tol/1000,1e-6/

      nca=npm
      nd=n
      do i=1,n
         sig(i)=0.01
      enddo
      np=iord+1
      nhalf=nint(float(n)/2.)
      do i=1,npm
         a(i)=0.
         ia(i)=0
      enddo
      a(1)=y(nhalf)
      ia(1)=1
      a(2)=(y(n)-y(1))/(x(n)-x(1))
      ia(2)=1

      alamda=-1.
      cold=1.e10
      do iter=1,itermax

         call mrqmin(x,y,sig,nd,a,ia,np,covar,alpha,nca,
     $        chisq,funcs,alamda)

         chirel=abs(cold-chisq)/chisq
         if(chirel.lt.tol) goto 669
      enddo
 669  continue

      call mrqmin(x,y,sig,nd,a,ia,np,covar,alpha,nca,
     $     chisq,funcs,0.)

      do i=1,nl
         yl(i)=a(1)+a(2)*xl(i)
      enddo

      return
      end
         
      subroutine funcs(x,a,y,dyda,na)
      real a(na),dyda(na)
      y=0.
      do i=1,na
         xp=x-1.
         y=y+a(i)*xp**(i-1)
         dyda(i)=x**(i-1)
      enddo
      return
      end

      subroutine smooth2(nin,xin,yin,nin2,xin2,yin2,ns)
      parameter(narrm=2000)
      real xin(nin),yin(nin),xin2(narrm*112),yin2(narrm*112)

      nin2=0
      do i=1,nin,ns
         n=0
         sum1=0.
         sum2=0.
         jmin=i
         jmax=min(nin,i+ns)
         do j=jmin,jmax
            n=n+1
            sum1=sum1+xin(j)
            sum2=sum2+yin(j)
         enddo
         sum1=sum1/float(n)
         sum2=sum2/float(n)
         nin2=nin2+1
         xin2(nin2)=sum1
         yin2(nin2)=sum2
      enddo

      return
      end

      subroutine smooth3(nin,xin,yin,nin2,xin2,yin2,ns)
      parameter(narrm=2000)
      real xin(nin),yin(nin),xin2(narrm*112),yin2(narrm*112)
      real yin3(narrm*112)

      nin2=0
      do i=1,nin,ns
         n=0
         sum1=0.
         sum2=0.
         jmin=i
         jmax=min(nin,i+ns)
         do j=jmin,jmax
            n=n+1
            sum1=sum1+xin(j)
            sum2=sum2+yin(j)
            yin3(n)=yin(j)
         enddo
         call biwgt(yin3,n,xb,xs)
         sum1=sum1/float(n)
         sum2=sum2/float(n)
         nin2=nin2+1
         xin2(nin2)=sum1
         yin2(nin2)=sum2
         yin2(nin2)=xb
      enddo

      return
      end

      subroutine smooth(n,x,y,n2,x2,y2,y3,sm)
      parameter(nmax=20000,mm=2,nwk=nmax+6*(nmax*mm+1),mm2=mm*2)
      real x(n),y(n),x2(n2),y2(n2),y3(n)
      real*8 dx(nmax),dy(nmax),wx(nmax),cf(nmax),wk(nwk),val,splder
      real*8 q(mm2)

      if(n.gt.nmax) print *,'make nmax bigger in smooth'

      val=0.d0
      val=0.001d0
      val=dble(sm)
      md=3
      if(val.eq.0.) md=2
      m=2

      do i=1,n
         dx(i)=dble(x(i))
         dy(i)=dble(y(i))
         wx(i)=1.d0
      enddo

      call gcvspl(dx,dy,nmax,wx,1.d0,m,n,1,md,val,cf,nmax,wk,ier)
c      if(ier.ne.0) print *,'ier= ',ier

      do i=1,n2
         in=i
         y2(i)=sngl(splder(0,m,n,dble(x2(i)),dx,cf,in,q))
      enddo

      do i=1,n
         in=i
         y3(i)=sngl(splder(0,m,n,dble(x(i)),dx,cf,in,q))
      enddo

      return
      end
