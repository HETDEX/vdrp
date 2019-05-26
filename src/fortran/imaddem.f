
      parameter (narrm1=3000,narrm2=3000)
      real xd(narrm1,narrm2),wvadd(narrm1),wv(narrm1,narrm2)
      real f2f(narrm1,narrm2),wa(narrm1),fa(narrm1)
      real xd2(narrm1,narrm2),xpos(narrm2),ypos(narrm2),xfa(narrm2)
      integer naxes(2)
      character file1*120
      logical simple,extend,anyf
      parameter(pi=3.141593e0)      
      common/csigma/ rsig,fmof,bmof,imoff

      imoff=0
      imoff=1

      read(*,*) invert ! 0 for add lines, 1 for invert by -1

c- get the fwhm
      open(unit=1,file='norm.use',status='old',err=910)
      read(1,*) file1,rfw,enorm
      goto 911
 910  continue
      rfw=1.9
      enorm=1.0
 911  continue
      close(1)
      fmof=rfw
      bmof=3.5
c- get the amp2amp
      open(unit=1,file='anorm.use',status='old',err=912)
      na=0
      do i=1,narrm1
         read(1,*,end=914) x1,x2
         na=na+1
         wa(na)=x1
         fa(na)=x2
      enddo
 914  continue
      goto 913
 912  continue
      na=2
      wa(1)=3480.
      wa(2)=5540.
      fa(1)=1.
      fa(2)=1.
 913  continue
      close(1)

      rsig=rfw/2.35
      wsig=2.3
      dwave=1.97
      h4=0.
      open(unit=1,file="offset",status="old")
      read(1,*) xoff,yoff
      close(1)

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      file1='ifucen.fits'
      call ftopen(im1,file1,iread,iblock,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)
      do i=1,nrow
         xpos(i)=xd(1,i)-xoff
         ypos(i)=xd(2,i)-yoff
c         print *,i,xpos(i),ypos(i)
      enddo

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      file1='wv.fits'
      call ftopen(im1,file1,iread,iblock,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,wv,anyf,ier)
      call ftclos(im1,ier)

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      file1='fb.fits'
      call ftopen(im1,file1,iread,iblock,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,f2f,anyf,ier)
      call ftclos(im1,ier)

      file1='skysub.fits'
      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         call ftclos(im1,ier)
         write(*,*) 'Error opening image : ',file1
      endif
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxis.eq.1) naxes(2)=1
      if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm1,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      do j=1,nrow
         do i=1,ncol
            xd2(i,j)=xd(i,j)
         enddo
      enddo

      if(invert.eq.0) then
         open(unit=1,file='inlist',status='old')
         open(unit=11,file='outlist',status='unknown')
         ntot=0
         ntotm=1000
         do iall=1,1000
            read(1,*,end=666) x,y,w,xf
            xf=xf*enorm
            call getflux(x,y,nrow,xpos,ypos,xf,xfa,frach,fracout)
            if(frach.gt.0) then
               ntot=ntot+1
               if(ntot.gt.ntotm) goto 666
               sumf=0.
               do j=1,nrow
                  sumf=sumf+xfa(j)
               enddo
               write(11,*) x,y,w,xf,sumf/xf
            endif
            sum1=0.
            sum2=0.
            sumt=0.
            do j=1,nrow
               xfrac=xfa(j)
               do i=1,ncol
                  wvadd(i)=0.
               enddo
               if(xfrac.gt.0) then
                  sum=0.
                  do i=1,ncol
                     xp=wv(i,j)
                     wg=(xp-w)/wsig
                     gaus=exp(-wg*wg/2.)/sqrt(2.*wsig*wsig*pi)*dwave
                     wvadd(i)=xfrac*gaus*(1.+h4*fh4(wg))
                     call xlinint(wv(i,j),na,wa,fa,a2a)
                     wvadd(i)=wvadd(i)*a2a*f2f(i,j)
                     sum=sum+wvadd(i)
                  enddo
                  do i=1,ncol
                     xd2(i,j)=xd2(i,j)+wvadd(i)
                  enddo
               endif
            enddo
         enddo
 666     continue
         close(1)
         close(11)
      else
         do j=1,nrow
            do i=1,ncol
               xd2(i,j)=-xd2(i,j)
            enddo
         enddo
      endif

      ier=0
      call ftinit(50,'imadd.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm1,naxes(1),naxes(2),xd2,ier)
      call ftclos(50,ier)

 706  continue
      end

      subroutine getflux(x,y,n,xpos,ypos,xf,xfa,frach,sum)
      real xpos(n),ypos(n),xfa(n)
      parameter(pi=3.141593e0)
      common/csigma/ rsig,fmof,bmof,imoff

      rfib=1.55/2.
      ngrid=400
      xsig=8.
      xs=x-xsig*rsig
      xe=x+xsig*rsig
      ys=y-xsig*rsig
      ye=y+xsig*rsig
      deltx=(xe-xs)/float(ngrid)
c      area=xf*deltx**2
      area=xf*deltx*deltx/(2.*rsig*rsig*pi)
      areamoff=4.*(2.**(1./bmof)-1.)*(bmof-1.)/pi/fmof/fmof
      areamoff=xf*deltx*deltx*areamoff

      xstep=(xe-xs)/float(ngrid-1)
      ystep=(ye-ys)/float(ngrid-1)
      do i=1,n
         xfa(i)=0.
      enddo
      ntot=0
      nhit=0
      do ix=1,ngrid
         xp=xs+float(ix-1)*xstep
         do iy=1,ngrid
            yp=ys+float(iy-1)*ystep
            ntot=ntot+1
            do i=1,n
               dist=sqrt((xp-xpos(i))**2+(yp-ypos(i))**2)
               if(dist.lt.rfib) then
                  dist2=sqrt((x-xp)**2+(y-yp)**2)
                  g=dist2/rsig
                  gaus=exp(-g*g/2.)*area
                  xmoff=areamoff*((1.+4.*(2.**(1./bmof)-1.)*
     $                 (dist2/fmof)**2)**(-bmof))
                  if(imoff.eq.1) gaus=xmoff
                  xfa(i)=xfa(i)+gaus
                  nhit=nhit+1
                  goto 666
               endif
            enddo
 666        continue
         enddo
      enddo
      fracm=float(ntot-nhit)/float(ntot)
      frach=float(nhit)/float(ntot)
      sum=0.
      do i=1,n
         sum=sum+xfa(i)
      enddo

      return
      end

      function fh4(x)
      fh4=1./sqrt(24.)*(4.*x*x*x*x-12.*x*x+3.)
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
