
      parameter (narrm1=3000,narrm2=3000)
      real xd(narrm1,narrm2),wvadd(narrm1),wv(narrm1,narrm2)
      real xd2(narrm1,narrm2),xpos(narrm2),ypos(narrm2),xfa(narrm2)
      integer naxes(2)
      character file1*120
      logical simple,extend,anyf
      parameter(pi=3.141593e0)      

      rfw=1.5
      rsig=rfw/2.35
      wsig=2.3
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

      open(unit=1,file='inlist',status='old')
      open(unit=11,file='outlist',status='unknown')
      ntot=0
      ntotm=25
      do iall=1,1000
         read(1,*,end=666) x,y,w,xf
         call getflux(x,y,nrow,xpos,ypos,xf,xfa,rsig,frach)
c         print *,iall,frach
         if(frach.gt.0) then
            ntot=ntot+1
            if(ntot.gt.ntotm) goto 666
            write(11,*) x,y,w,xf,frach
         endif
         sum1=0.
         sum2=0.
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
                  gaus=exp(-wg*wg/2.)/sqrt(2.*wsig*wsig*pi)
                  wvadd(i)=xfrac*gaus*(1.+h4*fh4(wg))
                  sum=sum+wvadd(i)
               enddo
               do i=1,ncol
                  xd2(i,j)=xd2(i,j)+wvadd(i)/sum*xfrac
               enddo
            endif
         enddo
      enddo
 666  continue
      close(1)
      close(11)

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

      subroutine getflux(x,y,n,xpos,ypos,xf,xfa,rsig,frach)
      real xpos(n),ypos(n),xfa(n)
      parameter(pi=3.141593e0)
      rfib=1.5/2.
      ngrid=100
      xs=x-6*rsig
      xe=x+6*rsig
      ys=y-6*rsig
      ye=y+6*rsig
      xstep=(xe-xs)/float(ngrid-1)
      ystep=(ye-ys)/float(ngrid-1)
      do i=1,n
         xfa(i)=0.
      enddo
      ntot=0
      nhit=0
      do xp=xs,xe,xstep
         do yp=ys,ye,ystep
            ntot=ntot+1
            do i=1,n
               dist=sqrt((xp-xpos(i))**2+(yp-ypos(i))**2)
               if(dist.lt.rfib) then
                  g=dist/rsig
                  gaus=exp(-g*g/2.)/sqrt(2.*rsig*rsig*pi)
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
      do i=1,n
         if(sum.gt.0) then
            xfa(i)=xfa(i)/sum*xf*frach
         else
            xfa(i)=0.
         endif
      enddo
      return
      end

      function fh4(x)
      fh4=1./sqrt(24.)*(4.*x*x*x*x-12.*x*x+3.)
      return
      end


