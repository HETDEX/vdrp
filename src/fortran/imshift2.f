
c-- rotates a fits image, but make sure that nbox is ODD!

      parameter (narrm=6048,nbox=7)
      real xd(narrm,narrm),xo(narrm,narrm)
      integer naxes(2),nflag(narrm,narrm),nflag2(narrm,narrm)
      character file1*40
      logical simple,extend,anyf

 1    call qc1('Image ','imshift.def',file1)
      call qi1('Which extension ','imshift.def',iext)
      call savdef

      ier=0
      im1=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 1
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(2).gt.10*narrm) naxes(2)=1
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)

      call ftg2de(im1,0,-666.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      call ftinit(50,'imshift.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
c      call ftcopy(im1,50,0,ier)
      if(ier.eq.107) ier=0
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         print *,'Try removing imshift.fits'
         goto 706
      endif

      call qr2('Shift in x and y ','imshift.def',dx,dy)
      call savdef

      if(dx.eq.0.and.dy.eq.0) then
         do j=1,nrow
            do i=1,ncol
               xo(i,j)=xd(i,j)
            enddo
         enddo
         goto 966
      endif

      cint=0
      if(abs(dx-float(nint(dx))).le.0.01.and.
     $   abs(dy-float(nint(dy))).le.0.01) cint=1

      if(cint.eq.1) then
         do j=1,nrow
            ja=j-nint(dy)
            ja=max(1,ja)
            ja=min(nrow,ja)
            do i=1,ncol
               ia=i-nint(dx)
               ia=max(1,ia)
               ia=min(ncol,ia)
               xo(i,j)=xd(ia,ja)
            enddo
         enddo
         goto 966
      endif

      do j=1,nrow
         do i=1,ncol
            xo(i,j)=0.
            nflag(i,j)=0
            nflag2(i,j)=0
         enddo
      enddo

      xcen=float(ncol)/2.
      ycen=float(nrow)/2.
      xbox=float(nbox)
      xbox2=xbox*xbox
      fac=1./xbox
      half=float((nbox+1)/2)*fac
      halfx=half+xcen
      halfy=half+ycen

      do j=1,nrow
         write(6,"('Doing ',i4,a1,$)") j,char(13)
         call flush(6)
         xj1=float(j)-halfy
         do i=1,ncol
            xi1=float(i)-halfx
            iflag=0
            if(nint(xd(i,j)).eq.-666) iflag=1
            val=xd(i,j)/xbox2
            do jb=1,nbox
               xj=xj1+float(jb)*fac
               do ib=1,nbox
                  xi=xi1+float(ib)*fac
                  ixn=nint(xi+dx+xcen)
                  iyn=nint(xj+dy+ycen)
                  if(ixn.ge.1.and.ixn.le.ncol.and.
     $                 iyn.ge.1.and.iyn.le.nrow) then
                     if(iflag.eq.0) then
                        xo(ixn,iyn)=xo(ixn,iyn)+val
                        nflag2(ixn,iyn)=nflag2(ixn,iyn)+1
                     endif
                     if(iflag.eq.1) nflag(ixn,iyn)=nflag(ixn,iyn)+1
                  endif
               enddo
            enddo
         enddo
      enddo

      ntot=nbox*nbox
      ncut=nint(float(ntot)/3.)
      do j=1,nrow
         do i=1,ncol
            if(nflag(i,j).lt.ncut.and.nflag(i,j).ne.0) then
               xo(i,j)=xo(i,j)*float(ntot)/float(ntot-nflag(i,j))
            elseif(nflag(i,j).ge.ncut) then
               xo(i,j)=-666.
            endif
            if(nflag2(i,j).eq.0) xo(i,j)=-666.
         enddo
      enddo

 966  continue
      call ftp2de(50,0,narrm,naxes(1),naxes(2),xo,ier)
      call ftclos(50,ier)

 706  continue
      end
