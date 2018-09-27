
c-- rotates a fits image, but make sure that nbox is ODD!

      parameter (narrm=2048,nbox=21)
      real xd(narrm,narrm),xo(narrm,narrm)
      integer naxes(2),nflag(narrm,narrm)
      character file1*40
      logical simple,extend,anyf

 1    call qc1('Image ','imrot.def',file1)
      call qi1('Which extension ','imrot.def',iext)
      call savdef

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 1
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
      call ftclos(im1,ier)

      call qr1('Rotation angle in deg (+=cw) ','imrot.def',theta)
      call savdef
      theta=theta/57.29578

      do j=1,nrow
         do i=1,ncol
            xo(i,j)=0.
            nflag(i,j)=0
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
      a1=cos(theta)
      a2=sin(theta)
      a3=-sin(theta)
      a4=cos(theta)

      do j=1,nrow
c         write(6,"('Doing ',i4,a1,$)") j,char(13)
c         call flush(6)
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
                  ixn=nint(a1*xi+a2*xj+xcen)
                  iyn=nint(a3*xi+a4*xj+ycen)
                  if(ixn.ge.1.and.ixn.le.ncol.and.
     $                 iyn.ge.1.and.iyn.le.nrow) then
                     if(iflag.eq.0) xo(ixn,iyn)=xo(ixn,iyn)+val
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
         enddo
      enddo

      print *,ier

c-- open the output file                                                                                                        
      ier=0
      call ftgiou(im1,ier)
      call ftinit(im1,'imrot.fits',iblock,ier)
      call ftphps(im1,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         print *,'Try removing imrot.fits'
         goto 706
      endif

      call ftp2de(im1,igc,narrm,naxes(1),naxes(2),xo,ier)
      call ftclos(im1,ier)

 706  continue
      end
