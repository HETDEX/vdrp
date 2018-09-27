
      parameter (narrm=5000)
      real xc1(narrm,narrm),xd(narrm*2,narrm*2)
      integer naxes(2)
      character file1*100

      nside=8
      n1out=nside*60
      n2out=n1out
      do i=1,n1out
         do j=1,n2out
            xd(i,j)=-1e6
         enddo
      enddo
      open(unit=1,file='j0',status='old')
      do irow=1,nside
         do icol=1,nside
            read(1,*,end=666) file1
            ier=0
            call geti(file1,ncol,nrow,xc1,ier)
            if(ier.eq.1) goto 999

            is=(icol-1)*(51+10)
            ie=is+ncol-1
            js=(irow-1)*(51+10)
            je=js+nrow-1
            do i=is,ie
               ia=i-is+1
               do j=js,je
                  ja=j-js+1
                  xd(i,j)=xc1(ia,ja)
               enddo
            enddo
 999        continue
         enddo
      enddo
 666  continue
      close(1)

      naxis=2
      naxes(1)=n1out
      naxes(2)=n2out
      iblock=1
      igc=1
      ier=0
      call ftinit(50,'imdao.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm*2,naxes(1),naxes(2),xd,ier)
      call ftclos(50,ier)

 706  continue
      end

      subroutine geti(file1,ncol,nrow,x,ierro)
      parameter(narr=5000)
      real x(narr,narr)
      integer naxes(2)
      character file1*100
      logical simple,extend,anyf
      ier=0
      im1=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         ierro=1
         goto 706
      endif
c      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narr.or.naxes(2).gt.narr) then
         write(*,"('Arrays too small - make narr bigger')")
         write(*,"('Axes equal to ')") naxes(1),naxes(2)
         goto 706
      endif
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narr,ncol,nrow,x,anyf,ier)
      call ftclos(im1,ier)
      return
 706  continue
      return
      end
