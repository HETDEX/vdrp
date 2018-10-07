
      parameter (narrm=5000)
      real xc1(narrm,narrm),xd(narrm*2,narrm*2)
      integer naxes(2)
      character file1*100

      icut=1
      iflip=1
      n1out=1300
      n2out=1300
      do i=1,n1out
         do j=1,n2out
            xd(i,j)=0.
         enddo
      enddo
      ioff=650
      joff=650
      open(unit=1,file='infp',status='old')
      do iall=1,1000
         read(1,*,end=666) file1,i2,x3,x4
         ier=0
         call geti(file1,ncol,nrow,xc1,ier)
         if(ier.eq.1) goto 999
         if(iflip.eq.1) then
            x3t=x3
            x3=x4
            x4=x3t
         endif

         if(icut.eq.1) then
            if(i2.eq.82) then
               do ix=1,51
                  do iy=1,25
                     xc1(ix,iy)=0.
                  enddo
               enddo
            endif
            if(i2.eq.103) then
               do ix=1,51
                  do iy=37,51
                     xc1(ix,iy)=0.
                  enddo
               enddo
            endif
         endif

         is=ioff+nint(x3)-25.5
         ie=is+ncol-1
         js=joff+nint(x4)-25.5
         je=js+nrow-1
c         print *,is,ie,js,je
         do i=is,ie
            ia=i-is+1
            do j=js,je
               ja=j-js+1
               xd(i,j)=xc1(ia,ja)
            enddo
         enddo
 999     continue
      enddo
 666  continue
      close(1)

      naxis=2
      naxes(1)=n1out
      naxes(2)=n2out
      iblock=1
      igc=1
      ier=0
      call ftinit(50,'immosaic.fits',iblock,ier)
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
