
      parameter (narrm1=14000,narrm2=4000)
      real xd(narrm1,narrm2),xd2(narrm1,narrm2)
      integer naxes(2)
      character file1*120
      logical simple,extend,anyf

      dflag=0.

 1    call qc1('First image ','imar.def',file1)
      call qi1('Which extension ','imar.def',iext1)
      call savdef

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im1,iext1,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxis.eq.1) naxes(2)=1
      if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
         print *,naxes(1),naxes(2)
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,-666.,narrm1,ncol,nrow,xd,anyf,ier)

 2    call qc1('Second image ','imar.def',file1)
      call qi1('Which extension ','imar.def',iext2)
      call savdef

      im2=0
      ier=0
      call ftgiou(im2,ier)
      iread=0
      call ftopen(im2,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im2,iext2,ihd,ier)
      call ftghpr(im2,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im2,igc,-666.,narrm1,ncol,nrow,xd2,anyf,ier)
      call ftclos(im2,ier)

 3    call qi1('Add (1), Sub (2), Mult (3), or Div (4) ',
     $     'imar.def',itype)
      if(itype.lt.1.or.itype.gt.4) goto 3
      call savdef

      if(itype.eq.1) then
         do j=1,nrow
            do i=1,ncol
               if(xd(i,j).ne.dflag) then
                  xd(i,j)=xd(i,j)+xd2(i,j)
               else
                  xd(i,j)=dflag
               endif
            enddo
         enddo
      elseif(itype.eq.2) then
         do j=1,nrow
            do i=1,ncol
               if(xd(i,j).ne.dflag) then
                  xd(i,j)=xd(i,j)-xd2(i,j)
               else
                  xd(i,j)=dflag
               endif
            enddo
         enddo
      elseif(itype.eq.3) then
         do j=1,nrow
            do i=1,ncol
               if(xd(i,j).ne.dflag) then
                  xd(i,j)=xd(i,j)*xd2(i,j)
               else
                  xd(i,j)=dflag
               endif
            enddo
         enddo
      elseif(itype.eq.4) then
         do j=1,nrow
            do i=1,ncol
               if(xd(i,j).ne.dflag) then
                  if(xd2(i,j).ge.0.06) then
                     xd(i,j)=xd(i,j)/xd2(i,j)
                  else
                     xd(i,j)=0.
                  endif
               else
                  xd(i,j)=dflag
               endif
            enddo
         enddo
      endif

      call ftclos(im1,ier)

c-- open the output file
      ier=0
      call ftgiou(im1,ier)
      call ftinit(im1,'imar.fits',iblock,ier)
      call ftphps(im1,-32,naxis,naxes,ier)
c      call ftcopy(im1,50,0,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(im1,igc,narrm1,naxes(1),naxes(2),xd,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftclos(im1,ier)

 706  continue
      end
