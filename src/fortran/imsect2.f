
      parameter(nmax=4000)
      real xd(nmax,nmax),xo(nmax,nmax)
      integer naxes(2),is(nmax),ie(nmax),js(nmax),je(nmax)
      character file1*120
      logical simple,extend,anyf

      read *,file1,iext
      print *,file1(1:40)

      open(unit=1,file='region.use',status='old')
      nr=0
      n1=0
      do i=1,100
         read(1,*,end=666) i1,i2,i3,i4
         nr=nr+1
         is(nr)=i1
         ie(nr)=i2
         js(nr)=i3
         je(nr)=i4
         n1=n1+i2-i1+1
      enddo
 666  continue
      close(1)
      n2=i4-i3+1
      n1=n1+(nr-1)*2

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      iblock=1
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 706
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,nmax,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

c      print *,"Before : ",naxes(1),naxes(2)

      naxes(1)=n1
      naxes(2)=n2

c      print *,"After  : ",naxes(1),naxes(2)

      do j=js(1),je(1)
         do i=1,n1
            xo(i,j)=0
         enddo
         ic=0
         do ir=1,nr
            if(ir.gt.1) ic=ic+2
            do i=is(ir),ie(ir)
               ic=ic+1
               xo(ic,j)=xd(i,j)
            enddo
         enddo
      enddo


      im1=0
      ier=0
      call ftinit(50,'imsect.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,nmax,naxes(1),naxes(2),xo,ier)
      call ftclos(50,ier)

 706  continue
      end
