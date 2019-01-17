
      parameter (narrm=5048)
      real xd(narrm,narrm),xtrace(narrm,narrm)
      integer naxes(2),iwave(narrm)
      character file1*120,file2*130,file3*130,amp*14,a1*14,ae*5,aef*5
      logical simple,extend,anyf

      read *,file1
      read *,ifib,w0

      im1=0
      ier=0
      iread=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the wavelength                                                                                           
      iext=12
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      n=0
      wdiff0=1e10
      do i=1,ncol
         w=xd(i,ifib)
         wdiff=abs(w-w0)
         if(wdiff.lt.wdiff0) then
            wdiff0=wdiff
            imin=i
         endif
      enddo

      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the trace
      iext=13
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xtrace,anyf,ier)
      call ftclos(im1,ier)

      print *,imin,nint(xtrace(imin,ifib)),ifib,file1

      end

      subroutine xlinint(xp,n,x,y,yp)
      real x(n),y(n)
      do j=1,n-1
         if(xp.ge.x(j).and.xp.lt.x(j+1)) then
            yp=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
            if(y(j).eq.0.or.y(j+1).eq.0) yp=0.
            return
         endif
      enddo
      if(xp.lt.x(1)) yp=y(1)
      if(xp.gt.x(n)) yp=y(n)
c      if(xp.lt.x(1)) yp=0.
c      if(xp.gt.x(n)) yp=0.
      return
      end
