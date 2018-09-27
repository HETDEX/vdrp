
      parameter (narrm1=24000,narrm2=4000)
      real xd(narrm1,narrm2),xd2(narrm1,narrm2),xd3(narrm1,narrm2)
      real xs(narrm1),ys(narrm1),xde(narrm1,narrm2)
      integer naxes(2)
      character file1*120
      logical simple,extend,anyf

      iext1=1
      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,'data.fits',iread,iblock,ier)
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

      iext2=1

      im2=0
      ier=0
      call ftgiou(im2,ier)
      iread=0
      call ftopen(im2,'newfb.fits',iread,iblock,ier)
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

      im2=0
      ier=0
      call ftgiou(im2,ier)
      iread=0
      call ftopen(im2,'wv.fits',iread,iblock,ier)
      call ftmahd(im2,iext2,ihd,ier)
      call ftghpr(im2,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm1.or.naxes(2).gt.narrm2) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im2,igc,-666.,narrm1,ncol,nrow,xd3,anyf,ier)
      call ftclos(im2,ier)

      open(unit=1,file='in',status='old')
      ns=0
      do i=1,narrm1
         read(1,*,end=666) x1,x2
         ns=ns+1
         xs(ns)=x1
         ys(ns)=x2
      enddo
 666  continue
      close(1)

      xeff=3.5
      rnoise=3.0
      do j=1,nrow
         do i=1,ncol
            wave=xd3(i,j)
            if(i.eq.1) then
               xd31=xd3(i,j)-(xd3(i+1,j)-xd3(i,j))
            else
               xd31=xd3(i-1,j)
            endif
            if(i.eq.ncol) then
               xd32=xd3(i,j)+(xd3(i,j)-xd3(i-1,j))
            else
               xd32=xd3(i+1,j)
            endif
            w1=(xd31+wave)/2.
            w2=(xd32+wave)/2.
            sumw=0.
            nsumw=0
            do iw=1,ns-1
               if(xs(iw).gt.w1.and.xs(iw+1).lt.w2) then
                  sumw=sumw+ys(iw)
                  nsumw=nsumw+1
               endif
            enddo
            if(nsumw.gt.0) sumw=sumw/float(nsumw)
            call xlinint(wave,ns,xs,ys,val)
c            val=sumw
            skysub=val*xd2(i,j)
            if(abs(xd(i,j)).gt.0.00001) then
               xd(i,j)=xd(i,j)-skysub
            else
               xd(i,j)=0.
            endif
            xval=sumw+xeff*rnoise
            if(val.gt.0) then
               xde(i,j)=sqrt(xval)
            else
               xde(i,j)=0
            endif
         enddo
      enddo

      call ftclos(im1,ier)

c-- open the output file
      ier=0
      call ftgiou(im1,ier)
      call ftinit(im1,'imar.fits',iblock,ier)
      call ftphps(im1,-32,naxis,naxes,ier)
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

c-- open the output file
c      ier=0
c      call ftgiou(im1,ier)
c      call ftinit(im1,'imare.fits',iblock,ier)
c      call ftphps(im1,-32,naxis,naxes,ier)
c      if(ier.ne.0) then
c         print *,'Error in output file ',ier
c         goto 706
c      endif
c      call ftp2de(im1,igc,narrm1,naxes(1),naxes(2),xde,ier)
c      if(ier.ne.0) then
c        print *,'Error in output file ',ier
c        goto 706
c      endif
c      call ftclos(im1,ier)

 706  continue
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
