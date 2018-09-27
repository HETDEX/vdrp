
      parameter (narrm=5048)
      real xd(narrm,narrm),xdw(narrm,narrm)
      real wa(narrm),fa(narrm),waved(narrm)
      integer naxes(2),iwave(narrm)
      character file1*130,file2*130
      logical simple,extend,anyf

      read *,file1
      read *,file2

      open(unit=1,file=file2,status='old',err=678)
      na=0
      do i=1,narrm
         read(1,*,end=677) x1,x2
         na=na+1
         wa(na)=x1
         fa(na)=x2
      enddo
 677  continue
      close(1)
      goto 679
 678  continue
      na=3
      wa(1)=3500.
      fa(1)=1.
      wa(2)=4500.
      fa(2)=1.
      wa(3)=5500.
      fa(3)=1.
      write(*,*) "Amp Norm does not exist: ",file3
 679  continue

      im1=0
      ier=0
      iread=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the wavelength                                                                                           
      iext=5
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xdw,anyf,ier)
      call ftclos(im1,ier)


      im1=0
      ier=0
      call ftgiou(im1,ier)
      call ftopen(im1,file1,iread,iblock,ier)
c - this is the  spectrum
      iext=6
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      wave1=3550.
      wave2=4550.
      wave3=5450.

      call xlinint(wave1,na,wa,fa,f1)
      call xlinint(wave2,na,wa,fa,f2)
      call xlinint(wave3,na,wa,fa,f3)

      open(unit=11,file='out',status='unknown')
      do j=1,112
         wdiff1=1e10
         wdiff2=1e10
         wdiff3=1e10
         do i=1,ncol
            wd1=sqrt((xdw(i,j)-wave1)**2)
            wd2=sqrt((xdw(i,j)-wave2)**2)
            wd3=sqrt((xdw(i,j)-wave3)**2)
            if(wd1.lt.wdiff1) then
               wdiff1=wd1
               iw1=i
               jw1=j
            endif
            if(wd2.lt.wdiff2) then
               wdiff2=wd2
               iw2=i
               jw2=j
            endif
            if(wd3.lt.wdiff3) then
               wdiff3=wd3
               iw3=i
               jw3=j
            endif
         enddo
c         write(11,*) xd(iw1,j)*f1,xd(iw2,j)*f2,xd(iw3,j)*f3,j
         write(11,*) xd(iw1,j),xd(iw2,j),xd(iw3,j),j
      enddo
      close(11)

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

      SUBROUTINE moment(data,n,ave,adev,sdev,var,skew,curt)
      INTEGER n
      REAL adev,ave,curt,sdev,skew,var,data(n)
      INTEGER j
      REAL p,s,ep
c      if(n.le.1)pause 'n must be at least 2 in moment'
      s=0.
      do 11 j=1,n
        s=s+data(j)
11    continue
      ave=s/n
      adev=0.
      var=0.
      skew=0.
      curt=0.
      ep=0.
      do 12 j=1,n
        s=data(j)-ave
        ep=ep+s
        adev=adev+abs(s)
        p=s*s
        var=var+p
        p=p*s
        skew=skew+p
        p=p*s
        curt=curt+p
12    continue
      adev=adev/n
      var=(var-ep**2/n)/(n-1)
      sdev=sqrt(var)
      if(var.ne.0.)then
        skew=skew/(n*sdev**3)
        curt=curt/(n*var**2)-3.
      else
c        pause 'no skew or kurtosis when zero variance in moment'
      endif
      return
      END
