
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,1000),xin(1000),ya2(nmax)
      real yel(nmax),yeu(nmax),xg(nmax),yg(nmax),ydiff(nmax)
      real xball(nmax),xballa(nmax),xballa2(nmax),xballs(nmax)
      real xballr(nmax),fap(nmax),facapa(nmax),xin2(nmax)
      integer ibad(nmax),iap(nmax)
      character file1*80,file2*80

c- sigcut is the point-to-point normalization scatter
c- rmscut is the rms of the difference after normalization
      sigcut=0.15
      rmscut=0.01
      print *,"Input sigcut and rmscut (0.15,0.01):"
      read *,sigcut,rmscut

      open(unit=1,file=
     $     '/work/03946/hetdex/hdr1/calib/goodsed.dat',
     $     status='old')
      do i=1,nmax
         read(1,*,end=866) x1,x2
         xg(i)=x1
         yg(i)=x2
      enddo
 866  continue
      close(1)

      nap=0
      open(unit=1,file="apsum.out",status="old",err=555)
      do i=1,nmax
         read(1,*,end=555) i1,x2
         nap=nap+1
         iap(nap)=i1
         fap(nap)=x2
         xin(nap)=x2
      enddo
 555  continue
      close(1)

      xbap=1.
      if(nap.gt.0) then
         call biwgt(xin,nap,xbap,xsap)
         print *,"N_good and apcor: ",nap,xbap
      endif

      open(unit=1,file='list',status='old')

      nall=0
      nl=0
      do il=1,1000
         read(1,*,end=666) file1
         open(unit=2,file=file1,status='old')
         read(file1(3:7),'(i5)') iid
         if(nap.gt.0) then
            facap=xbap
            do ii=1,nap
               if(iap(ii).eq.iid) facap=fap(ii)
            enddo
         else
            facap=xbap
         endif
         facap=facap/xbap
         facapa(il)=facap

         nl=nl+1
         n=0
         do i=1,2000
            read(2,*,end=667) x1,x2
            n=n+1
c- this next line normalizes for edge effects. Not sure yet!
            x2=x2/facap
            x(n)=x1
            ya(n,il)=x2
c            ydiff(n)=x2-yg(n)
            ydiff(n)=x2/yg(n)
            ya2(n)=x2
         enddo
 667     continue
         close(2)
c remove the last point
         call biwgt(ydiff,n-1,xb,xs)
         rms=1e10
         if(xb.gt.0) then
            rms=0
            do i=1,n-1
               rms=rms+(yg(i)-ya2(i)/xb)**2
            enddo
            rms=sqrt(rms/float(n-1))
         endif
         call biwgt(ya2,n,xb2,xs2)
         ibad(il)=0
         xballa(il)=xb
         xballa2(il)=xb2
         xballs(il)=xs
         xballr(il)=rms
         if(rms.gt.rmscut) ibad(il)=1
         if(rms.gt.2.*rmscut) ibad(il)=2
         if(xs.gt.sigcut) ibad(il)=1
         if(xb2.lt.0.005) ibad(il)=1
         if(ibad(il).eq.0) then
            nall=nall+1
            xball(nall)=xb
         endif
      enddo
 666  continue
      close(1)
      call biwgt(xball,nall,xb,xs)

      open(unit=11,file='out',status='unknown')
      write(*,*) " ID    Offset    scale   use  Offset2   RMS"
      do i=1,nl
         if(ibad(i).eq.0) then
            write(*,1001) i,xballa(i),xballs(i),ibad(i),xballa2(i),
     $           xballr(i),facapa(i)
            write(11,1001) i,xballa(i),xballs(i),ibad(i),xballa2(i),
     $           xballr(i),facapa(i)
         else
            write(*,1001) i,0.,0.,ibad(i),0.,0.,facapa(i)
            write(11,1001) i,0.,0.,ibad(i),0.,0.,facapa(i)
         endif
      enddo
      close(11)

      open(unit=11,file='comb.out',status='unknown')
      do i=1,n
         nin=0
         nin2=0
         do j=1,nl
            if(ibad(j).eq.0) then
               nin=nin+1
               xin(nin)=ya(i,j)
            endif
            if(ibad(j).lt.2) then
               nin2=nin2+1
               xin2(nin2)=ya(i,j)
            endif
         enddo
         call biwgt(xin,nin,xb,xs)
         call biwgt(xin2,nin2,xb2,xs2)
         xb=max(0.,xb)
         xs=max(0.,xs)
         xb2=max(0.,xb2)
         y(i)=xb
         if(nin.gt.0) then
            yel(i)=y(i)-xs/sqrt(float(nin))
            yeu(i)=y(i)+xs/sqrt(float(nin))
         else
            yel(i)=0.
            yeu(i)=0.
         endif
         write(11,1101) x(i),y(i),yel(i),yeu(i),xb2
      enddo
      close(11)

 1001 format(i4,2(1x,f9.4),2x,i1,2(1x,f9.4),1x,f6.3)
 1101 format(f6.1,4(1x,f6.3))

      end
