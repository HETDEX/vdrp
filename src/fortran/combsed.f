
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,1000),xin(1000),ya2(nmax)
      real yel(nmax),yeu(nmax),xg(nmax),yg(nmax),ydiff(nmax)
      real xball(nmax),xballa(nmax),xballa2(nmax),xballs(nmax)
      real xballr(nmax)
      integer ibad(nmax)
      character file1*80,file2*80

c- sigcut is the point-to-point normalization scatter
c- rmscut is the rms of the difference after normalization
      sigcut=0.15
      rmscut=0.01
      print *,"Input sigcut and rmscut (0.15,0.01):"
      read *,sigcut,rmscut

      open(unit=1,file=
     $     '/work/00115/gebhardt/maverick/detect/sed/good.dat',
     $     status='old')
      do i=1,nmax
         read(1,*,end=866) x1,x2
         xg(i)=x1
         yg(i)=x2
      enddo
 866  continue
      close(1)

      open(unit=1,file='list',status='old')

      nall=0
      nl=0
      do il=1,1000
         read(1,*,end=666) file1
         open(unit=2,file=file1,status='old')
         nl=nl+1
         n=0
         do i=1,2000
            read(2,*,end=667) x1,x2
            n=n+1
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
         if(xs.gt.sigcut) ibad(il)=1
         if(xb2.lt.0.005) ibad(il)=1
         if(ibad(il).eq.0) then
            nall=nall+1
            xball(nall)=xb
         endif
c         print *,il,ibad(il),sigcut,xs,rms,rmscut
      enddo
 666  continue
      close(1)
      call biwgt(xball,nall,xb,xs)
c      print *,nl,nall,xb,xs

      open(unit=11,file='out',status='unknown')
      write(*,*) " ID    Offset    scale   use  Offset2   RMS"
      do i=1,nl
c         if(xballa(i).lt.(xb-xs)) ibad(i)=1
         if(ibad(i).eq.0) then
            write(*,1001) i,xballa(i),xballs(i),ibad(i),xballa2(i),
     $           xballr(i)
         endif
         write(11,1001) i,xballa(i),xballs(i),ibad(i),xballa2(i),
     $        xballr(i)
      enddo
      close(11)

      open(unit=11,file='comb.out',status='unknown')
      do i=1,n
         nin=0
         do j=1,nl
            if(ibad(j).eq.0) then
               nin=nin+1
               xin(nin)=ya(i,j)
            endif
         enddo
         call biwgt(xin,nin,xb,xs)
         y(i)=xb
         yel(i)=y(i)-xs/sqrt(float(nin))
         yeu(i)=y(i)+xs/sqrt(float(nin))
         write(11,*) x(i),y(i),yel(i),yeu(i)
      enddo
      close(11)

 1001 format(i4,2(1x,f9.4),2x,i1,2(1x,f9.4))

      end
