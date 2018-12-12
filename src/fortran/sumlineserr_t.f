
      parameter(nmax=10000)
      real x(nmax),y(nmax),y2(nmax),ysum(nmax),ysum2(nmax)
      real ysum3(nmax),gna(nmax),ye(nmax),ysum3e(nmax)
      real yin1(nmax),yin2(nmax),ysume(nmax),ysum2e(nmax)
      real gsum(nmax),wv(5),wn(5),gnw(nmax)
      character file1*80,file2*80,c1*8

      ylocut=-100.
      yhicut=1000.
      fcut=0.03

      open(unit=1,file='list2',status='old',err=866)

      do i=1,nmax
         ysum(i)=0.
         ysum2(i)=0.
         ysume(i)=0.
         ysum2e(i)=0.
         ysum3(i)=0.
         gsum(i)=0.
         gnw(i)=0.
      enddo
      nw=5
      wv(1)=3500.
      wv(2)=4000.
      wv(3)=4500.
      wv(4)=5000.
      wv(5)=5500.

c - sumgall is total counts (amplitude of fit)
c   gna is normalized counts for each
      sumg=0.
      ng=0
      do ig=1,1000
         read(1,*,end=777) file1,iflag,gn
         if(iflag.eq.0) then
            ng=ig
            gna(ng)=gn
            sumg=sumg+gn
         endif
      enddo
 777  continue
      if(ng.eq.0) goto 866
      rewind(1)
      do i=1,ng
         gna(i)=gna(i)/sumg
      enddo
      sumgall=sumg

c - first get normalization for each wavelength      
      do il=1,1000
         read(1,*,end=966) file1,iflag,gn,wn(1),wn(2),wn(3),wn(4),wn(5)
         if(gna(il).lt.fcut) iflag=1
         if(iflag.eq.0) then
            open(unit=2,file=file1,status='old')
            n=0
            do i=1,2000
               read(2,*,end=967) x1,x2,x3,x4,x5,x6,x7,x8,x9
               call xlinint(x1,nw,wv,wn,fadc)
               n=n+1
               gnw(n)=gnw(n)+gn*fadc
            enddo
 967        continue
            close(2)
         endif
      enddo
 966  continue
      rewind(1)

c - get the weighted sum
      do il=1,1000
c         read(1,*,end=666) file1,iflag,gn
         read(1,*,end=666) file1,iflag,gn,wn(1),wn(2),wn(3),wn(4),wn(5)
         gn0=gn
         if(gna(il).lt.fcut) iflag=1
         if(iflag.eq.0) then
            open(unit=2,file=file1,status='old')
            n=0
            do i=1,2000
               read(2,*,end=667) x1,x2,x3,x4,x5,x6,x7,x8,x9
               call xlinint(x1,nw,wv,wn,fadc)
               n=n+1
               gn=gn0/fadc/gnw(n)
c               gn2=gn
               gn2=gn0/gnw(n)
               x(n)=x1
               y(n)=x2
               y2(n)=x3
               ye(n)=x8
               ysum(n)=ysum(n)+x2*gn
               ysum2(n)=ysum2(n)+x3*gn
               ysume(n)=ysume(n)+x8*x8*gn
               ysum2e(n)=ysum2e(n)+x9*x9*gn
               if(x2.ne.0) gsum(n)=gsum(n)+gn2*gn2
            enddo
         endif
 667     continue
         close(2)
      enddo
 666  continue
      rewind(1)

c - get the straight sum
      do il=1,1000
         read(1,*,end=676) file1,iflag,gn
         if(iflag.eq.0) then
            open(unit=2,file=file1,status='old')
            n=0
            do i=1,2000
               read(2,*,end=677) x1,x2,x3,x4,x5,x6,x7,x8,x9
               n=n+1
               ysum3(n)=ysum3(n)+x2
               ysum3e(n)=ysum3e(n)+x8*x8
            enddo
 677        continue
            close(2)
         endif
      enddo
 676  continue
      close(1)

      open(unit=11,file='splines.out',status='unknown')
      do i=1,n
         facu=1./gsum(i)*1.3
         xs1=sqrt(ysume(i)*facu)
         xs2=sqrt(ysum2e(i)*facu)
         xs3=sqrt(ysum3e(i))
         write(11,1101) x(i),ysum(i)*facu,ysum2(i)*facu*1.0e17,
     $        xs1,xs2*1.0e17,ysum3(i),xs3,facu
      enddo
      close(11)
 866  continue
 1101 format(1x,f8.2,7(1x,1pe11.2))

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
