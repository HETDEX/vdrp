
      parameter(nmax=10000)
      real x(nmax),y(nmax),y2(nmax),ysum(nmax),ysum2(nmax)
      real ya1(1000,nmax),ya2(1000,nmax),yin1(nmax),yin2(nmax)
      character file1*80,file2*80,c1*8

      open(unit=1,file='list2',status='old')

      do i=1,nmax
         ysum(i)=0.
         ysum2(i)=0.
      enddo

      ic=0
      nl=0
      nsum=0
      ymaxs=0.
      sumg=0.
      do il=1,1000
         read(1,*,end=666) file1,iflag,gn
         if(iflag.eq.0) then
            nsum=nsum+1
            sumg=sumg+gn
            open(unit=2,file=file1,status='old')
            n=0
            do i=1,2000
               read(2,*,end=667) x1,x2,x3,x4,x5,x6,x7
               n=n+1
               x(n)=x1
c               ysum(n)=ysum(n)+x3*x4*x5*x6*gn/1.e17
               ysum(n)=ysum(n)+x3*gn*1.e17
               ysum2(n)=ysum2(n)+(x7*gn*1.e17)**2
            enddo
         endif
 667     continue
         close(2)
      enddo
 666  continue
      close(1)

c      sumg=sumg/float(nsum)
      open(unit=11,file='sumg.out',status='unknown')
      write(11,*) sumg
      close(11)
      open(unit=11,file='splines.out',status='unknown')
      do i=1,n
c         write(11,*) x(i),ysum(i)/sumg
         imin=max(i-1,1)
         imax=min(imin+3,n)
         ssum1=0.
         ssum2=0.
         do ii=imin,imax
            ssum1=ssum1+ysum(ii)
            ssum2=ssum2+ysum2(ii)
         enddo
         if(ssum2.gt.1e-6) then
            write(11,*) x(i),ssum1/sqrt(ssum2)
         else
            write(11,*) x(i),0.
         endif
      enddo
      close(11)

      end
