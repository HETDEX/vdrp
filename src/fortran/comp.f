
      parameter(nmax=10000)
      real x(nmax),y(nmax),w(nmax),wca(nmax),flux(nmax)
      real xa(nmax),ya(nmax),frac(nmax),fluxo(nmax),rada(nmax)
      integer ichi(nmax),icho(nmax)

      open(unit=1,file='in',status='old')
      open(unit=2,file='out',status='old')
      
      r0=4.
      w0=3.0

      n=0
      do i=1,nmax
         read(2,*,end=666) i1,i2,x3,x4,x5,x6,x7
         n=n+1
         x(n)=x3
         y(n)=x4
         w(n)=x5
         fluxo(n)=x7
         icho(n)=0
      enddo
 666  continue
      close(2)

c      open(unit=11,file='comp.out',status='unknown')
      sumx=0.
      sumy=0.
      sumw=0.
      nsum=0
      ni=0
      do i=1,10000
         read(1,*,end=667) x1,x2,x3,x4,x5
         ni=ni+1
         wca(ni)=x3
         flux(ni)=x4
         xa(ni)=x1
         ya(ni)=x2
         frac(ni)=x5
         ichi(ni)=0
         do j=1,n
            rad=sqrt((xa(ni)-x(j))**2+(ya(ni)-y(j))**2)
            if(rad.lt.r0.and.abs(wca(ni)-w(j)).lt.w0) then
               ichi(ni)=1
               icho(j)=1
               rada(ni)=rad
               sumx=sumx+xa(ni)-x(j)
               sumy=sumy+ya(ni)-y(j)
               sumw=sumw+wca(ni)-w(j)
               nsum=nsum+1
c               print *,j,flux(ni),fluxo(j),flux(ni)/fluxo(j)
            endif
         enddo
      enddo
 667  continue
      write(11,*) sumx/float(nsum),sumy/float(nsum),sumw/float(nsum)
      close(1)
      close(11)

      open(unit=11,file='found',status='unknown')
      open(unit=12,file='nfound',status='unknown')
      open(unit=13,file='ofound',status='unknown')
      nt=0
      do i=1,ni
         if(ichi(i).eq.0) then
            nt=nt+1
            write(12,1002) i,xa(i),ya(i),wca(i),flux(i),frac(i),0.
         else
            write(11,1002) i,xa(i),ya(i),wca(i),flux(i),frac(i),rada(i)
         endif
      enddo
c      print *,"Total Input not found: ",nt
      close(11)
      close(12)
      
      nt=0
      do i=1,n
         if(icho(i).eq.0) then
            nt=nt+1
            write(13,1002) i,x(i),y(i),w(i),fluxo(i),1.0,0.
         endif
      enddo
      close(13)
 1001 format("Input not found: ",i3,4(1x,f8.2))
 1002 format(i3,6(1x,f8.2))
      end
