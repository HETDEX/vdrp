
      parameter(nmax=10000)
      real x(nmax),y(nmax),w(nmax),wca(nmax),flux(nmax)
      real xa(nmax),ya(nmax),frac(nmax),fluxo(nmax),sum(nmax)
      integer ichi(nmax),icho(nmax)

      open(unit=1,file='in',status='old')
      
      r0=0.1
      w0=0.1

      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,x5
         n=n+1
         x(n)=x1
         y(n)=x2
         w(n)=x3
         flux(n)=x4
         frac(n)=x5
         ichi(n)=1
      enddo
 666  continue
      close(1)

      do i=1,n-1
         sum(i)=frac(i)
         if(ichi(i).eq.1) then
            do j=i+1,n
               rad=sqrt((x(i)-x(j))**2+(y(i)-y(j))**2)
               if(rad.lt.r0.and.abs(w(i)-w(j)).lt.w0) then
                  ichi(j)=0
                  sum(i)=sum(i)+frac(j)
c                  if(frac(j).gt.frac(i)) then
c                     ichi(j)=1
c                     ichi(i)=0
c                  endif
               endif
            enddo
         endif
      enddo
 667  continue
      open(unit=11,file='in2',status='unknown')
      do i=1,n
c         if(ichi(i).eq.1) write(11,1101) x(i),y(i),w(i),flux(i),
c     $        sum(i),frac(i)
         if(ichi(i).eq.1) write(11,1101) x(i),y(i),w(i),flux(i),
     $        sum(i)
      enddo
      close(11)
 1101 format(2(1x,f8.3),1x,f9.2,1x,f13.2,1x,f6.2)

      end
