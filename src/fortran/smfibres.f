
      parameter(nmax=100000)
      real fib(nmax),sig(nmax),w(nmax),f(nmax),xs(nmax),ys(nmax)
      real siga(112),fibi(112),xin(nmax)

      open(unit=1,file='in',status='old')

      n=0
      ymin=1e10
      ymax=0
      
      w0=5461
      wl=w0-2.
      wh=w0+2
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4
         if(x3.gt.wl.and.x3.lt.wh) then
            n=n+1
            fib(n)=x1
            sig(n)=x2
            w(n)=x3
            f(n)=x4
         endif
      enddo
 666  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.4)
      call pgscf(2)
      call pgslw(2)

      xmin=fib(1)
      xmax=fib(n)
      ymin=1.5
      ymax=3.2
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgsch(0.5)
      call pgpt(n,fib,sig,17)

      open(unit=11,file='fibout',status='unknown')
      do i=1,112
         fibi(i)=float(i)
         nin=0
         do j=1,n
            if(nint(fib(j)).eq.i) then
               nin=nin+1
               xin(nin)=sig(j)
            endif
         enddo
         call biwgt(xin,nin,xb,xs)
         siga(i)=xb
         if(siga(i).gt.2.0.and.siga(i).lt.3.5) write(11,1001) i,siga(i)
      enddo
      close(11)

      call pgsci(2)
      call pgslw(5)
      call pgline(112,fibi,siga)

      call pgend
 1001 format(1x,i4,1x,f6.2)
      end
