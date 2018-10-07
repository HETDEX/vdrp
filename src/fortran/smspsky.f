
      parameter(nmax=100000000,nmax2=100000)
      real x(nmax),y(nmax),xs(nmax2),ys(nmax2),y3(nmax2)
      real yn(nmax2),yb(nmax2),xn(nmax2),xb(nmax2)

      open(unit=1,file='in',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 666  continue
      close(1)

      open(unit=11,file='out',status='unknown')

      ibin=301
      ib1=(ibin-1)/2
      xib=float(ibin)
      nbb=0
      do j=1,n,ibin
         nbb=nbb+1
         istart=max(1,j-ib1)
         iend=istart+ibin-1
         if(iend.gt.n) then
            iend=n
            istart=n-ibin+1
         endif
         sum=0.
         nb=0
         do is=istart,iend
            sum=sum+y(is)
            nb=nb+1
            yb(nb)=y(is)
            xb(nb)=x(is)
         enddo
         call biwgt(yb,nb,xbb,xsb)
         yn(nbb)=xbb
         call biwgt(xb,nb,xbb,xsb)
         xn(nbb)=xbb
         write(11,*) xn(nbb),yn(nbb)
      enddo

      end
