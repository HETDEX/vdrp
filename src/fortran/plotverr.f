
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      real xb(nmax),yb(nmax),y2(nmax),y3(nmax)
      character file1*80,file2*80,c1*18

      ibin=13
      ib1=(ibin-1)/2
      xib=float(ibin)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      xmin=3500.
      xmax=5510.
      ymin=7.
      ymax=25.
      call pgsls(1)
      call pgslw(2)
      call pgsci(1)
      call pgsch(1.2)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgsch(1.4)
      call pglabel('Wavelength','Error','')
      call pgslw(3)
      call pgsch(1.)

      open(unit=1,file='in',status='old')

      n=0
      do i=1,nmax
         read(1,*,end=667) x1,x2,x3
         n=n+1
         x(n)=x1
         y(n)=x2
         y2(n)=x3
         xct=x2*x2
         xph=x2
c         xrs=0.3*xct
c         xerr=sqrt(xph*xph+xrs+xrs)
         xrs=0.05*xct
         xerr=(xph+xrs)/1.3
         y3(n)=xerr
      enddo
 667  continue
      close(1)
      nbb=0
      do j=1,n,ibin
         nbb=nbb+1
         istart=max(0,j-ib1)
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
      enddo
      call pgsci(1)
      call pgline(n,x,y)
      call pgsci(2)
      call pgline(n,x,y2)
      call pgsci(4)
      call pgline(n,x,y3)

      call pgend

      end
