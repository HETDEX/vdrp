
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      real xb(nmax),yb(nmax)
      character file1*80,file2*80,c1*18

      nf=18
      ibin=5
c      ibin=1
      ib1=(ibin-1)/2
      xib=float(ibin)

c      call pgbegin(0,'?',3,3)
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      xmin=3810.
      xmax=3870.
      xmin=3500.
      xmax=5500.

      open(unit=1,file='splist',status='old')

      nl=0
      ic=0
      do il=1,10000
c         read(1,*,end=666) file1
         read(1,*,end=666) file1,icol,ils,ilw
         xnorm=1.
         open(unit=2,file=file1,status='old')
         ymin=1e10
         ymax=-ymin
         n=0
         do i=1,nmax
            read(2,*,end=667) x1,x2
c            read(2,*,end=667) x1,x2,x3,x4,x5,x6
c            x2=x6
            if(x2.ne.0) then
               n=n+1
               x(n)=x1
               y(n)=x2*xnorm
               ymin=min(ymin,y(n))
               ymax=max(ymax,y(n))
            endif
         enddo
 667     continue
         close(2)
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
         c1=file1(1:17)
         icp=1
         call pgsls(1)
         call pgslw(2)
         call pgsci(1)
         call pgsch(1.8)
         ybit=(ymax-ymin)/10.
         ymin=ymin-ybit
         ymax=ymax+ybit
         ymin=3.
         ymax=12.
         if(il.eq.1) call pgenv(xmin,xmax,ymin,ymax,0,0)
         call pgsci(icol)
         call pgsls(ils)
         call pgslw(ilw)
c         call pgline(n,x,y)
         call pgline(nbb,xn,yn)
         call pgsch(1.8)
         if(il.eq.1) call pglabel('Wavelength',
     $        '1e-17 ergs/cm\U2\D/s','')
c         if(il.eq.1) call pglabel('Wavelength',
c     $        'Counts','')
c         call pgmtxt('T',0.9,0.5,0.5,file1(1:nf))
         call pgsch(1.5)
         call pgsci(1)
 866     continue
         close(2)
      enddo
 666  continue
      close(1)

      call pgend

      end
