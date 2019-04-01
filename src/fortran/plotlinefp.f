
      parameter(nmax=1000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      real xb(nmax),yb(nmax)
      integer is1(nmax),is2(nmax)
      character file1*80,file2*80,c1*18

      factor=0.8
      ibin=1
      ib1=(ibin-1)/2
      xib=float(ibin)

      do i=1,nmax
         is1(i)=0
         is2(i)=0
      enddo

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(3)

      xmin=4.9
      xmax=10.
      ymin=0.
      ymax=0.2
      call pgsls(1)
      call pgsci(1)
      call pgsch(1.4)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgsch(1.4)
      call pglabel('S/N Limit','False positive rate','')
      call pgslw(5)
      call pgsch(1.)

      open(unit=1,file='list',status='old')

      nl=0
      ic=1
      do il=1,10000
         read(1,*,end=666) file1,ils
         open(unit=2,file=file1,status='old')
         n=0
         do i=1,nmax
            read(2,*,end=667) x1,x2,i3,i4
            n=n+1
            x(n)=x1
            y(n)=x2*factor
            if(ils.eq.1) then
               is1(n)=is1(n)+i3*factor
               is2(n)=is2(n)+i4
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
         ic=ic+1
         icp=int(float(ic)/4.)+1
         if(icp.eq.14) ic=1
         if(ic.eq.14) ic=1
         call pgsci(ic)
         call pgsls(ils)
c         call pgline(n,x,y)
         call pgline(nbb,xn,yn)
         call pgsls(1)
 866     continue
         close(2)
      enddo
 666  continue
      close(1)

      do i=1,5
         y(i)=float(is1(i))/float(is2(i))
         print *,x(i),y(i),is1(i),is2(i)
      enddo

      call pgsci(1)
      call pgslw(23)
      call pgline(5,x,y)

      call pgend

      end
