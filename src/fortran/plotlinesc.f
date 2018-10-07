
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      real xb(nmax),yb(nmax),yin(nmax)
      character file1*80,file2*80,c1*18

      ibin=3
      ib1=(ibin-1)/2
      xib=float(ibin)

      call pgbegin(0,'?',3,3)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=3500.
      xmax=5500.
      ymin=-1.5
      ymax=1.5
      call pgsls(1)
      call pgslw(1)

      open(unit=1,file='list',status='old')

      nl=0
      do il=1,10000
         read(1,*,end=666) file1
         open(unit=2,file=file1,status='old')
         ymin=1e10
         ymax=-1e10
         n=0
         do i=1,nmax
            read(2,*,end=667) x1,x3,x2
            if(x2.ge.-1e10.or.x2.le.1e10) then
               n=n+1
               x(n)=x1
               y(n)=x2
               yin(n)=y(n)
               ymin=min(ymin,y(n))
               ymax=max(ymax,y(n))
            else
               goto 866
            endif
         enddo
 667     continue
         close(2)
         nbb=0
         ymin=1e10
         ymax=-1e10
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
            ymin=min(ymin,yn(nbb))
            ymax=max(ymax,yn(nbb))
         enddo
c         c1=file1(1:17)
         c1=file1(1:7)
         ymin=max(ymin,-50.)
c         call biwgt(yin,nb,xbb,xbs)
c         call biwgt(yin,10,xbb,xbs)
c         ymin=xbb
c         ymax=xbb+4.*xbs
         call pgsci(1)
         call pgsch(1.)
         call pgenv(xmin,xmax,ymin,ymax,0,0)
         call pgsch(1.)
c         call pgline(n,x,y)
         call pgline(nbb,xn,yn)
         call pgsci(1)
         call pgsch(2.5)
         call pgmtxt('T',0.7,0.5,0.5,c1)
         call pgsci(1)
 866     continue
         close(2)
      enddo
 666  continue
      close(1)

      call pgend

      end
