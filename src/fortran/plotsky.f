
      parameter(nmax=10000)
      real x(nmax),y(nmax),xa(nmax),ya(nmax)
      real sky(nmax,5000),wsky(nmax),xin(nmax)
      real skyb(nmax),skys(nmax)
      integer nsa(5000)
      character file1*80,file2*80

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgsls(1)
      call pgslw(1)

      xmin=3490.
      xmax=5510.
      ymin=0.
      ymax=300.

      nsky=5000
      do i=1,nsky
         wsky(i)=xmin+float(i-1)/float(nsky-1)*(xmax-xmin)
         nsa(i)=0
      enddo

      open(unit=1,file='list',status='old')

      ia=0
      ns1=0
      do il=1,1000
         read(1,*,end=666) file1,file2
         open(unit=2,file=file1,status='old')
         open(unit=3,file=file2,status='old')
         if(il.eq.1) then
            call pgsci(1)
            call pgenv(xmin,xmax,ymin,ymax,0,0)
            call pglabel('Wavelength','Counts','')
         endif
         na=0
         do i=1,10000
            read(3,*,end=667) x1,x2
            na=na+1
            xa(na)=x1
            ya(na)=x2
         enddo
 667     continue
         close(3)
         n=0
         do i=1,10000
            read(2,*,end=668) x1,x2
            n=n+1
            x(n)=x1
            call xlinint(x1,na,xa,ya,yp)
            y(n)=x2/yp
            do j=1,nsky-1
               if(x(n).ge.wsky(j).and.x(n).lt.wsky(j+1)) then
                  nsa(j)=nsa(j)+1
                  sky(j,nsa(j))=y(n)
                  goto 866
               endif
            enddo
 866        continue
         enddo
 668     continue
         close(2)
         ia=ia+1
         if(ia.eq.15) ia=1
c         call pgsci(ia)
c         call pgline(n,x,y)
      enddo
 666  continue
      close(1)

      open(unit=11,file='sky.out',status='unknown')
      do j=1,nsky
         do i=1,nsa(j)
            xin(i)=sky(j,i)
         enddo
         call biwgt(xin,nsa(j),xb,xs)
         skyb(j)=xb
         skys(j)=xs
         if(nsa(j).gt.5) write(11,*) wsky(j),xb,xs
      enddo
      close(11)
      call pgsci(1)
      call pgslw(6)
      call pgline(nsky,wsky,skyb)

      call pgend

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
