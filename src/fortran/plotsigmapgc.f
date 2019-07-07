
      parameter(nmax=10000)
      real r(nmax),s(nmax),rd(nmax),sd(nmax),sde(nmax)
      real sdl(nmax),sdh(nmax),xml(nmax),rm(nmax)
      real xmll(nmax),xmlh(nmax),xl(2),yl(2)
      character file1*40,title*40

      ilog=0
      title=' '

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      open(unit=1,file='sigma.dat',status='old')
      nd=0
      do i=1,nmax
         read(1,*,end=668) x1,x2,x3,x4
         nd=nd+1
         if(ilog.eq.0) rd(nd)=x1
         if(ilog.eq.1) rd(nd)=log10(x1)
         sd(nd)=x2
         sde(nd)=x2-x3
         sdl(nd)=x3
         sdh(nd)=x4
      enddo
 668  continue
      close(1)

      rmaxp=rd(nd)*1.05
      rmin=0
      ymin=205.
      ymax=415.
      call pgenv(rmin,rmaxp,ymin,ymax,0,0)
      call pglabel('Radius (arcsec)','\gs (km/s)',title)

      call pgpt(nd,rd,sd,17)
      call pgerry(nd,rd,sdh,sdl,1.)

      call pgsls(1)
      call pgsci(2)

      ig=0
      nm=0
      open(unit=1,file='siglist',status='old')
      do i=1,100
         read(1,*,end=666) file1
         open(unit=2,file=file1,status='old')
         n=0
         do j=1,nmax
            read(2,*,end=667) x1,x2,x3
            n=n+1
            if(ilog.eq.0) r(n)=x1
            if(ilog.eq.1) r(n)=log10(x1)
            s(n)=x3
         enddo
 667     continue
         close(2)
         call pgsls(1)
         call pgslw(4)
         call pgline(n,r,s)
         call pgsls(1)
         call pgslw(2)
      enddo
 666  continue
      close(1)
      call pgsci(1)

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

      subroutine plotpoly(n,z,wl,wu,z1,z2,ic)
      parameter(nmax=1000)
      real z(n),wl(n),wu(n)
      real zp(nmax),w(nmax)
      ip=0
      do i=1,n
         if(z(i).ge.z1.and.z(i).le.z2) then
            ip=ip+1
            zp(ip)=z(i)
            w(ip)=wl(i)
         endif
      enddo
      do i=n,1,-1
         if(z(i).ge.z1.and.z(i).le.z2) then
            ip=ip+1
            zp(ip)=z(i)
            w(ip)=wu(i)
         endif
      enddo
      ip=ip+1
      zp(ip)=z(1)
      w(ip)=w(1)
      call pgsci(ic)
      call pgpoly(ip,zp,w)
      call pgsci(1)

      return
      end
