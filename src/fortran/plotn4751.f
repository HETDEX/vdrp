
      parameter(nmax=10000)
      real r(nmax),s(nmax),spsf(nmax),rg(nmax),sg(nmax)
      character file1*40
      parameter(gee=4.3e-3)
      common/cpsf/see

      ipsf=1
      see=0.27
      see=0.32
      see=see/2.35
      rbox=0.22/2.

      open(unit=1,file='n4751-obsvel-majoraxis.out',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3
         n=n+1
         r(n)=abs(x1)
         s(n)=abs(x3)
      enddo
 666  continue
      close(1)

      open(unit=1,file='sbgas.dat',status='old')
      ng=0
      do i=1,nmax
         read(1,*,end=669) x1,x2
         ng=ng+1
         rg(ng)=x1
         sg(ng)=x2
      enddo
 669  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.5)
      call pgslw(2)

      call pgenv(0.,7.,390.,720.,0,0)
      call pglabel("Radius","Vcirc","")

      call pgpt(n,r,s,17)
      call pgslw(4)

c      open(unit=1,file='siglist',status='old')
      open(unit=1,file='masslist',status='old')
      do ia=1,100
         read(1,*,end=667) file1,il
         open(unit=2,file=file1,status='old')
         n=0
         do j=1,nmax
c            read(2,*,end=668) x1,x2,x3
            read(2,*,end=668) x1,x2
            n=n+1
            r(n)=x1
c            s(n)=x3
            rpc=x1*127.
            s(n)=sqrt(gee*x2/rpc)
         enddo
 668     continue
         close(2)

c - get the PSF-corrected sigma
         if(ipsf.eq.1) then
            npsf=500
            rpsf=1.
c            area=(2.*rpsf/float(npsf))**2
            area=(2.*rpsf/float(npsf))**1
            do i=1,n
               rp=r(i)
               xbmin=rp-rpsf
               xbmax=rp+rpsf
               xbstep=(xbmax-xbmin)/float(npsf)
               sum2=0.
               sum1=0.
               do xb=xbmin,xbmax,xbstep
                  rad=xb-rp
                  xbin=xb
                  if(xb.lt.0) xbin=-xb
                  call xlinint(xbin,n,r,s,sv)
                  call xlinint(xbin,ng,rg,sg,sgasv)
                  if(xb.lt.0) sv=-sv
                  call psf(rad,psfv)
c                  psfv=0.
c                  if(abs(rad).lt.rbox) psfv=1.
                  sum1=sum1+area*psfv*sgasv
c                  sum2=sum2+area*psfv*sv*sv
                  sum2=sum2+area*psfv*sv*sgasv
               enddo
c               spsf(i)=sqrt(sum2/sum1)
               spsf(i)=(sum2/sum1)
            enddo
         endif
         call pgsci(il)
         call pgline(n,r,spsf)
c         call pgline(n,r,s)
      enddo
 667  continue
      close(1)

      call pgend
      end

      subroutine psf(r,v)
      parameter(pi=3.141593e0)
      common/cpsf/see

c -- Gaussian PSF

      gc=0.5/pi/see/see
      gden=2.*see*see
      v=gc*exp(-r*r/gden)

      return
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
