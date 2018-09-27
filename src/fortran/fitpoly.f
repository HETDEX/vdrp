
      parameter(nmax=10000)
      real x(nmax),y(nmax),xp(nmax),yp(nmax),c(5)
      real c1a(nmax),c2a(nmax),c3a(nmax),c4a(nmax)
      real c0a(nmax)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      xmin=3500.
      xmax=5500.
      ymin=0.
      ymax=0.2
      call pgsls(1)
      call pgslw(1)

      call pgsci(1)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Wavelength','Throughput','')

      open(unit=1,file='in',status='old')

      n=0
      do i=1,2000
         read(1,*,end=667) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 667  continue
      close(2)
      call pgline(n,x,y)

      ip=4
      np=100
      do i=1,np
         xp(i)=xmin+(xmax-xmin)/float(np-1)*float(i-1)
      enddo

      c0min=4000.
      c0max=5300.
      n0step=30
      do i=1,n0step
         c0a(i)=c0min+(c0max-c0min)/float(n0step-1)*float(i-1)
      enddo
      c1min=0.005
      c1max=0.19
      n1step=200
      do i=1,n1step
         c1a(i)=c1min+(c1max-c1min)/float(n1step-1)*float(i-1)
      enddo
      c2min=-1e-4
      c2max=1e-4
      n2step=20
      do i=1,n2step
         c2a(i)=c2min+(c2max-c2min)/float(n2step-1)*float(i-1)
      enddo
      c3min=-1e-8
      c3max=-1e-7
      n3step=25
      do i=1,n3step
         c3a(i)=c3min+(c3max-c3min)/float(n3step-1)*float(i-1)
      enddo
      c4min=-3e-12
      c4max=3e-12
      n4step=20
      do i=1,n4step
         c4a(i)=c4min+(c4max-c4min)/float(n4step-1)*float(i-1)
      enddo

      smin=1e10
c      do ic0=1,n0step
      do ic1=1,n1step
      do ic2=1,n2step
      do ic3=1,n3step
      do ic4=1,n4step
      c0v=c0a(ic0)
      c0v=4500.
      c1v=c1a(ic1)
      c2v=c2a(ic2)
      c3v=c3a(ic3)
      c4v=c4a(ic4)

      do i=1,np
         xv=xp(i)-c0v
         yp(i)=c1v+c2v*xv+c3v*xv**2+c4v*xv**3
      enddo
      sum=0.
      do i=1,n
         if(x(i).lt.5450.) then
            call xlinint(x(i),np,xp,yp,yf)
            sum=sum+(y(i)-yf)**2
         endif
      enddo
      if(sum.lt.smin) then
         smin=sum
         c0fit=c0v
         c1fit=c1v
         c2fit=c2v
         c3fit=c3v
         c4fit=c4v
      endif

c      enddo
      enddo
      enddo
      enddo
      enddo
      open(unit=11,file='out2',status='unknown')
      write(11,*) sum,c0fit,c1fit,c2fit,c3fit,c4fit
      close(11)
      print *,sum,c0fit,c1fit,c2fit,c3fit,c4fit
      xfmin=0.0005
      do i=1,np
         xv=xp(i)-c0fit
         yp(i)=c1fit+c2fit*xv+c3fit*xv**2+c4fit*xv**3
         yp(i)=max(yp(i),xfmin)
      enddo
      call pgsci(2)
      call pgline(np,xp,yp)
      open(unit=11,file='out',status='unknown')
      do i=1,n
         call xlinint(x(i),np,xp,yp,yf)
         write(11,1101) x(i),yf,0.,0.
      enddo
      close(11)

      call pgend
 1101 format(1x,f6.1,3(1x,f7.4))

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
