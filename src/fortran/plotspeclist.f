
      parameter(nmax=10000)
      real x(nmax),y(nmax),xnorm(nmax),xin(nmax)
      real xs(nmax),ys(nmax),ya(nmax)
      character file1*80,file2*80,c1*8

      x80=0.85

      irange=0
      open(unit=1,file='sprange.dat',status='old',err=555)
      read(1,*) x1,x2
      xw1=x1-x2
      xw2=x1+x2
      irange=1
 555  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      ymin=1e10
      ymax=-1e10
      open(unit=1,file='fitghsp.in',status='unknown')
      ns=0
      do i=1,nmax
         read(1,*,end=777) x1,x2
         ns=ns+1
         xs(ns)=x1
         ys(ns)=x2
         if(x1.gt.xw1.and.x1.lt.xw2) then
            ymin=min(ymin,ys(ns))
            ymax=max(ymax,ys(ns))
         endif
      enddo
 777  continue
      close(1)

      open(unit=1,file='list2',status='old')

      nt=0
      do il=1,1000
         read(1,*,end=668) file1,x2,x3n
         open(unit=2,file=file1,status='old')
         ymint=1e10
         do i=1,2000
            read(2,*,end=669) x1,x2,x3
            if(x1.gt.xw1.and.x1.lt.xw2) then
               ymint=min(ymint,x3*1.e17)
            endif
         enddo
 669     continue
         close(2)
         nt=nt+1
         ya(nt)=ymint
         xnorm(nt)=x3n
         xin(nt)=x3n
      enddo
 668  continue
      rewind(1)

      call biwgt(xin,nt,xb,xs)
      xcut=xin(nint(x80*float(nt)))
      call biwgt(ya,nt,xb,xs)
      ymin=min(ymin,xb)

      ybit=(ymax-ymin)/20.
      ymin=ymin-ybit
      ymax=ymax+ybit
      call pgsci(1)
      xmin=xw1
      xmax=xw2
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel('Wavelength','Flux (1e-17 ergs/cm\U2\D/s)','')
      
      ic=0
      do il=1,1000
         read(1,*,end=666) file1,x2,x3
         if(x3.gt.xcut) then
            open(unit=2,file=file1,status='old')
            n=0
            do i=1,2000
               read(2,*,end=667) x1,x2,x3
               n=n+1
               x(n)=x1
               y(n)=x3*1.e17
            enddo
 667        continue
            close(2)
            ic=ic+1
            call pgsci(ic)
            call pgline(n,x,y)
            if(ic.eq.12) ic=1
         endif
      enddo
 666  continue
      close(1)

      call pgsci(1)
      call pgslw(5)
      call pgline(ns,xs,ys)

      call pgend

      end
