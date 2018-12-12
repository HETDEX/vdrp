
      parameter(nmax=15000)
      real x(nmax),xl(100),yl(100)
      character file1*80,fname*14

      nbin=19
      xmin=-4.
      xmax=4.
      binsize=(xmax-xmin)/float(nbin-1)
      nl=100
      do i=1,nl
         xl(i)=xmin+(xmax-xmin)/float(nl-1)*float(i-1)
      enddo

      open(unit=1,file="list",status='old')

      call pgbegin(0,'?',3,3)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.4)
      call pgslw(2)

      do j=1,1000
         read(1,*,end=667) file1
         fname=file1(8:21)
         open(unit=2,file=file1,status='old')
         n=0
         do i=1,nmax
            read(2,*,end=666) x2
            n=n+1
            x(n)=x2
         enddo
 666     continue
         close(2)

         call biwgt(x,n,xb,xs)
         sig=xs
         amp=float(n)/sqrt(2.*3.14*sig*sig)*binsize
         ymax=0.
         do ig=1,nl
            yl(ig)=amp*exp(-(xl(ig))**2/2./sig/sig)
            ymax=max(ymax,yl(ig))
         enddo

         ymax=ymax*1.2
         call pgsch(1.2)
         call pgenv(xmin,xmax,0.,ymax,0,0)
         call pghist(n,x,xmin,xmax,nbin,1)
         call pgsci(2)
         call pgline(nl,xl,yl)
         call pgsci(1)
         call pgsch(1.7)
         call pgmtxt('T',-1.8,0.1,0.,fname)
         write(fname(1:5),1001) xs
         call pgmtxt('T',-1.8,0.65,0.,fname(1:5))


      enddo
 667  continue
      close(1)

      call pgend
 1001 format(f5.2)
      end
