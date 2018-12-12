
      parameter(nmax=1000)
      real z(nmax),he(nmax),he1(nmax),he2(nmax),heb(nmax)
      character lab*10

      open(unit=1,file='herr.dat',status='old')
      read(1,*)
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,x5
         n=n+1
         z(n)=x1
         he(n)=x4
         heb(n)=x5
      enddo
 666  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.4)
      call pgslw(3)
      call pgscf(2)

      call pgenv(0.,3.5,0.,2.4,0,0)
      call pgslw(4)
      call pgline(n,z,he)
      call pgsls(4)
      call pgline(n,z,heb)
      call pgsls(1)

      call pglabel('z','percent accuracy on R','')
      call pgslw(4)

      open(unit=1,file='hetdex.h',status='old')
      read(1,*)
      n=0
      call pgsch(2.5)
      call pgsci(4)
      ilc=0
      do i=1,nmax
         read(1,*,end=667) x1,x2,x3,ic,it,ilab,lab
         n=n+1
         z(n)=x1
         he1(n)=x2
         he2(n)=x3
         call pgsci(ic)
         if(ilab.eq.1) then
            ilc=ilc+1
            yp=2.3-0.1*ilc
            call pgsch(1.1)
            call pgptxt(2.65,yp,0.,0.,lab)
            call pgsch(2.5)
         endif
         if(x3.gt.x2) then
c            call pgerry(1,z(n),he1(n),he2(n),1.)
            call pgsch(1.4)
c            ybit=(x3-x2)/3.
            ybit=(x3-x2)/5.
            call pgarro(z(n),he1(n),z(n),he1(n)+ybit)
            call pgerr1(2,z(n),he1(n),0.,3.0)
            call pgarro(z(n),he2(n),z(n),he2(n)-ybit)
            call pgerr1(2,z(n),he2(n),0.,3.0)
            call pgsch(2.5)
         else
            call pgpt1(z(n),he1(n),it)
         endif
      enddo
 667  continue
      close(1)
      
      fac=sqrt(1./5.)
      open(unit=1,file='cosmic_variance_distance.dat',status='old')
      read(1,*)
      n=0
      do i=1,nmax
         read(1,*,end=668) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,
     $        x13,x14,x15,x16,x17
         n=n+1
         z(n)=(x1+x2)/2.
         he(n)=x14*fac
      enddo
 668  continue
      close(1)    
      call pgsci(1)
      call pgsls(2)
      call pgline(n,z,he)

      call pgend
      end
