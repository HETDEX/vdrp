
      parameter(nmax=1000)
      real z(nmax),he(nmax),he1(nmax),he2(nmax),heb(nmax)

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

      call pgenv(0.,3.5,0.,2.2,0,0)
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
      do i=1,nmax
         read(1,*,end=667) x1,x2,x3,ic,it
         n=n+1
         z(n)=x1
         he1(n)=x2
         he2(n)=x3
         call pgsci(ic)
         if(x3.gt.x2) then
c            call pgerry(1,z(n),he1(n),he2(n),1.)
            call pgsch(1.4)
            ybit=(x3-x2)/3.
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

      call pgend
      end
