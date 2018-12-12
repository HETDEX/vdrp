
      parameter(nmax=1000)
      real z(nmax),he(nmax),he1(nmax),he2(nmax)

      open(unit=1,file='herr.dat',status='old')
      read(1,*)
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4
         n=n+1
         z(n)=x1
         he(n)=x4
      enddo
 666  continue
      close(1)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.4)
      call pgslw(2)
      call pgscf(2)

      call pgenv(1.,6.5,0.,5.2,0,0)
      call pgline(n,z,he)

      call pglabel('z','percent accuracy on H','')
      call pgslw(4)

      open(unit=1,file='ciphi',status='old')
      read(1,*)
      n=0
      do i=1,nmax
         read(1,*,end=667) x1,x2,x3
         n=n+1
         z(n)=x1
         he1(n)=x2
         he2(n)=x3
      enddo
 667  continue
      close(1)

      call pgsci(4)
      call pgpt(n,z,he1,17)
      call pgline(n,z,he1)
      call pgpt(n,z,he2,17)
      call pgline(n,z,he2)

      open(unit=1,file='ciplow',status='old')
      read(1,*)
      n=0
      do i=1,nmax
         read(1,*,end=669) x1,x2,x3
         n=n+1
         z(n)=x1
         he1(n)=x2
         he2(n)=x3
      enddo
 669  continue
      close(1)

      call pgsci(3)
      call pgpt(n,z,he1,17)
      call pgline(n,z,he1)
      call pgpt(n,z,he2,17)
      call pgline(n,z,he2)

      open(unit=1,file='adept1',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=668) x1,x2,x3
         n=n+1
         z(n)=x1
         he1(n)=x2
         he2(n)=x3
      enddo
 668  continue
      close(1)

      call pgsci(2)
      call pgpt(n,z,he1,17)
      call pgline(n,z,he1)
      call pgpt(n,z,he2,17)
      call pgline(n,z,he2)

      call pgend
      end
