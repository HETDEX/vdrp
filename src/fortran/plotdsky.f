
      parameter(nmax=200000)
      real sb(nmax),sg(nmax),sr(nmax)
      real bb(nmax),bg(nmax),br(nmax)
      real xi(nmax)
      character c7*10

      open(unit=1,file='in',status='old')
      
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.5)
      call pgslw(2)

      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,x5,x6,c7,x8,x9
         n=n+1
         sb(n)=x1
         sg(n)=x3
         sr(n)=x5
         bb(n)=x2
         bg(n)=x4
         br(n)=x6
         xi(n)=x9
      enddo
 666  continue
      close(1)

      call getbi(n,sb,xbb)
      call getbi(n,sg,xbg)
      call getbi(n,sr,xbr)
      call getbi(n,bb,bbb)
      call getbi(n,bg,bbg)
      call getbi(n,br,bbr)

      xmin=10.
      xmax=110.
      ymin=0.5
      ymax=1.5
      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pgsci(4)
      call pgpt(n,xi,bb,17)
      call pgsci(3)
      call pgpt(n,xi,bg,17)
      call pgsci(2)
      call pgpt(n,xi,br,17)

      call pgend
      end

      subroutine getbi(n,x,xb)
      real x(n),xin(10000)
      
      do i=1,n
         xin(i)=x(i)
      enddo
      call biwgt(xin,n,xb,xs)
      do i=1,n
         x(i)=x(i)/xb
      enddo
      return
      end
