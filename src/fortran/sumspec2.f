
      parameter(nmax=10000)
      real w(nmax),wd(nmax),sd(nmax),sumsp(nmax)
      real w1(nmax),w2(nmax)
      integer nsum(nmax)
      character file1*80
      
      ws=3490.
      we=5510.
      wbin=2.
      nbin=0
      do i=1,2000
         wnew=ws+float(i-1)*wbin
         if(wnew.lt.we) then
            nbin=nbin+1
            w(nbin)=wnew+wbin/2.
            w1(nbin)=wnew
            w2(nbin)=wnew+wbin
c            print *,nbin,w(nbin),w1(nbin),w2(nbin)
         else
            goto 766
         endif
      enddo
 766  continue

      do i=1,nbin
         sumsp(i)=0.
      enddo

      open(unit=1,file='list',status='old')
      do i=1,1000
         read(1,*,end=666) file1
         open(unit=2,file=file1,status='old')
         n=0
         do j=1,nmax
            read(2,*,end=667) x1,x2
            n=n+1
            wd(n)=x1
            sd(n)=x2
         enddo
 667     continue
         close(2)
         do j=1,nbin
            call xlinint(w(j),n,wd,sd,snew)
            sumsp(j)=sumsp(j)+snew
         enddo
c         do j=1,nbin
c            nsum(j)=0
c            do k=1,n
c               if(wd(k).gt.w1(j).and.wd(k).le.w2(j)) then
c                  sumsp(j)=sumsp(j)+sd(k)
c                  nsum(j)=nsum(j)+1
c               endif
c            enddo
c         enddo
      enddo
 666  continue
      close(1)

      open(unit=11,file='sumspec.out',status='new')
      do i=1,nbin
c         if(nsum(i).gt.0) then
c            write(11,*) w(i),sumsp(i)/float(nsum(i))
c         endif
            write(11,*) w(i),sumsp(i)
      enddo
      close(11)

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
