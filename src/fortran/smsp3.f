
      parameter(nmax=30000)
      real w(nmax),wd(nmax),sd(nmax),sumsp(nmax)
      real w1(nmax),w2(nmax)
      integer nsum(nmax)
      character file1*80
      
      ws=3500.
      we=5500.
      wbin=50.
      nbin=0
      do i=1,1000
         wnew=ws+float(i-1)*wbin
         if(wnew.lt.we) then
            nbin=nbin+1
            w(nbin)=wnew+wbin/2.
            w1(nbin)=wnew
            w2(nbin)=wnew+wbin
         else
            goto 766
         endif
      enddo
 766  continue

      do i=1,nbin
         sumsp(i)=0.
         nsum(i)=0
      enddo

      open(unit=1,file='list',status='old')
      nf=0
      do i=1,1000
         read(1,*,end=666) file1
         nf=nf+1
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
            do k=1,n
               if(wd(k).gt.w1(j).and.wd(k).le.w2(j)) then
                  sumsp(j)=sumsp(j)+sd(k)
                  nsum(j)=nsum(j)+1
               endif
            enddo
         enddo
      enddo
 666  continue
      close(1)

      open(unit=11,file='sumspec.out',status='new')
      do i=1,nbin
         write(11,*) w(i),sumsp(i)/float(nf)
      enddo
      close(11)

      end
