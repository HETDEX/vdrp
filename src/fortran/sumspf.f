
      parameter(nmax=10000)
      real x(nmax),y(nmax),y2(nmax),ysum(nmax),ysum2(nmax)
      real ya1(1000,nmax),ya2(1000,nmax),yin1(nmax),yin2(nmax)
      real ysum3(nmax),gna(nmax)
      character file1*80,file2*80,c1*8

      ylocut=-100.
      yhicut=1000.

      open(unit=1,file='list4',status='old',err=866)

      do i=1,nmax
         ysum(i)=0.
         ysum2(i)=0.
         ysum3(i)=0.
      enddo
      sumg=0
      do ig=1,1000
         read(1,*,end=777) file1,iflag,gn
         ng=ig
         gna(ng)=gn
         sumg=sumg+gn
      enddo
 777  continue
      if(ng.eq.0) goto 866
      rewind(1)
      do i=1,ng
         gna(i)=gna(i)/sumg
      enddo

      ic=0
      nl=0
      nsum=0
      ymaxs=0.
      sumg=0.
      sum2t=0.
      sum3t=0.
      do il=1,1000
         read(1,*,end=666) file1,iflag,gn
         if(gna(il).lt.0.1) iflag=1
         if(iflag.eq.0) then
            nsum=nsum+1
            sumg=sumg+gn
            open(unit=2,file=file1,status='old')
            n=0
            do i=1,2000
               read(2,*,end=667) x1,x2,x3
               n=n+1
               x(n)=x1
               y(n)=x2
               y2(n)=x3
c               if(y(n).gt.ylocut.and.y(n).lt.yhicut) then
                  ysum(n)=ysum(n)+x2*gn
                  ysum2(n)=ysum2(n)+x3*gn
                  ysum3(n)=ysum3(n)+x3
                  sum2t=sum2t+x3*gn
                  sum3t=sum3t+x3
c                  if(n.eq.26) print *,x2,gn,x2*gn
c               endif
               ya1(nsum,n)=x2*gn
               ya2(nsum,n)=x3*gn
            enddo
         endif
 667     continue
         close(2)
      enddo
 666  continue
      close(1)

      if(nsum.eq.0.or.sumg.le.0) then
         fac=0.
      else
c         fac=sum3t/sum2t
         sumg=sumg/float(nsum)
         print *,1./sumg,sum3t,sum2t
         fac=1./sumg
      endif

      open(unit=11,file='sumspf.out',status='unknown')
      do i=1,n
         do j=1,nsum
            yin1(j)=ya1(j,i)
            yin2(j)=ya2(j,i)
         enddo
c         call biwgt(yin1,nsum,xb1,xs1)
c         call biwgt(yin2,nsum,xb2,xs2)
c         xs1=xs1*sqrt(float(nsum))
c         xs2=xs2*sqrt(float(nsum))
         xs1=1
         xs2=1
         write(11,*) x(i),ysum(i)*fac,ysum2(i)*fac,xs1,xs2
      enddo
      close(11)
 866  continue

      end
