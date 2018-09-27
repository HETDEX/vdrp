
      parameter(nmax=10000)
      real x(nmax),y(nmax),y2(nmax),ysum(nmax,9)
      character file1*80,file2*80,c1*8

      ylocut=-100.
      yhicut=1000.

      open(unit=1,file='list3',status='old',err=866)

      do i=1,nmax
         do j=1,9
            ysum(i,j)=0.
         enddo
      enddo

      ic=0
      nl=0
      nsum=0
      ymaxs=0.
      sumg=0.
      do il=1,1000
         read(1,*,end=666) file1,iflag,gn
         if(iflag.eq.0) then
            nsum=nsum+1
            sumg=sumg+gn
            open(unit=2,file=file1,status='old')
            n=0
            do i=1,2000
               read(2,*,end=667) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10
               n=n+1
               x(n)=x1
               ysum(n,1)=ysum(n,1)+x2*gn
               ysum(n,2)=ysum(n,2)+x3*gn
               ysum(n,3)=ysum(n,3)+x4*gn
               ysum(n,4)=ysum(n,4)+x5*gn
               ysum(n,5)=ysum(n,5)+x6*gn
               ysum(n,6)=ysum(n,6)+x7*gn
               ysum(n,7)=ysum(n,7)+x8*gn
               ysum(n,8)=ysum(n,8)+x9*gn
               ysum(n,9)=ysum(n,9)+x10*gn
            enddo
         endif
 667     continue
         close(2)
      enddo
 666  continue
      close(1)

      sumg=sumg/float(nsum)
      open(unit=11,file='2dsp.out',status='unknown')
      do i=1,n
         x1=x(i)
         x2=ysum(i,1)/sumg
         x3=ysum(i,2)/sumg
         x4=ysum(i,3)/sumg
         x5=ysum(i,4)/sumg
         x6=ysum(i,5)/sumg
         x7=ysum(i,6)/sumg
         x8=ysum(i,7)/sumg
         x9=ysum(i,8)/sumg
         x10=ysum(i,9)/sumg
         write(11,1101) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10
      enddo
      close(11)
 866  continue
 1101 format(f8.2,9(1x,f9.2))

      end
