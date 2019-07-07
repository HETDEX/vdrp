
      parameter(nmax=10000,nf=100)
      real xa(nf,nmax),ya(nf,nmax)
      character file1*40

      open(unit=1,file='inlist',status='old')

      nt=0
      do i=1,nf
         read(1,*,end=666) file1
         nt=nt+1
         open(unit=2,file=file1,status='old')
         n=0
         do j=1,nmax
            read(2,*,end=667) x1,x2
            n=n+1
            xa(nt,n)=x1
            ya(nt,n)=x2
         enddo
 667     continue
         close(2)
      enddo
 666  continue
      close(1)

      open(unit=11,file='out2',status='unknown')
      do j=1,n
         sum=0.
         do i=1,nt
            sum=sum+ya(i,j)
         enddo
         sum=sum/float(nt)
         write(11,*) xa(1,j),sum
      enddo
      close(11)

      end
