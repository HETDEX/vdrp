
      parameter(nmax=2000)
      real w(nmax),xin(nmax),xa(nmax,nmax),xsa(nmax,nmax),xca(nmax,nmax)
      real xin2(nmax),xin3(nmax)
      character file1*50
      
      open(unit=1,file='slist',status='old')

      nt=0
      do i=1,nmax
         read(1,*,end=666) file1
         nt=nt+1
         open(unit=2,file=file1,status='old')
         n=0
         do j=1,nmax
            read(2,*,end=667) x1,x2,x3,x4,x5
            n=n+1
            w(n)=x1
            xa(nt,n)=x2
            xsa(nt,n)=x3
            xca(nt,n)=x5
         enddo
 667     continue
         close(2)
      enddo
 666  continue
      close(1)


      open(unit=11,file='out',status='unknown')
      do j=1,n
         nin=0
         do i=1,nt
            if(xa(i,j).ne.0) then
               nin=nin+1
               xin(nin)=xa(i,j)
               xin2(nin)=xsa(i,j)
               xin3(nin)=xca(i,j)
            endif
         enddo
         if(nin.gt.99) then
            call biwgt(xin,nin,xb,xs)
            call biwgt(xin2,nin,xb2,xs2)
            call biwgt(xin3,nin,xb3,xs3)
            n95=nint(0.95*float(nin))
            write(11,1101) w(j),xs,xin(n95),xs2,xin2(n95),
     $           xs3,xin3(n95)
         else
            write(11,1101) w(j),0.,0.,0.,0.,0.,0.
         endif
      enddo
      close(11)
 1101 format(1x,f8.2,6(1x,f9.3))
      end
