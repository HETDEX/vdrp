
      parameter(nmax=200000)
      real xl(2),yl(2),sn(10)
      integer isn1(10),isn2(10)
      character c1*40,c2*80,c3*80,a1*40,fname*12

c      chimax=1.0
      chimax=2.3
      ns=5
      sn(1)=5.0
      sn(2)=6.0
      sn(3)=7.0
      sn(4)=8.0
      sn(5)=10.0
      do i=1,ns
         isn1(i)=0
         isn2(i)=0
      enddo

      read *,fname
      c1="cat.0"
      c2="/work/00115/gebhardt/maverick/detect/hdr1/"//fname//"/cat.0"
      
      open(unit=1,file=c1,status='old',err=666)
      do i=1,nmax
         read(1,*,end=666) a1,x2,x3,x4,x5,x6,x7,x8,x9
         if(x6.lt.chimax) then
            do j=1,ns
               if(x5.gt.sn(j)) isn1(j)=isn1(j)+1
            enddo
         endif
      enddo
 666  continue
      close(1)

      open(unit=1,file=c2,status='old',err=667)
      do i=1,nmax
         read(1,*,end=667) a1,x2,x3,x4,x5,x6,x7,x8,x9
         if(x6.lt.chimax) then
            do j=1,ns
               if(x5.gt.sn(j)) isn2(j)=isn2(j)+1
            enddo
         endif
      enddo
 667  continue
      close(1)

      open(unit=11,file='out',status='unknown')
      do i=1,ns
         write(*,1001) sn(i),
     $        float(isn1(i))/float(isn2(i)),isn1(i),isn2(i)
         write(11,1001) sn(i),
     $        float(isn1(i))/float(isn2(i)),isn1(i),isn2(i)
      enddo
      close(11)
 1001 format(f6.2,1x,f5.2,2(1x,i4))
      end
