
      real xin(100000)

      open(unit=1,file='in',status='old')
      
      w1=22500.
      w2=22900.

      nb=0
      do i=1,100000
         read(1,*,end=666) x1,x2
         if(x1.gt.w1.and.x1.lt.w2) then
            nb=nb+1
            xin(nb)=x2
         endif
      enddo
 666  continue
      call biwgt(xin,nb,xb,xs)
      rewind(1)

      open(unit=11,file='out',status='unknown')

      do i=1,100000
         read(1,*,end=667) x1,x2
         write(11,*) x1,x2/xb
      enddo
 667  continue
      close(1)
      close(11)

      end
