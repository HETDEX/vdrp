
      real*8 dx2,dx3
      character a1*19

      open(unit=1,file='in',status='old')
      open(unit=11,file='out',status='unknown')

      do i=1,100000
         read(1,*,end=666) a1,dx2,dx3,x4,x5,x6,x7,x8,x9
         write(11,1001) a1,dx2,dx3,x4,x5,x6,x7,x8,x9
      enddo
 666  continue
      close(1)

 1001 format(a19,f11.7,1x,f11.7,1x,f7.2,5(1x,f7.2))

      end
