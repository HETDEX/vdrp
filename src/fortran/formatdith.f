
      real*8 dx1,dx2
      character a3*6,a8*28,a9*17,a10*5

      open(unit=1,file='in',status='old')
      open(unit=11,file='out',status='unknown')

      do i=1,100000
         read(1,*,end=666) dx1,dx2,a3,x4,x5,x6,x7,a8,a9,a10
         write(11,1001)  dx1,dx2,a3,x4,x5,x6,x7,a8,a9,a10
      enddo
 666  continue
      close(1)

 1001 format(f11.7,1x,f11.7,1x,a6,4(1x,f9.3),1x,a28,1x,a17,1x,a5)

      end
