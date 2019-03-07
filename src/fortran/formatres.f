
      real*8 dx2,dx3
      character a16*12,a5*110,a1*10

      open(unit=1,file='inres',status='old')
      open(unit=2,file='inres2',status='old')
      open(unit=11,file='outres',status='unknown')

      read(1,*) dx1,dx2,x3,x4,x5,x6,x7,x8,x9,x10,x11,
     $        x12,x13,x14,i15,a16
      read(2,*) a1,j2,j3,j4,a5
      write(11,1001) dx1,dx2,x3,x4,x5,x6,x7,x8,x9,x10,x11,
     $        x12,x13,x14,i15,a16,j2,j3,j4,a5
      close(1)
      close(2)
      close(11)

 1001 format(f11.7,1x,f11.7,12(1x,f7.2),1x,i6,1x,a12,3(1x,i4),1x,a110)

      end
