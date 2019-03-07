
      character a1*12

      sn1=5.
      chi1=1.2
      sn2=40.
      chi2=4.0

      rad=3.
      wrad=50.

      open(unit=1,file='in',status='old')
      open(unit=11,file='out',status='unknown')

      n=0
      do i=1,10000000
         read(1,*,end=666) a1,x2,x3,x4,x5,x6,x7
         chi0=chi1+(chi2-chi1)/(sn2-sn1)*(x5-sn1)
         if(x6.le.chi0) then
            n=n+1
            write(11,1001) x2,x3,rad,x4,wrad,n,a1,1.7,5.,3.5,0,1,1,x5,x6
         endif
      enddo
 666  continue
      close(1)
      close(11)

 1001 format("rsp3mc",2(1x,f11.7),1x,f4.1,1x,f7.2,1x,f4.1,1x,i6,1x,
     $     a12,3(1x,f4.1),3(1x,i1),2(1x,f5.1))

      end
