
      real xin(1000)
      character a1*10,a2*5,a10*12,fname*35

      chicut=0.1
      fcut=10.

      open(unit=1,file='runstar',status='old')
      n=0
      do i=1,1000
         read(1,*,end=666) a1,a2,x3,x4,x5,x6,x7,x8,x9,a10,x11,i12,i13
         fname="sfit/"//a10//"_"//a2//"_4550.fw"
         open(unit=2,file=fname,status='old')
         read(2,*,end=333,err=333) y1,y2,y3
         if(y2.gt.chicut.and.y3.gt.fcut) then
            n=n+1
            xin(n)=y1
         endif
 333     continue
         close(2)
      enddo
 666  continue
      rewind(1)
      call biwgt(xin,n,xb,xs)
      open(unit=12,file='fwhm.out',status='unknown')
      write(12,*) xb,xs,n
      close(12)
      print *,xb,xs,n

      open(unit=11,file='out',status='unknown')
      do i=1,1000
         read(1,*,end=667) a1,a2,x3,x4,x5,x6,x7,x8,x9,a10,x11,i12,i13
         fname="sfit/"//a10//"_"//a2//"_4550.res"
         open(unit=2,file=fname,status='old')
         read(2,*,end=334,err=334) y1,y2
         write(11,1101) 
     $        "/work/00115/gebhardt/maverick/scripts/rsp/rsp3f",
c         write(11,1101) 
c     $        "~gebhardt/bin/rsp3",
     $        y1,y2,3,4505,1035,a2,a10,-xb,10,3.5,0,1,2
 334     continue
         close(2)
      enddo
 667  continue
      close(1)

 1101 format(a50,2(1x,f10.6),1x,i1,1x,i4,1x,i4,1x,a5,1x,a12,1x,f5.2,
     $     1x,i2,1x,f3.1,3(1x,i1))

      end
