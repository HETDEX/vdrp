
      character name*12,a1*12,a2*5

      xnd=0.9

      open(unit=1,file=
     $"/work/03946/hetdex/hdr1/calib/fwhm_and_fluxes_better.txt",
     $     status='old')
      read(1,*)

      read *,name

      fwhm=1.8
      open(unit=11,file='normexp.out',status='unknown')
      do i=1,10000
c         read(1,*,end=666) a1,x2,i3,x4,x5,x6,x7,x8,x9
         read(1,*,end=666) a1,a2,x2,x2mf,x2mb,i3,x4,x5,x6,x7,x8,x9
         if(a1.eq.name) then
            if(x2mf.gt.0.8.and.x2mf.lt.5.) then
               fwhm=x2mf
            else
               fwhm=1.7
            endif
c            fwhm=x2
            i1good=0
            i2good=0
            if(abs(x4-1.).lt.xnd.and.abs(x5-1.).lt.xnd.and.
     $           abs(x6-1.).lt.xnd) i1good=1
            if(abs(x7-1.).lt.xnd.and.abs(x8-1.).lt.xnd.and.
     $           abs(x9-1.).lt.xnd) i2good=1
            if(x4.eq.1.0) i1good=0
            if(x7.eq.1.0) i2good=0
            if(i1good.eq.1.and.i2good.eq.1) then
               xn1=(x4+x7)/2.
               xn2=(x5+x8)/2.
               xn3=(x6+x9)/2.
            endif
            if(i1good.eq.1.and.i2good.eq.0) then
               xn1=x4
               xn2=x5
               xn3=x6
            endif
            if(i1good.eq.0.and.i2good.eq.1) then
               xn1=x7
               xn2=x8
               xn3=x9
            endif
            if(i1good.eq.0.and.i2good.eq.0) then
               xn1=1.0
               xn2=1.0
               xn3=1.0
               write(*,1102) "Nothing worked here: ",x4,x5,x6,x7,x8,x9
            endif
            write(11,1101) "exp01",x2,xn1
            write(11,1101) "exp02",x2,xn2
            write(11,1101) "exp03",x2,xn3
            goto 667
         endif
      enddo
 666  continue
      write(11,1101) "exp01",1.8,1.0
      write(11,1101) "exp02",1.8,1.0
      write(11,1101) "exp03",1.8,1.0
 667  continue
      close(1)
      close(11)
      open(unit=11,file='fwhm.use',status='unknown')
      write(11,*) fwhm
      close(11)
 1101 format(a12,2(1x,f7.3))
 1102 format(a20,6(1x,f6.2))

      end
