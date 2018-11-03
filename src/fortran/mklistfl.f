
      character name*14
      read *,rcen,dcen
      read *,name
c      rcen=214.593585
c      dcen=52.35663
      cosd=cos(dcen/57.3)
      rstart=rcen-35./3600/cosd
      dstart=dcen-35./3600.
      n=0
      open(unit=11,file='out',status='unknown')
      do i=1,70
         rnew=rstart+float(i-1)/3600./cosd
         do j=1,70
            n=n+1
            dnew=dstart+float(j-1)/3600.
            write(11,1001) "rspfl3f",rnew,dnew,2.5,4505,1035,n,
     $           name,1.7,3.0,-3.5,0,1,1
         enddo
      enddo
      close(11)
 1001 format(a7,2(1x,f10.5),1x,f4.1,3(1x,i6),1x,a14,3(1x,f4.1),3(1x,i2))
      end
