
      character name*14
      read *,rcen,dcen
      read *,name
      xstep=2.0
      ystep=xstep
      side=70.
      irange=nint(side/xstep)
      jrange=irange

      cosd=cos(dcen/57.3)
      rstart=rcen-side/2./3600/cosd
      dstart=dcen-side/2./3600.
      n=0
      open(unit=11,file='out',status='unknown')
      do i=1,irange
         xoff=float(i-1)*xstep
c         rnew=rstart+float(i-1)/3600./cosd
         rnew=rstart+xoff/3600./cosd
         do j=1,jrange
            n=n+1
            yoff=float(j-1)*ystep
c            dnew=dstart+float(j-1)/3600.
            dnew=dstart+yoff/3600.
            write(11,1001) "rspfl3f",rnew,dnew,3.0,4505,1035,n,
     $           name,1.7,3.0,-3.5,0,1,1
         enddo
      enddo
      close(11)
 1001 format(a7,2(1x,f10.5),1x,f4.1,3(1x,i6),1x,a14,3(1x,f4.1),3(1x,i2))
      end
