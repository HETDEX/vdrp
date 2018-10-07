
c - read in RA,DEC, bin size, and N bins for each dimension
      read *,ra,dec,step,n

      atot=step*float(n-1)
      ahalf=atot/2.

      cdec=cos(dec/57.29)
      ras=ra-ahalf/cdec/3600.
      decs=dec-ahalf/3600.

      open(unit=11,file='out',status='unknown')
      do i=1,n
         r=ras+float(i-1)*step/cdec/3600.
         do j=1,n
            d=decs+float(j-1)*step/3600.
            write(11,1101) r,d
         enddo
      enddo
 1101 format(2(1x,f11.6))

      end

      
