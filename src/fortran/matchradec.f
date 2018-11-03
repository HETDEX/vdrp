
      parameter(nmax=1000000)
      real w(nmax),xf(nmax),sn(nmax),chi(nmax),sigma(nmax)
      real*8 dra(nmax),ddec(nmax),drad,dx1,dx2
      integer iok(nmax),iok2(nmax)

      parameter(radtodeg=57.29578)

      rad=3.5
      wc=2.5

      open(unit=1,file='allfit.dat',status='old')

      n=0
      do i=1,nmax
         read(1,*,end=666) dx1,dx2,x3,x4,x5,x6
         n=n+1
         dra(n)=dx1
         ddec(n)=dx2
         w(n)=x3
         sn(n)=x4
         chi(n)=x5
         sigma(n)=x6
         iok(n)=1
         iok2(n)=1
      enddo
 666  continue
      close(1)

      cosd=cos(sngl(ddec(1))/radtodeg)

      do i=1,n-1
         if(iok(i).eq.1) then
            imax=i
            snmax=sn(i)
            do j=i+1,n
               drad=dble(cosd)*(dra(i)-dra(j))**2+(ddec(i)-ddec(j))**2
               drad=3600.d0*dsqrt(drad)
               radc=sngl(drad)
               if(radc.lt.rad.and.abs(w(i)-w(j)).lt.wc) then
                  iok(j)=0
                  iok2(j)=0
                  iok2(i)=0
                  if(sn(j).ge.snmax) then
                     imax=j
                     snmax=sn(j)
                     iok(i)=0
                  endif
c                  print *,i,j,sn(i),sn(j),imax,iok(i)
               endif
            enddo
            iok2(imax)=1
         endif
      enddo

      iwrite=0
      do i=1,n
         if(iok2(i).eq.1) iwrite=1
      enddo
      if(iwrite.eq.0) goto 888

      open(unit=11,file='radec.out',status='unknown')
      do i=1,n
         if(iok2(i).eq.1) then
            write(11,1011) dra(i),ddec(i),w(i),sn(i),chi(i),sigma(i)
         endif
      enddo
      close(11)
 888  continue

 1011 format(2(1x,f11.7),4(1x,f8.2))
      end
