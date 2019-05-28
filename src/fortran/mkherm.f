      
      parameter(nvel=1000,nmax=9000,nsim=100)

      real v(nvel),y(nvel),yn(nsim),yh(nvel),yl(nvel)
      character file1*40

      parameter(pi=3.141593e0)
      data big/1.e20/

 1    write(*,"('File : '$)")
      read *,file1
      open(unit=1,file=file1,status='old',err=1)

      call pgbegin(0,'?',3,3)
      call pgscf(2)
      call pgsch(1.3)
      call pgslw(2)

      idum=-1

      do j=1,nmax
c         read(1,*,end=666) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11
         read(1,*,end=666) x1,x2,x3,x4,x5,x6,x7,x8,x9
         rad=x1
         vel=x2
         ve=x3
         sig=x4
         se=x5
         amp=x6
         ampe=x7
         h3=x6
         h3e=x7
         h4=x8
         h4e=x9
         amp=1.0
         ampe=0.

         if(rad.lt.0.) then
            rad=-rad
            vel=-vel
            h3=-h3
         endif

 1004    format('r',f6.1)
 1003    format('r',f6.2)
 1002    format('r0',f5.2)
 1001    format('r00',f4.2)
c 1003    format('r',f5.2)
c 1002    format('r0',f4.2)
c 1001    format('r00',f3.2)
         if(rad.lt.10.) then
            write(file1,1001) rad
         elseif(rad.ge.10..and.rad.lt.100.) then
            write(file1,1002) rad
         elseif(rad.ge.100..and.rad.lt.1000.) then
            write(file1,1003) rad
         else
            write(file1,1004) rad
         endif

         open(unit=2,file=file1(1:7)//'herm',status='unknown')

c         vmin=-600.
c         vmax=600.
         vmin=-4.*sig
         vmax=4.*sig
         ymax=-big
         vdiff=(vmax-vmin)
         gmax=1./sqrt(2.*pi)/sig
c         econt=gmax/15.
         econt=gmax/30.
         sum=0
         do i=1,nvel
            v(i)=vmin+(i-1)/float(nvel-1)*vdiff
            w=(v(i)-vel)/sig
            y(i)=amp/sqrt(2.*pi)*exp(-w*w/2.)/sig*(
     $           1.+h3*fh3(w)+h4*fh4(w))
            ymax=max(ymax,y(i))
            sum=sum+y(i)
            do isim=1,nsim
               veln=vel+ve*gasdev(idum)
               sign=sig+se*gasdev(idum)
               ampn=amp+ampe*gasdev(idum)
               h3n=h3+h3e*gasdev(idum)
               h4n=h4+h4e*gasdev(idum)
               cont=0.+econt*gasdev(idum)
               w=(v(i)-veln)/sign
               yn(isim)=ampn/sqrt(2.*pi)*exp(-w*w/2.)/sign*(
     $           1.+h3n*fh3(w)+h4n*fh4(w))+cont
c               print *,v(i),yn(isim),cont,ampn
c               read *
            enddo

            call biwgt(yn,nsim,xb,xs)

            y(i)=max(0.,y(i))
            yh(i)=y(i)+xs
            yl(i)=max(0.,y(i)-xs)
               
            write(2,*) v(i),y(i),yl(i),yh(i)
         enddo
         print *,sum
         close(2)

         call pgenv(vmin,vmax,0.,1.05*ymax,0,0)
         call pgline(nvel,v,y)
         call pgerry(nvel,v,yh,yl,1)

      enddo

 666  continue
      close(1)

      call pgend
      
      end

      function fh3(x)
      fh3=1./sqrt(6.)*(2.*sqrt(2.)*x*x*x-3.*sqrt(2.)*x)
      return
      end
      function fh4(x)
      fh4=1./sqrt(24.)*(4.*x*x*x*x-12.*x*x+3.)
      return
      end

