
      parameter(nmax=200000)
      real x(nmax),y(nmax)
      integer iifu(nmax),nif(nmax),nifo(nmax),ispec(nmax)
      character c7*10

      open(unit=1,file='ifulist',status='old')
      nifu=0
      do i=1,nmax
         read(1,*,end=666) i1,x2,x3,i4
         nifu=nifu+1
         iifu(i)=i1
         ispec(i)=i4
         nif(i)=0
         nifo(i)=0
      enddo
 666  continue
      close(1)

      open(unit=1,file='in',status='old')
      
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.5)
      call pgslw(2)

      n=0
      ntot=0
      do i=1,nmax
         read(1,*,end=667) i1,i2,i3
         do j=1,nifu
            if(i1.eq.iifu(j)) then
               nif(j)=nif(j)+i2
               nifo(j)=nifo(j)+1
               ntot=max(ntot,nifo(j))
            endif
         enddo
      enddo
 667  continue
      close(1)

      sum=0
      open(unit=11,file='out',status='unknown')
      do i=1,nifu
         x(i)=float(iifu(i))
         y(i)=float(nif(i))/float(ntot)
         write(11,*) iifu(i),y(i)
         sum=sum+y(i)
      enddo
      close(11)
      print *,"Nobs,Sum= ",Ntot,sum/float(nifu)

      call pgenv(10.,110.,0.,12.,0,0)
      call pglabel("IFUslot","N/obs","")
      do i=1,nifu
         if(ispec(i).lt.100) call pgsci(1)
         if(ispec(i).gt.100) call pgsci(2)
         call pgpt1(x(i),y(i),17)
      enddo

      call pgend
      end
