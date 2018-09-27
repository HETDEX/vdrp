
      parameter(nmax=100000)
      real w(nmax),s(nmax),wf(nmax),sf(nmax),wlo(nmax),whi(nmax)

      nwl=0
      open(unit=1,file='wvlist',status='old',err=555)
      do i=1,nmax
         read(1,*,end=668) x1,x2
         nwl=nwl+1
         wlo(nwl)=x1
         whi(nwl)=x2
      enddo
 668  continue
 555  close(1)

      open(unit=1,file='skymod',status='old')
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         w(n)=x1
         s(n)=x2
      enddo
 666  continue
      close(1)

      open(unit=1,file='in',status='old')
      nf=0
      do i=1,nmax
         read(1,*,end=667) x1,x2
         nf=nf+1
         wf(nf)=x1
         sf(nf)=x2         
      enddo
 667  continue
      close(1)

      open(unit=11,file='wvout',status='unknown')
      na=30
      as=-0.5
      ae=0.5
      rmin=1e10
      do ia=1,na
         woff=as+(ae-as)/float(na-1)*float(ia-1)
         rms=0.
         rms2=0.
         nrms2=0
         do i=1,nf
            wnew=wf(i)+woff
            call xlinint(wnew,n,w,s,snew)
            rms=rms+(snew-sf(i))**2
            do j=1,nwl
               if(wf(i).gt.wlo(j).and.wf(i).lt.whi(j)) then
                  rms2=rms2+(snew-sf(i))**2
                  nrms2=nrms2+1
               endif
            enddo
         enddo
         rms=sqrt(rms/float(nf))
         if(nrms2.gt.0) then
            rms2=sqrt(rms2/float(nrms2))
         else
            rms2=rms
         endif
         write(11,*) woff,rms,rms2
         if(rms.lt.rmin) then
            rmin=rms
            wmin=woff
         endif
      enddo
c      write(11,*) wmin
      close(11)         

      end

      subroutine xlinint(xp,n,x,y,yp)
      real x(n),y(n)
      do j=1,n-1
         if(xp.ge.x(j).and.xp.lt.x(j+1)) then
            yp=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
            return
         endif
      enddo
      if(xp.lt.x(1)) yp=y(1)
      if(xp.gt.x(n)) yp=y(n)
      return
      end

