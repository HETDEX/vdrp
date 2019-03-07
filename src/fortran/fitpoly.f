
      parameter(nmax=1000)
      real x(nmax),y(nmax),xp(nmax),yp(nmax),c(5)
      real c1a(nmax),c2a(nmax),c3a(nmax),c4a(nmax)
      real c0a(nmax),rbc(6),ynew(nmax),xin(nmax)
      real xin2(nmax),yin2(nmax),xa(nmax),ya(nmax)
      character title*12

c      ifit0=1 ! 5th order poly
      ifit0=2 ! Robin's fit

c - Robin's coefficients
      rbc(1)=-0.01972573
      rbc(2)=+0.43904144
      rbc(3)=-3.86654830
      rbc(4)=+16.7814045
      rbc(5)=-35.7178268
      rbc(6)=+29.7167950

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      open(unit=11,file='out',status='unknown')
      open(unit=1,file='title',status='old')
      read(1,*) title
      close(1)
      open(unit=1,file='tpavg.dat',status='old')
      na=0
      do i=1,nmax
         read(1,*,end=888) x1,x2
         na=na+1
         xa(na)=x1
         ya(na)=x2
      enddo
 888  continue
      close(1)

      c1fit=0
      c2fit=0
      xmin=3500.
      xmax=5500.
      ymin=0.
      ymax=0.23
      ymin=0.8
      ymax=1.2
      call pgsls(1)
      call pgslw(1)

      call pgsci(1)
      call pgenv(xmin,xmax,ymin,ymax,0,0)
c      call pglabel('Wavelength','Throughput','')
      call pglabel('Wavelength','data/fit',title)

      open(unit=1,file='in',status='old',err=777)

      ysum=0.
      n=0
      do i=1,2000
         read(1,*,end=667) x1,x2
         if(x2.lt.-1) goto 777
         n=n+1
         x(n)=x1
         y(n)=x2
         ysum=ysum+x2
      enddo
 667  continue
      close(1)
      iin1=1
      goto 778
 777  continue
      close(1)
      iin1=0
 778  continue
      if(ysum.eq.0.) iin1=0

      np=100
      do i=1,np
         xp(i)=xmin+(xmax-xmin)/float(np-1)*float(i-1)
         xrbc=xp(i)/1000.
         fit=0.
         do ifit=1,6
            fit=(fit*xrbc)+rbc(ifit)
         enddo
         ynew(i)=fit
      enddo

      if(ifit0.eq.1) then
      c0min=4000.
      c0max=5300.
      n0step=30
      do i=1,n0step
         c0a(i)=c0min+(c0max-c0min)/float(n0step-1)*float(i-1)
      enddo
      c1min=0.005
      c1max=0.28
      n1step=200
      do i=1,n1step
         c1a(i)=c1min+(c1max-c1min)/float(n1step-1)*float(i-1)
      enddo
      c2min=-1e-4
      c2max=1e-4
      n2step=20
      do i=1,n2step
         c2a(i)=c2min+(c2max-c2min)/float(n2step-1)*float(i-1)
      enddo
      c3min=-1e-8
      c3max=-1e-7
      n3step=25
      do i=1,n3step
         c3a(i)=c3min+(c3max-c3min)/float(n3step-1)*float(i-1)
      enddo
      c4min=-3e-12
      c4max=3e-12
      n4step=20
      do i=1,n4step
         c4a(i)=c4min+(c4max-c4min)/float(n4step-1)*float(i-1)
      enddo

      smin=1e10
c      do ic0=1,n0step
      do ic1=1,n1step
      do ic2=1,n2step
      do ic3=1,n3step
      do ic4=1,n4step
      c0v=c0a(ic0)
      c0v=4500.
      c1v=c1a(ic1)
      c2v=c2a(ic2)
      c3v=c3a(ic3)
      c4v=c4a(ic4)

      do i=1,np
         xv=xp(i)-c0v
         yp(i)=c1v+c2v*xv+c3v*xv**2+c4v*xv**3
      enddo
      sum=0.
      do i=1,n
         if(x(i).lt.5450.) then
            call xlinint(x(i),np,xp,yp,yf)
            sum=sum+(y(i)-yf)**2
         endif
      enddo
      if(sum.lt.smin) then
         smin=sum
         c0fit=c0v
         c1fit=c1v
         c2fit=c2v
         c3fit=c3v
         c4fit=c4v
      endif

      enddo
      enddo
      enddo
      enddo
      xfmin=0.0005
      do i=1,np
         xv=xp(i)-c0fit
         yp(i)=c1fit+c2fit*xv+c3fit*xv**2+c4fit*xv**3
         yp(i)=max(yp(i),xfmin)
      enddo

      elseif(ifit0.eq.2) then

         open(unit=2,file='in2',status='old',err=966)
         read(2,*) ng,xgnorm,xge
         close(2)
         iin2=1
         xgnorm=xgnorm*1.1
         goto 967
 966     continue
         iin2=0
 967     continue
         if(ng.lt.1) iin2=0
c         if(xge.gt.0.1) iin2=0

         if(iin1.eq.1) then
            xdmin=1e10
            do i=1,np
               if(abs(xp(i)-4940).lt.xdmin) then
                  xdmin=abs(xp(i)-4940)
                  ydmin=ynew(i)
               endif
            enddo
            yfac=xgnorm/ydmin
            do i=1,np
               yin2(i)=yfac*ynew(i)
            enddo
            
            do i=1,n-1
               call xlinint(x(i),np,xp,ynew,ynew0)
               xin(i)=y(i)/ynew0
            enddo
            call biwgt(xin,n-1,xb,xs)
            
            c1min=xb*0.8
            c1max=xb*1.2
            n1step=100
            do i=1,n1step
               c1a(i)=c1min+(c1max-c1min)/float(n1step-1)*float(i-1)
            enddo
            c2min=-0.4
            c2max=+0.4
            n2step=200
            do i=1,n2step
               c2a(i)=c2min+(c2max-c2min)/float(n2step-1)*float(i-1)
            enddo
            smin=1e10
            do ic1=1,n1step
               do ic2=1,n2step
                  c1v=c1a(ic1)
                  c2v=c2a(ic2)
                  do i=1,np
                     yp(i)=ynew(i)*(c1v+(xp(i)-4500.)*c2v/2000.)
                  enddo
                  sum=0.
                  do i=1,n
                     if(x(i).lt.5450.) then
                        call xlinint(x(i),np,xp,yp,yf)
                        sum=sum+((y(i)-yf)/yf)**2
                     endif
                  enddo
                  if(sum.lt.smin) then
                     smin=sum
                     c1fit=c1v
                     c2fit=c2v
                  endif
               enddo
            enddo
            print *,sum,c1fit,c2fit
            
            xfmin=0.0005
            do i=1,np
               yp(i)=ynew(i)*(c1fit+(xp(i)-4500.)*c2fit/2000.)
               yp(i)=max(yp(i),xfmin)
            enddo
            do i=1,n
               call xlinint(x(i),np,xp,yp,yf)
               call xlinint(x(i),np,xp,yin2,yf2)
               yf2=yf2*(1.0+(x(i)-4500)*0.18/2000.)
               xin(i)=y(i)/yf
               if(iin2.eq.0) then
                  yf2=0.
                  xin2(i)=0.
               else
                  xin2(i)=y(i)/yf2
               endif
               write(11,1101) x(i),yf,0.,0.,xin(i),yf2
            enddo
            call pgsci(1)
            call pgline(n-1,x,xin)
            call pgsci(2)
            call pgline(n-1,x,xin2)
         elseif(iin1.eq.0.and.iin2.eq.1) then
            xdmin=1e10
            do i=1,na
               if(abs(xa(i)-4940).lt.xdmin) then
                  xdmin=abs(xa(i)-4940)
                  ydmin=ya(i)
               endif
            enddo
            yfac=xgnorm/ydmin
            do i=1,na
               yin2(i)=yfac*ya(i)
            enddo
            do i=1,na
               xin(i)=1.
               call xlinint(xa(i),na,xa,yin2,yf)
               write(11,1101) xa(i),yf,0.,0.,0.,yf
            enddo
            call pgsci(1)
            call pgline(na,xa,xin)
         elseif(iin1.eq.0.and.iin2.eq.0) then
            do i=1,na
               xin(i)=1.
               write(11,1101) xa(i),ya(i),0.,0.,0.,0.
            enddo
            call pgsci(1)
            call pgline(na,xa,xin)
         endif
      endif
      close(11)

      open(unit=12,file='out2',status='unknown')
      write(12,*) c1fit,c2fit
      close(12)

      call pgend
 1101 format(1x,f6.1,5(1x,f7.4))

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
