
      parameter(pi=3.141593e0)      
      
      read *,fw0,r1,r2

      sig0=fw0/2.35
      xsig=8.
      nstep=1000
      xs=-xsig*sig0
      xe=xsig*sig0
      ys=-xsig*sig0
      ye=xsig*sig0

      deltx=(xe-xs)/float(nstep-1)
      area=1.0*deltx**2

      sum1=0.
      sum2=0.
      do ix=1,nstep
         xp=xs+float(ix-1)*(xe-xs)/float(nstep-1)
         do iy=1,nstep
            yp=ys+float(iy-1)*(ye-ys)/float(nstep-1)
            dist0=sqrt((xp)**2+(yp)**2)
            if(dist0.lt.r1) then
               g=dist0/sig0
               sum1=sum1+exp(-g*g/2.)/(2.*sig0*sig0*pi)*area
            endif
            if(dist0.lt.r2) then
               g=dist0/sig0
               sum2=sum2+exp(-g*g/2.)/(2.*sig0*sig0*pi)*area
            endif
         enddo
      enddo

      print *,sum1,sum2,sum1/sum2
      end
      
