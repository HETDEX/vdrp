      subroutine fithermec(n,x,y,sig,a,ia,na,covar)

      parameter(nca=6)

      real x(n),y(n),a(na),covar(na,na)
      real alpha(nca,nca),sig(n)
      integer ia(na)
      external funcs

      data tol,itermax/1.e-4,1000/

      alamda=-1
      alamo=1.e10
      cold=1.e10
      do iter=1,itermax
         call mrqmin(x,y,sig,n,a,ia,na,covar,alpha,nca,
     $        chisq,funcs,alamda)
         chirel=abs(cold-chisq)/chisq
         cold=chisq
         if(alamda.lt.alamo.and.chirel.lt.tol) goto 666
         if(alamda.gt.1.e9) goto 666
         alamo=alamda
      enddo
      print *,'Hit max iteration'
 666  continue

      call mrqmin(x,y,sig,n,a,ia,na,covar,alpha,nca,
     $     chisq,funcs,0.)

      return
      end

      subroutine funcs(x,a,yfit,dyda,na)
      real a(na),dyda(na)
      parameter(pi=3.141593e0)

      amp=a(1)
      vel=a(2)
      sig=a(3)
      h3=a(4)
      h4=a(5)
      con=a(6)
      w=(x-vel)/sig
      gaus=exp(-w*w/2.)/sqrt(2.*pi)

      yfit=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))+con
      dyda(1)=yfit/amp
      dyda(2)=yfit*w/sig+amp*gaus/sig*(-h3*dfh3(w)/sig-h4*dfh4(w)/sig)
      dyda(3)=-yfit/sig+yfit*w*(x-vel)/sig/sig+amp*gaus/sig*(
     $     -h3*dfh3(w)*(x-vel)/sig/sig-h4*dfh4(w)*(x-vel)/sig/sig)
      dyda(4)=amp*gaus/sig*fh3(w)
      dyda(5)=amp*gaus/sig*fh4(w)
      dyda(6)=1.

      return
      end

      subroutine getlim(n,x,y,xmin,xmax,ymin,ymax)
      real x(n),y(n)
      data big/1.e20/
      xmin=big
      xmax=-big
      ymin=big
      ymax=-big
      do i=1,n
         xmin=min(xmin,x(i))
         xmax=max(xmax,x(i))
         ymin=min(ymin,y(i))
         ymax=max(ymax,y(i))
      enddo
      xbit=(xmax-xmin)/10.
      xmin=xmin-xbit
      xmax=xmax+xbit
      ybit=(ymax-ymin)/10.
      ymin=ymin-ybit
      ymax=ymax+ybit
      return
      end

      function fh3(x)
      fh3=1./sqrt(6.)*(2.*sqrt(2.)*x*x*x-3.*sqrt(2.)*x)
      return
      end
      function fh4(x)
      fh4=1./sqrt(24.)*(4.*x*x*x*x-12.*x*x+3.)
      return
      end
      function dfh3(x)
      dfh3=1./sqrt(6.)*(6.*sqrt(2.)*x*x-3.*sqrt(2.))
      return
      end
      function dfh4(x)
      dfh4=1./sqrt(24.)*(16.*x*x*x-24.*x)
      return
      end

      subroutine ghline(a,xmin,xmax)
      parameter(n=1000,pi=3.141593e0)
      real a(6),x(n),y(n)
      xbit=(xmax-xmin)/10.
      xminp=xmin-xbit
      xmaxp=xmax+xbit
      amp=a(1)
      vel=a(2)
      sig=a(3)
      h3=a(4)
      h4=a(5)
      con=a(6)
      do i=1,n
         x(i)=xminp+(xmaxp-xminp)*(i-1)/float(n-1)
         w=(x(i)-vel)/sig
         gaus=exp(-w*w/2.)/sqrt(2.*pi)
         y(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))+con
      enddo
      call pgline(n,x,y)
      return
      end
