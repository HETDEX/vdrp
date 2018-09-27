
      parameter(nmax=10000)
      real x(nmax),y(nmax),xs(nmax),ys(nmax),xp(nmax),yp(nmax)
      real a(4),sig(nmax)
      character file1*40
      
      sig0=1.5

 1    call qc1('Spectral data ','find.def',file1)
      call savdef
      open(unit=1,file=file1,status='old',err=1)
      
      call qi1('Number of Peaks ','find.def',npeak)
c      call qr3('Continuum, Sigma, and Cut ','find.def',cont,sig,csig)
      call qi2('Difference between peaks, and Number of points ',
     $     'find.def',ndiff,npts)
      call savdef
      xdiff=ndiff

      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2
         n=n+1
         x(n)=x1
         y(n)=x2
      enddo
 666  continue
      close(1)

      call sort2(n,y,x)

      do i=1,n
         xs(i)=x(n-i+1)
         ys(i)=y(n-i+1)
      enddo

      call sort2(n,x,y)

      np=1
      xp(np)=xs(1)
      yp(np)=ys(1)
      do i=2,n
         np=np+1
         xp(np)=xs(i)
         yp(np)=ys(i)
         do j=1,i-1
            if(abs(xs(j)-xs(i)).lt.xdiff) then
               np=np-1
               goto 667
            endif
         enddo
 667     continue
      enddo

      open(unit=1,file='find.out',status='unknown')
      do i=1,npeak
         do j=1,n
            if(xp(i).eq.x(j)) ipv=j
         enddo
         xs(1)=xp(i)
         ys(1)=yp(i)
         a(1)=ys(1)
         a(2)=xs(1)
         a(3)=sig0
         a(4)=0
         xold=xs(1)
         nin=1
         do j=-npts,-1
            ip=ipv+j
            if(ip.ge.1) then
               nin=nin+1
               xs(nin)=x(ip)
               ys(nin)=y(ip)
            endif
         enddo
         do j=1,npts
            ip=ipv+j
            if(ip.le.n) then
               nin=nin+1
               xs(nin)=x(ip)
               ys(nin)=y(ip)
            endif
         enddo
         do j=1,nin
            sig(j)=1.
         enddo
c         if(ipv.gt.2) then
            call sort2(nin,xs,ys)
            call fitgaus(nin,xs,ys,sig,4,a)
c         else
c            a(2)=xp(i)
c         endif
         print *,i,a(2),a(3)/sqrt(2.),a(1),a(4)
         write(1,*) a(2),a(3)/sqrt(2.),a(1),a(4)
      enddo
      close(1)

      end

      subroutine fitgaus(n,x,y,s,na,a)
      real x(n),y(n),s(n),a(na),covar(100,100),alpha(100,100)
      integer ia(100)
      external fgaussc

      tol=1.e-5
      itermax=1000

      do i=1,na
         ia(i)=1
      enddo
      ia(3)=0

      alamda=-1
      alamo=1.e10
      cold=1.e10
      do iter=1,itermax
 
         call mrqmin(x,y,s,n,a,ia,na,covar,alpha,100,
     $        chisq,fgaussc,alamda)
         
         chirel=abs(cold-chisq)/chisq
         cold=chisq
         if(alamda.lt.alamo.and.chirel.lt.tol) goto 667
         if(alamda.gt.1.e9) goto 667
         alamo=alamda
      enddo
      pause 'Hit max iteration'
 667  continue
 
      call mrqmin(x,y,s,n,a,ia,na,covar,alpha,100,
     $     chisq,fgaussc,0.)
 
      return

      end

      SUBROUTINE fgaussc(x,a,y,dyda,na)
      INTEGER na
      REAL x,y,a(na),dyda(na)
      REAL arg,ex,fac
      arg=(x-a(2))/a(3)
      ex=exp(-arg**2)
      fac=a(1)*ex*2.*arg
      y=a(1)*ex+a(4)
      dyda(1)=ex
      dyda(2)=fac/a(3)
      dyda(3)=fac*arg/a(3)
      dyda(4)=1.
      return
      END
