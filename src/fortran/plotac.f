
      parameter(nmax=20000)
      real m(nmax),b(nmax),v(nmax),r(nmax),cs(nmax),cg(nmax)
      real ct(nmax),a(nmax),ai(nmax),csub(nmax)

      open(unit=1,file='sorted',status='old')
      csmin=1.e31
      cgmin=1.e31
      ctmin=1.e31
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,x5,x6,x7
         if(x6.gt.0.1) then
            n=n+1
            m(n)=x1
            b(n)=x2
            v(n)=x3
            r(n)=x4
            a(n)=x6
c            call xlinint(a(n),ni,ai,csub,csubp)
c            x5=x5-csubp
            cs(n)=x5
            csmin=min(csmin,cs(n))
         endif
      enddo
 666  continue
      close(1)

      do i=1,n
         cs(i)=cs(i)-csmin
      enddo

      call pgbegin(0,'?',2,2)
      call pgpap(0.,1.0)
      call pgscf(2)
      call pgslw(2)

      big=1.2
      small=0.3
      cl=0.
      ch=50.
      x1=2.
      x2=5.
      call pgsch(big)
      call pgenv(x1,x2,cl,ch,0,0)
      call pglabel('M/L\DV','\Gx\U2','')
      call pgsch(small)
      call pgpt(n,m,ct,17)
      call pgsch(big)
      call pgslw(4)
      call plotenv(n,m,ct,x1,x2,ch,14)
      call pgslw(2)
      call pgsci(2)
c      call plotenv(n,m,cs,x1,x2,ch,14)
      call pgsci(4)
c      call plotenv(n,m,cg,x1,x2,ch,14)
      call pgsci(1)
c      x1=0.
c      x2=9.1e9
c      call pgenv(x1,x2,cl,ch,0,0)
c      call pglabel('Black Hole Mass','\\Gx\\U2','')
c      call pgsch(small)
c      call pgpt(n,b,ct,17)
c      call pgsch(big)
c      call pgslw(4)
c      call plotenv(n,b,ct,x1,x2,ch,10)
c      call pgslw(2)
c      call pgsci(2)
cc      call plotenv(n,b,cs,x1,x2,ch,10)
c      call pgsci(4)
cc      call plotenv(n,b,cg,x1,x2,ch,10)
c      call pgsci(1)
      x1=200.
      x2=750.
      call pgenv(x1,x2,cl,ch,0,0)
      call pglabel('V\\DC\\U (km/s)','\\Gx\\U2','')
      call pgsch(small)
      call pgpt(n,v,ct,17)
      call pgsch(big)
      call pgslw(4)
      call plotenv(n,v,ct,x1,x2,ch,13)
      call pgslw(2)
      call pgsci(2)
c      call plotenv(n,v,cs,x1,x2,ch,13)
      call pgsci(4)
c      call plotenv(n,v,cg,x1,x2,ch,13)
      call pgsci(1)
      x1=0.
      x2=43.
      call pgenv(x1,x2,cl,ch,0,0)
      call pglabel('R\\DC\\U (kpc)','\\Gx\\U2','')
      call pgsch(small)
      call pgpt(n,r,ct,17)
      call pgsch(big)
      call pgslw(4)
      call plotenv(n,r,ct,x1,x2,ch,15)
      call pgslw(2)
      call pgsci(2)
c      call plotenv(n,r,cs,x1,x2,ch,15)
      call pgsci(4)
c      call plotenv(n,r,cg,x1,x2,ch,10)

      call pgend
      end

      subroutine plotenv(n,x,y,x1,x2,ch,nsam)
      real x(n),y(n),xplot(1000),yplot(1000)
      real xs(100),ys(100)
      xdiff=(x2-x1)/float(nsam)
      np=0
      do i=1,nsam
         np=np+1
         xlo=x1+float(i-1)*xdiff
         xup=xlo+xdiff
         ymin=ch
         icheck=0
         do j=1,n
            if(x(j).ge.xlo.and.x(j).lt.xup) then
               if(y(j).lt.ymin) then
                  ymin=y(j)
                  xplot(np)=x(j)
                  yplot(np)=ymin
                  icheck=1
               endif
            endif
         enddo
         if(icheck.eq.0) np=np-1
      enddo
      call pgline(np,xplot,yplot)
c      call pgpt(np,xplot,yplot,17)
c      call pgsch(4.0)
c      call pgpt(np,xplot,yplot,17)
c      call pgsch(1.0)

      ns=100
      do i=1,ns
         xs(i)=x1+(x2-x1)*float(i-1)/float(ns-1)
      enddo
c      call smooth(np,xplot,yplot,ns,xs,ys)
c      call pgline(ns,xs,ys)
      return
      end

      subroutine smooth(n,x,y,n2,x2,y2)
      parameter(nmax=2000,mm=2,nwk=nmax+6*(nmax*mm+1),mm2=mm*2)
      real x(n),y(n),x2(n2),y2(n2)
      real*8 dx(nmax),dy(nmax),wx(nmax),cf(nmax),wk(nwk),val,splder
      real*8 q(mm2)

      if(n.gt.nmax) pause 'make nmax bigger in smooth'

c      call qd1('Enter smoothing val ','smflat.def',val)
c      call savdef
      val=0.d0
      md=3
      if(val.eq.0.) md=2
      m=2

      do i=1,n
         dx(i)=dble(x(i))
         dy(i)=dble(y(i))
         wx(i)=1.d0
      enddo

      call gcvspl(dx,dy,nmax,wx,1.d0,m,n,1,md,val,cf,nmax,wk,ier)
      if(ier.ne.0) print *,'ier= ',ier

      do i=1,n2
         in=i
         y2(i)=sngl(splder(0,m,n,dble(x2(i)),dx,cf,in,q))
      enddo

      return
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
