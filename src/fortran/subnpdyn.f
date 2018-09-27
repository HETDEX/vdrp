      function funcf(x,r)
      parameter(nmax=1000,mm=2,mm2=mm*2)
      real*8 dr(nmax),cf(nmax),splder,q(mm2)
      common /ffunc/ dr,cf,xrmaxi,apowi,m,n
      if(x.le.xrmaxi) then
         xdum1=sngl(splder(0,m,n,dble(x),dr,cf,n/2,q))
         xdum2=sngl(splder(1,m,n,dble(x),dr,cf,n/2,q))
      else
         xdumm=sngl(splder(0,m,n,dble(xrmaxi),dr,cf,n/2,q))
         xdum1=xdumm+apowi*(xrmaxi-x)
         xdum2=apowi
      endif
      der=(10**xdum1)*xdum2
      den=10**(2.*x)-10**(2.*r)
      if(den.gt.0.) then
         funcf=der/sqrt(den)
      else
         funcf=0.
      endif
      return
      end

      function funcm(x)
      parameter(nmax=1000,mm=2,mm2=mm*2)
      real*8 drr(nmax),cfr(nmax),cfm(nmax),cfv(nmax),splder,q(mm2)
      common /mfunc/ drr,cfr,cfm,cfv,rp,xrmax,rmax,apow,np
c      if(log10(x).gt.sngl(drr(1))) then
      den=sngl(splder(0,2,np,dble(log10(x)),drr,cfr,np/2,q))
c      else
c         den=sngl(splder(0,2,np,drr(1),drr,cfr,1,q))
c      endif
      funcm=10**den*x*x
      return
      end

      function funcv(x)
      parameter(nmax=1000,mm=2,mm2=mm*2)
      real*8 drr(nmax),cfr(nmax),cfm(nmax),cfv(nmax),splder,q(mm2)
      common /mfunc/ drr,cfr,cfm,cfv,rp,xrmax,rmax,apow,np
      if(x.le.xrmax) then
         rho=sngl(splder(0,2,np,dble(x),drr,cfr,np/2,q))
         xm=sngl(splder(0,2,np,dble(x),drr,cfm,np/2,q))
      else
         rhom=sngl(splder(0,2,np,dble(xrmax),drr,cfr,np/2,q))
         rho=rhom+apow*(xrmax-x)
         xm=sngl(splder(0,2,np,dble(xrmax),drr,cfm,np/2,q))
      endif
      funcv=10**rho*10**xm/10**x
      return
      end

      function funcs(x)
      parameter(nmax=1000,mm=2,mm2=mm*2)
      real nuv2
      real*8 drr(nmax),cfr(nmax),cfm(nmax),cfv(nmax),splder,q(mm2)
      common /mfunc/ drr,cfr,cfm,cfv,rp,xrmax,rmax,apow,np
      nuv2=sngl(splder(0,2,np,dble(x),drr,cfv,np/2,q))
      den=sqrt(10**(2.*x)-10**(2.*rp))
      if(den.eq.0.) then
         funcs=0.
      else
         funcs=10**nuv2*10**(2.*x)/den
      endif
      return
      end

      function funcs2(x)
      parameter(nmax=1000,mm=2,mm2=mm*2)
      real*8 drr(nmax),cfr(nmax),cfm(nmax),cfv(nmax),splder,q(mm2)
      common /mfunc/ drr,cfr,cfm,cfv,rp,xrmax,rmax,apow,np
      rho=sngl(splder(0,2,np,dble(x),drr,cfr,np/2,q))
      xm=sngl(splder(0,2,np,dble(x),drr,cfm,np/2,q))
      funcs2=10**rho*10**xm/10**x*(sqrt(10**(2.*rmax)-10**(2.*rp))-
     $     sqrt(10**(2.*x)-10**(2.*rp)))
      return
      end

      function funci(x)
      parameter(nmax=1000,mm=2,mm2=mm*2)
      real nu
      real*8 drr(nmax),cfr(nmax),cfm(nmax),cfv(nmax),splder,q(mm2)
      common /mfunc/ drr,cfr,cfm,cfv,rp,xrmax,rmax,apow,np
      nu=sngl(splder(0,2,np,dble(x),drr,cfr,np/2,q))
      den=sqrt(10**(2.*x)-10**(2.*rp))
      if(den.eq.0.) then
         funci=0.
      else
         funci=10**nu*10**(2.*x)/den
      endif
      return
      end

      subroutine derivs(x,y,dydx)
      parameter(nmax=1000,mm=2,mm2=mm*2)
      parameter (pi=3.141593e0)
      real*8 drr(nmax),cfr(nmax),cfm(nmax),cfv(nmax),splder,q(mm2)
      real y(2),dydx(2)
      common /mfunc/ drr,cfr,cfm,cfv,rp,xrmax,rmax,apow,np
      rho=sngl(splder(0,2,np,dble(log10(x)),drr,cfr,np/2,q))
      xm=sngl(splder(0,2,np,dble(log10(x)),drr,cfm,np/2,q))
      dydx(1)=4.*pi*x*x*10**rho
      dydx(2)=-10**rho*10**xm/x/x
      return
      end

      SUBROUTINE midsql2(funk,r,aa,bb,s,n)
      INTEGER n
      REAL aa,bb,s,funk
      EXTERNAL funk
      INTEGER it,j
      REAL ddel,del,sum,tnm,x,func,a,b
      func(x,r)=2.*x*funk(aa+x**2,r)
      b=sqrt(bb-aa)
      a=0.
      if (n.eq.1) then
        s=(b-a)*func(0.5*(a+b),r)
      else
        it=3**(n-2)
        tnm=it
        del=(b-a)/(3.*tnm)
        ddel=del+del
        x=a+0.5*del
        sum=0.
        do 11 j=1,it
          sum=sum+func(x,r)
          x=x+ddel
          sum=sum+func(x,r)
          x=x+del
11      continue
        s=(s+(b-a)*sum/tnm)/3.
      endif
      return
      END

      subroutine threed2(npt,xmax,vr,vlum,vden,vsig,eps)
      implicit real*8 (a-h,o-z)
      parameter(nmax=1000,mm=2,mm2=mm*2)
      real rp,xrmax,rmax,apow
      real*8 drr(nmax),cfr(nmax),cfm(nmax),cfv(nmax),splder,q(mm2)
      dimension vlum(npt),vden(npt),vsig(npt),vr(npt)
      dimension y(2)
      external deriv1,deriv2,rkqc
c variables in this common block are initialized by this subroutine
c and used by deriv1,deriv2
      common /temp/ rl(400),denl(400),xm,np2
      common /mfunc/ drr,cfr,cfm,cfv,rp,xrmax,rmax,apow,np
      parameter (pi=3.14159265358979d0)

      if(npt.le.1) pause 'error: npt .le. 1'
      xm=xmax
      np2=npt
c
c  fill radius vector and density vector
c 
      do j=1,npt
         rl(j)=drr(j)
         vr(j)=10.0**rl(j)
         rho=splder(0,2,npt,rl(j),drr,cfr,npt/2,q)
         vden(j)=10**rho
         denl(j)=rho
       enddo
c
c  first integrate outward to determine L(r)
c
       nvar=1
c
c  assuming L(r) is a power law, we can estimate log slope 
c  of density at small radii to find starting value of L(r)
c
       gg=-(log(vden(2)/vden(1)))/(log(vr(2)/vr(1)))
       if(gg.ge.2.8) pause 'warning: central mass nearly divergent'
       y(1)=4.d0*pi*vden(1)*vr(1)**3/(3.d0-gg)
       vlum(1)=y(1)
       do j=1,npt-1
          x1=vr(j)
          x2=vr(j+1)
          h1=x2-x1
          call odeint(y,nvar,x1,x2,eps,h1,hmin,nok,nbad,deriv1,rkqc)
          vlum(j+1)=y(1)
       enddo
c
c now integrate inward to determine p(r) (NOTE: in principle P(r) and L(r) 
c could be found in a single pass, simply by subtracting off p(infinity) to
c satisfy boundary condition p->0 as r->infinity. However numerical errors
c make this procedure unsatisfactory.)
c
       nvar=2
c
c estimate log slope of p(r) at large radii to find starting value of p(r)
       x1=vr(npt)
       x2=vr(npt-1)
       y1=-vlum(npt)*vden(npt)/(x1*x1)
       y2=-vlum(npt-1)*vden(npt-1)/(x2*x2)
       gg=-log(y1/y2)/log(x1/x2)
       if(gg.le.1.2) pause 'warning: outside pressure nearly divergent'
       y(2)=-y1*x1/(gg-1.d0)
       vsig(npt)=y(2)
       do j=1,npt-1
          x1=vr(npt+1-j)
          x2=vr(npt-j)
c
c note that although we integrate both p(r) and L(r), the starting value
c of L(r) at each step is taken from the previous integration, which is
c more accurate at small radii
c
          y(1)=vlum(npt+1-j)
          h1=x2-x1
          call odeint(y,nvar,x1,x2,eps,h1,hmin,nok,nbad,deriv2,rkqc)
          vsig(npt-j)=y(2)
       enddo
       do j=1,npt
c now convert from pressure to rms velocity
          vsig(j)=sqrt(vsig(j)/vden(j))
       enddo
       return
       end
      SUBROUTINE sort2(n,arr,brr)
      INTEGER n,M,NSTACK
      REAL arr(n),brr(n)
      PARAMETER (M=7,NSTACK=50)
      INTEGER i,ir,j,jstack,k,l,istack(NSTACK)
      REAL a,b,temp
      jstack=0
      l=1
      ir=n
1     if(ir-l.lt.M)then
        do 12 j=l+1,ir
          a=arr(j)
          b=brr(j)
          do 11 i=j-1,1,-1
            if(arr(i).le.a)goto 2
            arr(i+1)=arr(i)
            brr(i+1)=brr(i)
11        continue
          i=0
2         arr(i+1)=a
          brr(i+1)=b
12      continue
        if(jstack.eq.0)return
        ir=istack(jstack)
        l=istack(jstack-1)
        jstack=jstack-2
      else
        k=(l+ir)/2
        temp=arr(k)
        arr(k)=arr(l+1)
        arr(l+1)=temp
        temp=brr(k)
        brr(k)=brr(l+1)
        brr(l+1)=temp
        if(arr(l+1).gt.arr(ir))then
          temp=arr(l+1)
          arr(l+1)=arr(ir)
          arr(ir)=temp
          temp=brr(l+1)
          brr(l+1)=brr(ir)
          brr(ir)=temp
        endif
        if(arr(l).gt.arr(ir))then
          temp=arr(l)
          arr(l)=arr(ir)
          arr(ir)=temp
          temp=brr(l)
          brr(l)=brr(ir)
          brr(ir)=temp
        endif
        if(arr(l+1).gt.arr(l))then
          temp=arr(l+1)
          arr(l+1)=arr(l)
          arr(l)=temp
          temp=brr(l+1)
          brr(l+1)=brr(l)
          brr(l)=temp
        endif
        i=l+1
        j=ir
        a=arr(l)
        b=brr(l)
3       continue
          i=i+1
        if(arr(i).lt.a)goto 3
4       continue
          j=j-1
        if(arr(j).gt.a)goto 4
        if(j.lt.i)goto 5
        temp=arr(i)
        arr(i)=arr(j)
        arr(j)=temp
        temp=brr(i)
        brr(i)=brr(j)
        brr(j)=temp
        goto 3
5       arr(l)=arr(j)
        arr(j)=a
        brr(l)=brr(j)
        brr(j)=b
        jstack=jstack+2
        if(jstack.gt.NSTACK)pause 'NSTACK too small in sort2'
        if(ir-i+1.ge.j-l)then
          istack(jstack)=ir
          istack(jstack-1)=i
          ir=j-1
        else
          istack(jstack)=j-1
          istack(jstack-1)=l
          l=i
        endif
      endif
      goto 1
      END
C GCVSPL.FOR, 1986-05-12
C
C***********************************************************************
C
C SUBROUTINE GCVSPL (REAL*8)
C
C Purpose:
C *******
C
C       Natural B-spline data smoothing subroutine, using the Generali-
C       zed Cross-Validation and Mean-Squared Prediction Error Criteria
C       of Craven & Wahba (1979). Alternatively, the amount of smoothing
C       can be given explicitly, or it can be based on the effective
C       number of degrees of freedom in the smoothing process as defined
C       by Wahba (1980). The model assumes uncorrelated, additive noise
C       and essentially smooth, underlying functions. The noise may be
C       non-stationary, and the independent co-ordinates may be spaced
C       non-equidistantly. Multiple datasets, with common independent
C       variables and weight factors are accomodated.
C
C
C Calling convention:
C ******************
C
C       CALL GCVSPL ( X, Y, NY, WX, WY, M, N, K, MD, VAL, C, NC, WK, IER )
C
C Meaning of parameters:
C *********************
C
C       X(N)    ( I )   Independent variables: strictly increasing knot
C                       sequence, with X(I-1).lt.X(I), I=2,...,N.
C       Y(NY,K) ( I )   Input data to be smoothed (or interpolated).
C       NY      ( I )   First dimension of array Y(NY,K), with NY.ge.N.
C       WX(N)   ( I )   Weight factor array; WX(I) corresponds with
C                       the relative inverse variance of point Y(I,*).
C                       If no relative weighting information is
C                       available, the WX(I) should be set to ONE.
C                       All WX(I).gt.ZERO, I=1,...,N.
C       WY(K)   ( I )   Weight factor array; WY(J) corresponds with
C                       the relative inverse variance of point Y(*,J).
C                       If no relative weighting information is
C                       available, the WY(J) should be set to ONE.
C                       All WY(J).gt.ZERO, J=1,...,K.
C                       NB: The effective weight for point Y(I,J) is
C                       equal to WX(I)*WY(J).
C       M       ( I )   Half order of the required B-splines (spline
C                       degree 2*M-1), with M.gt.0. The values M =
C                       1,2,3,4 correspond to linear, cubic, quintic,
C                       and heptic splines, respectively.
C       N       ( I )   Number of observations per dataset, with N.ge.2*M.
C       K       ( I )   Number of datasets, with K.ge.1.
C       MD      ( I )   Optimization mode switch:
C                       |MD| = 1: Prior given value for p in VAL
C                                 (VAL.ge.ZERO). This is the fastest
C                                 use of GCVSPL, since no iteration
C                                 is performed in p.
C                       |MD| = 2: Generalized cross validation.
C                       |MD| = 3: True predicted mean-squared error,
C                                 with prior given variance in VAL.
C                       |MD| = 4: Prior given number of degrees of
C                                 freedom in VAL (ZERO.le.VAL.le.N-M).
C                        MD  < 0: It is assumed that the contents of
C                                 X, W, M, N, and WK have not been
C                                 modified since the previous invoca-
C                                 tion of GCVSPL. If MD < -1, WK(4)
C                                 is used as an initial estimate for
C                                 the smoothing parameter p.
C                       Other values for |MD|, and inappropriate values
C                       for VAL will result in an error condition, or
C                       cause a default value for VAL to be selected.
C                       After return from MD.ne.1, the same number of
C                       degrees of freedom can be obtained, for identical
C                       weight factors and knot positions, by selecting
C                       |MD|=1, and by copying the value of p from WK(4)
C                       into VAL. In this way, no iterative optimization
C                       is required when processing other data in Y.
C       VAL     ( I )   Mode value, as described above under MD.
C       C(NC,K) ( O )   Spline coefficients, to be used in conjunction
C                       with function SPLDER. NB: the dimensions of C
C                       in GCVSPL and in SPLDER are different! In SPLDER,
C                       only a single column of C(N,K) is needed, and the
C                       proper column C(1,J), with J=1...K should be used
C                       when calling SPLDER.
C       NC       ( I )  First dimension of array C(NC,K), NC.ge.N.
C       WK(IWK) (I/W/O) Work vector, with length IWK.ge.6*(N*M+1)+N.
C                       On normal exit, the first 6 values of WK are
C                       assigned as follows:
C
C                       WK(1) = Generalized Cross Validation value
C                       WK(2) = Mean Squared Residual.
C                       WK(3) = Estimate of the number of degrees of
C                               freedom of the residual sum of squares
C                               per dataset, with 0.lt.WK(3).lt.N-M.
C                       WK(4) = Smoothing parameter p, multiplicative
C                               with the splines' derivative constraint.
C                       WK(5) = Estimate of the true mean squared error
C                               (different formula for |MD| = 3).
C                       WK(6) = Gauss-Markov error variance.
C
C                       If WK(4) -->  0 , WK(3) -->  0 , and an inter-
C                       polating spline is fitted to the data (p --> 0).
C                       A very small value > 0 is used for p, in order
C                       to avoid division by zero in the GCV function.
C
C                       If WK(4) --> inf, WK(3) --> N-M, and a least-
C                       squares polynomial of order M (degree M-1) is
C                       fitted to the data (p --> inf). For numerical
C                       reasons, a very high value is used for p.
C
C                       Upon return, the contents of WK can be used for
C                       covariance propagation in terms of the matrices
C                       B and WE: see the source listings. The variance
C                       estimate for dataset J follows as WK(6)/WY(J).
C
C       IER     ( O )   Error parameter:
C
C                       IER = 0:        Normal exit
C                       IER = 1:        M.le.0 .or. N.lt.2*M
C                       IER = 2:        Knot sequence is not strictly
C                                       increasing, or some weight
C                                       factor is not positive.
C                       IER = 3:        Wrong mode  parameter or value.
C
C Remarks:
C *******
C
C       (1) GCVSPL calculates a natural spline of order 2*M (degree
C       2*M-1) which smoothes or interpolates a given set of data
C       points, using statistical considerations to determine the
C       amount of smoothing required (Craven & Wahba, 1979). If the
C       error variance is a priori known, it should be supplied to
C       the routine in VAL, for |MD|=3. The degree of smoothing is
C       then determined to minimize an unbiased estimate of the true
C       mean squared error. On the other hand, if the error variance
C       is not known, one may select |MD|=2. The routine then deter-
C       mines the degree of smoothing to minimize the generalized
C       cross validation function. This is asymptotically the same
C       as minimizing the true predicted mean squared error (Craven &
C       Wahba, 1979). If the estimates from |MD|=2 or 3 do not appear
C       suitable to the user (as apparent from the smoothness of the
C       M-th derivative or from the effective number of degrees of
C       freedom returned in WK(3) ), the user may select an other
C       value for the noise variance if |MD|=3, or a reasonably large
C       number of degrees of freedom if |MD|=4. If |MD|=1, the proce-
C       dure is non-iterative, and returns a spline for the given
C       value of the smoothing parameter p as entered in VAL.
C
C       (2) The number of arithmetic operations and the amount of
C       storage required are both proportional to N, so very large
C       datasets may be accomodated. The data points do not have
C       to be equidistant in the independant variable X or uniformly
C       weighted in the dependant variable Y. However, the data
C       points in X must be strictly increasing. Multiple dataset
C       processing (K.gt.1) is numerically more efficient dan
C       separate processing of the individual datasets (K.eq.1).
C
C       (3) If |MD|=3 (a priori known noise variance), any value of
C       N.ge.2*M is acceptable. However, it is advisable for N-2*M
C       be rather large (at least 20) if |MD|=2 (GCV).
C
C       (4) For |MD| > 1, GCVSPL tries to iteratively minimize the
C       selected criterion function. This minimum is unique for |MD|
C       = 4, but not necessarily for |MD| = 2 or 3. Consequently, 
C       local optima rather that the global optimum might be found,
C       and some actual findings suggest that local optima might
C       yield more meaningful results than the global optimum if N
C       is small. Therefore, the user has some control over the
C       search procedure. If MD > 1, the iterative search starts
C       from a value which yields a number of degrees of freedom
C       which is approximately equal to N/2, until the first (local)
C       minimum is found via a golden section search procedure
C       (Utreras, 1980). If MD < -1, the value for p contained in
C       WK(4) is used instead. Thus, if MD = 2 or 3 yield too noisy
C       an estimate, the user might try |MD| = 1 or 4, for suitably
C       selected values for p or for the number of degrees of
C       freedom, and then run GCVSPL with MD = -2 or -3. The con-
C       tents of N, M, K, X, WX, WY, and WK are assumed unchanged
C       if MD < 0.
C
C       (5) GCVSPL calculates the spline coefficient array C(N,K);
C       this array can be used to calculate the spline function
C       value and any of its derivatives up to the degree 2*M-1
C       at any argument T within the knot range, using subrou-
C       tines SPLDER and SEARCH, and the knot array X(N). Since
C       the splines are constrained at their Mth derivative, only
C       the lower spline derivatives will tend to be reliable
C       estimates of the underlying, true signal derivatives.
C
C       (6) GCVSPL combines elements of subroutine CRVO5 by Utre-
C       ras (1980), subroutine SMOOTH by Lyche et al. (1983), and
C       subroutine CUBGCV by Hutchinson (1985). The trace of the
C       influence matrix is assessed in a similar way as described
C       by Hutchinson & de Hoog (1985). The major difference is
C       that the present approach utilizes non-symmetrical B-spline
C       design matrices as described by Lyche et al. (1983); there-
C       fore, the original algorithm by Erisman & Tinney (1975) has
C       been used, rather than the symmetrical version adopted by
C       Hutchinson & de Hoog.
C
C References:
C **********
C
C       P. Craven & G. Wahba (1979), Smoothing noisy data with
C       spline functions. Numerische Mathematik 31, 377-403.
C
C       A.M. Erisman & W.F. Tinney (1975), On computing certain
C       elements of the inverse of a sparse matrix. Communications
C       of the ACM 18(3), 177-179.
C
C       M.F. Hutchinson & F.R. de Hoog (1985), Smoothing noisy data
C       with spline functions. Numerische Mathematik 47(1), 99-106.
C
C       M.F. Hutchinson (1985), Subroutine CUBGCV. CSIRO Division of
C       Mathematics and Statistics, P.O. Box 1965, Canberra, ACT 2601,
C       Australia.
C
C       T. Lyche, L.L. Schumaker, & K. Sepehrnoori (1983), Fortran
C       subroutines for computing smoothing and interpolating natural
C       splines. Advances in Engineering Software 5(1), 2-5.
C
C       F. Utreras (1980), Un paquete de programas para ajustar curvas
C       mediante funciones spline. Informe Tecnico MA-80-B-209, Depar-
C       tamento de Matematicas, Faculdad de Ciencias Fisicas y Matema-
C       ticas, Universidad de Chile, Santiago.
C
C       Wahba, G. (1980). Numerical and statistical methods for mildly,
C       moderately and severely ill-posed problems with noisy data.
C       Technical report nr. 595 (February 1980). Department of Statis-
C       tics, University of Madison (WI), U.S.A.
C
C Subprograms required:
C ********************
C
C       BASIS, PREP, SPLC, BANDET, BANSOL, TRINV
C
C***********************************************************************
C
      SUBROUTINE GCVSPL ( X, Y, NY, WX, WY, M, N, K, MD, VAL, C, NC,
     1                   WK, IER )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER ( RATIO=2D0, TAU=1.618033983D0, IBWE=7,
     1           ZERO=0D0, HALF=5D-1 , ONE=1D0, TOL=1D-6,
     2           EPS=1D-15, EPSINV=ONE/EPS )
      DIMENSION X(N), Y(NY,K), WX(N), WY(K), C(NC,K), WK(N+6*(N*M+1))
      SAVE M2, NM1, EL
      DATA M2, NM1, EL / 2*0, 0D0 /
C
C***  Parameter check and work array initialization
C
      IER = 0
C***  Check on mode parameter
      IF ((IABS(MD).GT.4) .OR.(  MD.EQ. 0  ) .OR.
     1  ((IABS(MD).EQ.1).AND.( VAL.LT.ZERO)).OR.
     2  ((IABS(MD).EQ.3).AND.( VAL.LT.ZERO)).OR.
     3  ((IABS(MD).EQ.4).AND.((VAL.LT.ZERO) .OR.(VAL.GT.N-M)))) THEN
         IER = 3      !Wrong mode value
         RETURN
      ENDIF
C***  Check on M and N
      IF (MD.GT.0) THEN
         M2  = 2 * M
         NM1 = N - 1
      ELSE
         IF ((M2.NE.2*M).OR.(NM1.NE.N-1)) THEN
            IER = 3      !M or N modified since previous call
            RETURN
         ENDIF
      ENDIF
      IF ((M.LE.0).OR.(N.LT.M2)) THEN
         IER = 1      !M or N invalid
         RETURN
      ENDIF
C***  Check on knot sequence and weights
      IF (WX(1).LE.ZERO) IER = 2
      DO 10 I=2,N
         IF ((WX(I).LE.ZERO).OR.(X(I-1).GE.X(I))) IER = 2
         IF (IER.NE.0) RETURN
   10 CONTINUE
      DO 15 J=1,K
         IF (WY(J).LE.ZERO) IER = 2
         IF (IER.NE.0) RETURN
   15 CONTINUE
C
C***  Work array parameters (address information for covariance 
C***  propagation by means of the matrices STAT, B, and WE). NB:
C***  BWE cannot be used since it is modified by function TRINV.
C
      NM2P1 = N*(M2+1)
      NM2M1 = N*(M2-1)
C     ISTAT = 1            !Statistics array STAT(6)
C     IBWE  = ISTAT + 6      !Smoothing matrix BWE( -M:M  ,N)
      IB    = IBWE  + NM2P1      !Design matrix    B  (1-M:M-1,N)
      IWE   = IB    + NM2M1      !Design matrix    WE ( -M:M  ,N)
C     IWK   = IWE   + NM2P1      !Total work array length N + 6*(N*M+1)
C
C***  Compute the design matrices B and WE, the ratio
C***  of their L1-norms, and check for iterative mode.
C
      IF (MD.GT.0) THEN
         CALL BASIS ( M, N, X, WK(IB), R1, WK(IBWE) )
         CALL PREP  ( M, N, X, WX, WK(IWE), EL )
         EL = EL / R1      !L1-norms ratio (SAVEd upon RETURN)
      ENDIF
      IF (IABS(MD).NE.1) GO TO 20
C***     Prior given value for p
         R1 = VAL
         GO TO 100
C
C***  Iterate to minimize the GCV function (|MD|=2),
C***  the MSE function (|MD|=3), or to obtain the prior
C***  given number of degrees of freedom (|MD|=4).
C
   20 IF (MD.LT.-1) THEN
         R1 = WK(4)      !User-determined starting value
      ELSE
         R1 = ONE / EL      !Default (DOF ~ 0.5)
      ENDIF      
      R2 = R1 * RATIO
      GF2 = SPLC(M,N,K,Y,NY,WX,WY,MD,VAL,R2,EPS,C,NC,
     1          WK,WK(IB),WK(IWE),EL,WK(IBWE))
   40 GF1 = SPLC(M,N,K,Y,NY,WX,WY,MD,VAL,R1,EPS,C,NC,
     1          WK,WK(IB),WK(IWE),EL,WK(IBWE))
      IF (GF1.GT.GF2) GO TO 50
         IF (WK(4).LE.ZERO) GO TO 100            !Interpolation
         R2  = R1
         GF2 = GF1
         R1  = R1 / RATIO
         GO TO 40
   50 R3 = R2 * RATIO
   60 GF3 = SPLC(M,N,K,Y,NY,WX,WY,MD,VAL,R3,EPS,C,NC,
     1          WK,WK(IB),WK(IWE),EL,WK(IBWE))
      IF (GF3.GT.GF2) GO TO 70
         IF (WK(4).GE.EPSINV) GO TO 100      !Least-squares polynomial
         R2  = R3      
         GF2 = GF3
         R3  = R3 * RATIO
         GO TO 60
   70 R2  = R3
      GF2 = GF3
      ALPHA = (R2-R1) / TAU
      R4 = R1 + ALPHA
      R3 = R2 - ALPHA
      GF3 = SPLC(M,N,K,Y,NY,WX,WY,MD,VAL,R3,EPS,C,NC,
     1          WK,WK(IB),WK(IWE),EL,WK(IBWE))
      GF4 = SPLC(M,N,K,Y,NY,WX,WY,MD,VAL,R4,EPS,C,NC,
     1          WK,WK(IB),WK(IWE),EL,WK(IBWE))
   80 IF (GF3.LE.GF4) THEN
         R2  = R4
         GF2 = GF4
         ERR = (R2-R1) / (R1+R2)
         IF ((ERR*ERR+ONE.EQ.ONE).OR.(ERR.LE.TOL)) GO TO 90
         R4  = R3
         GF4 = GF3
         ALPHA = ALPHA / TAU
         R3  = R2 - ALPHA
         GF3 = SPLC(M,N,K,Y,NY,WX,WY,MD,VAL,R3,EPS,C,NC,
     1             WK,WK(IB),WK(IWE),EL,WK(IBWE))
      ELSE
         R1  = R3
         GF1 = GF3
         ERR = (R2-R1) / (R1+R2)
         IF ((ERR*ERR+ONE.EQ.ONE).OR.(ERR.LE.TOL)) GO TO 90
         R3  = R4
         GF3 = GF4
         ALPHA = ALPHA / TAU
         R4 = R1 + ALPHA
         GF4 = SPLC(M,N,K,Y,NY,WX,WY,MD,VAL,R4,EPS,C,NC,
     1             WK,WK(IB),WK(IWE),EL,WK(IBWE))
      ENDIF
      GO TO 80
   90 R1 = HALF * (R1+R2)
C
C***  Calculate final spline coefficients
C
  100 GF1 = SPLC(M,N,K,Y,NY,WX,WY,MD,VAL,R1,EPS,C,NC,
     1          WK,WK(IB),WK(IWE),EL,WK(IBWE))
C
C***  Ready
C
      RETURN
      END
C BASIS.FOR, 1985-06-03
C
C***********************************************************************
C
C SUBROUTINE BASIS (REAL*8)
C
C Purpose:
C *******
C
C       Subroutine to assess a B-spline tableau, stored in vectorized
C       form.
C
C Calling convention:
C ******************
C
C       CALL BASIS ( M, N, X, B, BL, Q )
C
C Meaning of parameters:
C *********************
C
C       M               ( I )   Half order of the spline (degree 2*M-1),
C                               M > 0.
C       N               ( I )   Number of knots, N >= 2*M.
C       X(N)            ( I )   Knot sequence, X(I-1) < X(I), I=2,N.
C       B(1-M:M-1,N)    ( O )   Output tableau. Element B(J,I) of array
C                               B corresponds with element b(i,i+j) of
C                               the tableau matrix B.
C       BL              ( O )   L1-norm of B.
C       Q(1-M:M)        ( W )   Internal work array.
C
C Remark:
C ******
C
C       This subroutine is an adaptation of subroutine BASIS from the
C       paper by Lyche et al. (1983). No checking is performed on the
C       validity of M and N. If the knot sequence is not strictly in-
C       creasing, division by zero may occur.
C
C Reference:
C *********
C
C       T. Lyche, L.L. Schumaker, & K. Sepehrnoori, Fortran subroutines
C       for computing smoothing and interpolating natural splines.
C       Advances in Engineering Software 5(1983)1, pp. 2-5.
C
C***********************************************************************
C
      SUBROUTINE BASIS ( M, N, X, B, BL, Q )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER ( ZERO=0D0, ONE=1D0 )
      DIMENSION X(N), B(1-M:M-1,N), Q(1-M:M)
C
      IF (M.EQ.1) THEN
C***         Linear spline
         DO 3 I=1,N
            B(0,I) = ONE
    3    CONTINUE
         BL = ONE
         RETURN
      ENDIF
C
C***  General splines
C
      MM1 = M - 1
      MP1 = M + 1
      M2  = 2 * M
      DO 15 L=1,N
C***     1st row
         DO 5 J=-MM1,M
            Q(J) = ZERO
    5    CONTINUE
         Q(MM1) = ONE
         IF ((L.NE.1).AND.(L.NE.N))
     1      Q(MM1) = ONE / ( X(L+1) - X(L-1) )
C***     Successive rows
         ARG = X(L)
         DO 13 I=3,M2
            IR = MP1 - I
            V  = Q(IR)
            IF (L.LT.I) THEN
C***               Left-hand B-splines
               DO 6 J=L+1,I
                  U     = V
                  V     = Q(IR+1)
                  Q(IR) = U + (X(J)-ARG)*V
                  IR    = IR + 1
    6          CONTINUE
            ENDIF
            J1 = MAX0(L-I+1,1)
            J2 = MIN0(L-1,N-I)
            IF (J1.LE.J2) THEN
C***               Ordinary B-splines
               IF (I.LT.M2) THEN
                  DO 8 J=J1,J2
                     Y     = X(I+J)
                     U     = V
                     V     = Q(IR+1)
                     Q(IR) = U + (V-U)*(Y-ARG)/(Y-X(J))
                     IR = IR + 1
    8             CONTINUE
               ELSE
                  DO 10 J=J1,J2
                     U     = V
                     V     = Q(IR+1)
                     Q(IR) = (ARG-X(J))*U + (X(I+J)-ARG)*V
                     IR    = IR + 1
   10             CONTINUE
               ENDIF
            ENDIF
            NMIP1 = N - I + 1
            IF (NMIP1.LT.L) THEN
C***           Right-hand B-splines
               DO 12 J=NMIP1,L-1
                  U     = V
                  V     = Q(IR+1)
                  Q(IR) = (ARG-X(J))*U + V
                  IR    = IR + 1
   12          CONTINUE
            ENDIF
   13    CONTINUE
         DO 14 J=-MM1,MM1
            B(J,L) = Q(J)
   14    CONTINUE
   15 CONTINUE
C
C***  Zero unused parts of B
C
      DO 17 I=1,MM1
         DO 16 K=I,MM1
            B(-K,    I) = ZERO
            B( K,N+1-I) = ZERO
   16    CONTINUE
   17 CONTINUE
C
C***  Assess L1-norm of B
C
      BL = 0D0
      DO 19 I=1,N
         DO 18 K=-MM1,MM1
            BL = BL + ABS(B(K,I))
   18    CONTINUE
   19 CONTINUE
      BL = BL / N
C
C***  Ready
C
      RETURN
      END
C PREP.FOR, 1985-07-04
C
C***********************************************************************
C
C SUBROUTINE PREP (REAL*8)
C
C Purpose:
C *******
C
C       To compute the matrix WE of weighted divided difference coeffi-
C       cients needed to set up a linear system of equations for sol-
C       ving B-spline smoothing problems, and its L1-norm EL. The matrix
C       WE is stored in vectorized form.
C
C Calling convention:
C ******************
C
C       CALL PREP ( M, N, X, W, WE, EL )
C
C Meaning of parameters:
C *********************
C
C       M               ( I )   Half order of the B-spline (degree
C                               2*M-1), with M > 0.
C       N               ( I )   Number of knots, with N >= 2*M.
C       X(N)            ( I )   Strictly increasing knot array, with
C                               X(I-1) < X(I), I=2,N.
C       W(N)            ( I )   Weight matrix (diagonal), with
C                               W(I).gt.0.0, I=1,N.
C       WE(-M:M,N)      ( O )   Array containing the weighted divided
C                               difference terms in vectorized format.
C                               Element WE(J,I) of array E corresponds
C                               with element e(i,i+j) of the matrix
C                               W**-1 * E.
C       EL              ( O )   L1-norm of WE.
C
C Remark:
C ******
C
C       This subroutine is an adaptation of subroutine PREP from the paper
C       by Lyche et al. (1983). No checking is performed on the validity
C       of M and N. Division by zero may occur if the knot sequence is
C       not strictly increasing.
C
C Reference:
C *********
C
C       T. Lyche, L.L. Schumaker, & K. Sepehrnoori, Fortran subroutines
C       for computing smoothing and interpolating natural splines.
C       Advances in Engineering Software 5(1983)1, pp. 2-5.
C
C***********************************************************************
C
      SUBROUTINE PREP ( M, N, X, W, WE, EL )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER ( ZERO=0D0, ONE=1D0 )
      DIMENSION X(N), W(N), WE((2*M+1)*N)      !WE(-M:M,N)
C
C***  Calculate the factor F1
C
      M2   = 2 * M
      MP1  = M + 1
      M2M1 = M2 - 1
      M2P1 = M2 + 1
      NM   = N - M
      F1   = -ONE
      IF (M.NE.1) THEN
         DO 5 I=2,M
            F1 = -F1 * I
    5    CONTINUE
         DO 6 I=MP1,M2M1
            F1 = F1 * I
    6    CONTINUE
      END IF
C
C***  Columnwise evaluation of the unweighted design matrix E
C
      I1 = 1
      I2 = M
      JM = MP1
      DO 17 J=1,N
         INC = M2P1
         IF (J.GT.NM) THEN
            F1 = -F1
            F  =  F1
         ELSE
            IF (J.LT.MP1) THEN
                INC = 1
                F   = F1
            ELSE
                F   = F1 * (X(J+M)-X(J-M))
            END IF
         END IF
         IF ( J.GT.MP1) I1 = I1 + 1
         IF (I2.LT.  N) I2 = I2 + 1
         JJ = JM
C***     Loop for divided difference coefficients
         FF = F
         Y = X(I1)
         I1P1 = I1 + 1
         DO 11 I=I1P1,I2
            FF = FF / (Y-X(I))
   11    CONTINUE
         WE(JJ) = FF
         JJ = JJ + M2
         I2M1 = I2 - 1
         IF (I1P1.LE.I2M1) THEN
            DO 14 L=I1P1,I2M1
               FF = F
               Y  = X(L)
               DO 12 I=I1,L-1
                  FF = FF / (Y-X(I))
   12          CONTINUE
               DO 13 I=L+1,I2
                  FF = FF / (Y-X(I))
   13          CONTINUE
               WE(JJ) = FF
               JJ = JJ + M2
   14       CONTINUE
         END IF
         FF = F
         Y = X(I2)
         DO 16 I=I1,I2M1
            FF = FF / (Y-X(I))
   16    CONTINUE
         WE(JJ) = FF
         JJ = JJ + M2
         JM = JM + INC
   17 CONTINUE
C
C***  Zero the upper left and lower right corners of E
C
      KL = 1
      N2M = M2P1*N + 1
      DO 19 I=1,M
         KU = KL + M - I
         DO 18 K=KL,KU
            WE(    K) = ZERO
            WE(N2M-K) = ZERO
   18    CONTINUE
         KL = KL + M2P1
   19 CONTINUE
C
C***  Weighted matrix WE = W**-1 * E and its L1-norm
C
   20 JJ = 0
      EL = 0D0
      DO 22 I=1,N
         WI = W(I)
         DO 21 J=1,M2P1
            JJ     = JJ + 1
            WE(JJ) = WE(JJ) / WI
            EL     = EL + ABS(WE(JJ))
   21    CONTINUE
   22 CONTINUE
      EL = EL / N
C
C***  Ready
C
      RETURN
      END
C SPLC.FOR, 1985-12-12
C
C Author: H.J. Woltring
C
C Organizations: University of Nijmegen, and
C                Philips Medical Systems, Eindhoven
C                (The Netherlands)
C
C***********************************************************************
C
C FUNCTION SPLC (REAL*8)
C
C Purpose:
C *******
C
C       To assess the coefficients of a B-spline and various statistical
C       parameters, for a given value of the regularization parameter p.
C
C Calling convention:
C ******************
C
C       FV = SPLC ( M, N, K, Y, NY, WX, WY, MODE, VAL, P, EPS, C, NC,
C       1           STAT, B, WE, EL, BWE)
C
C Meaning of parameters:
C *********************
C
C       SPLC            ( O )   GCV function value if |MODE|.eq.2,
C                               MSE value if |MODE|.eq.3, and absolute
C                               difference with the prior given number of
C                               degrees of freedom if |MODE|.eq.4.
C       M               ( I )   Half order of the B-spline (degree 2*M-1),
C                               with M > 0.
C       N               ( I )   Number of observations, with N >= 2*M.
C       K               ( I )   Number of datasets, with K >= 1.
C       Y(NY,K)         ( I )   Observed measurements.
C       NY              ( I )   First dimension of Y(NY,K), with NY.ge.N.
C       WX(N)           ( I )   Weight factors, corresponding to the
C                               relative inverse variance of each measure-
C                               ment, with WX(I) > 0.0.
C       WY(K)           ( I )   Weight factors, corresponding to the
C                               relative inverse variance of each dataset,
C                               with WY(J) > 0.0.
C       MODE            ( I )   Mode switch, as described in GCVSPL.
C       VAL             ( I )   Prior variance if |MODE|.eq.3, and
C                               prior number of degrees of freedom if
C                               |MODE|.eq.4. For other values of MODE,
C                               VAL is not used.
C       P               ( I )   Smoothing parameter, with P >= 0.0. If
C                               P.eq.0.0, an interpolating spline is
C                               calculated.
C       EPS             ( I )   Relative rounding tolerance*10.0. EPS is
C                               the smallest positive number such that
C                               EPS/10.0 + 1.0 .ne. 1.0.
C       C(NC,K)         ( O )   Calculated spline coefficient arrays. NB:
C                               the dimensions of in GCVSPL and in SPLDER
C                               are different! In SPLDER, only a single
C                               column of C(N,K) is needed, and the proper
C                               column C(1,J), with J=1...K, should be used
C                               when calling SPLDER.
C       NC              ( I )   First dimension of C(NC,K), with NC.ge.N.
C       STAT(6)         ( O )   Statistics array. See the description in
C                               subroutine GCVSPL.
C       B (1-M:M-1,N)   ( I )   B-spline tableau as evaluated by subroutine
C                               BASIS.
C       WE( -M:M  ,N)   ( I )   Weighted B-spline tableau (W**-1 * E) as
C                               evaluated by subroutine PREP.
C       EL              ( I )   L1-norm of the matrix WE as evaluated by
C                               subroutine PREP.
C       BWE(-M:M,N)     ( O )   Central 2*M+1 bands of the inverted
C                               matrix ( B  +  p * W**-1 * E )**-1
C
C Remarks:
C *******
C
C       This subroutine combines elements of subroutine SPLC0 from the
C       paper by Lyche et al. (1983), and of subroutine SPFIT1 by
C       Hutchinson (1985).
C
C References:
C **********
C
C       M.F. Hutchinson (1985), Subroutine CUBGCV. CSIRO division of
C       Mathematics and Statistics, P.O. Box 1965, Canberra, ACT 2601,
C       Australia.
C
C       T. Lyche, L.L. Schumaker, & K. Sepehrnoori, Fortran subroutines
C       for computing smoothing and interpolating natural splines.
C       Advances in Engineering Software 5(1983)1, pp. 2-5.
C
C***********************************************************************
C
      FUNCTION SPLC( M, N, K, Y, NY, WX, WY, MODE, VAL, P, EPS, C, NC,
     1              STAT, B, WE, EL, BWE)
C
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER ( ZERO=0D0, ONE=1D0, TWO=2D0 )
      DIMENSION Y(NY,K), WX(N), WY(K), C(NC,K), STAT(6),
     1         B(1-M:M-1,N), WE(-M:M,N), BWE(-M:M,N)
C
C***  Check on p-value
C
      DP = P
      STAT(4) = P
      PEL = P * EL
C***  Pseudo-interpolation if p is too small
      IF (PEL.LT.EPS) THEN
         DP = EPS / EL
         STAT(4) = ZERO
      ENDIF
C***  Pseudo least-squares polynomial if p is too large
      IF (PEL*EPS.GT.ONE) THEN
         DP = ONE / (EL*EPS)
         STAT(4) = DP
      ENDIF
C
C***  Calculate  BWE  =  B  +  p * W**-1 * E
C
      DO 40 I=1,N
         KM = -MIN0(M,I-1)
         KP =  MIN0(M,N-I)
         DO 30 L=KM,KP
            IF (IABS(L).EQ.M) THEN
               BWE(L,I) =          DP * WE(L,I)
            ELSE
               BWE(L,I) = B(L,I) + DP * WE(L,I)
            ENDIF
   30    CONTINUE
   40 CONTINUE
C
C***  Solve BWE * C = Y, and assess TRACE [ B * BWE**-1 ]
C
      CALL BANDET ( BWE, M, N )
      CALL BANSOL ( BWE, Y, NY, C, NC, M, N, K )
      STAT(3) = TRINV ( WE, BWE, M, N ) * DP      !trace * p = res. d.o.f.
      TRN = STAT(3) / N
C
C***  Compute mean-squared weighted residual
C
      ESN = ZERO
      DO 70 J=1,K
         DO 60 I=1,N
            DT = -Y(I,J)
            KM = -MIN0(M-1,I-1)
            KP =  MIN0(M-1,N-I)
            DO 50 L=KM,KP
               DT = DT + B(L,I)*C(I+L,J)
   50       CONTINUE
            ESN = ESN + DT*DT*WX(I)*WY(J)
   60    CONTINUE
   70 CONTINUE
      ESN = ESN / (N*K)
C
C***  Calculate statistics and function value
C
      STAT(6) = ESN / TRN             !Estimated variance
      STAT(1) = STAT(6) / TRN         !GCV function value
      STAT(2) = ESN                   !Mean Squared Residual
C     STAT(3) = trace [p*B * BWE**-1] !Estimated residuals' d.o.f.
C     STAT(4) = P                     !Normalized smoothing factor
      IF (IABS(MODE).NE.3) THEN
C***     Unknown variance: GCV
         STAT(5) = STAT(6) - ESN
         IF (IABS(MODE).EQ.1) SPLC = ZERO
         IF (IABS(MODE).EQ.2) SPLC = STAT(1)
         IF (IABS(MODE).EQ.4) SPLC = DABS( STAT(3) - VAL )
      ELSE
C***     Known variance: estimated mean squared error
         STAT(5) = ESN - VAL*(TWO*TRN - ONE)
         SPLC = STAT(5)
      ENDIF
C
      RETURN
      END
C BANDET.FOR, 1985-06-03
C
C***********************************************************************
C
C SUBROUTINE BANDET (REAL*8)
C
C Purpose:
C *******
C
C       This subroutine computes the LU decomposition of an N*N matrix
C       E. It is assumed that E has M bands above and M bands below the
C       diagonal. The decomposition is returned in E. It is assumed that
C       E can be decomposed without pivoting. The matrix E is stored in
C       vectorized form in the array E(-M:M,N), where element E(J,I) of
C       the array E corresponds with element e(i,i+j) of the matrix E.
C
C Calling convention:
C ******************
C
C       CALL BANDET ( E, M, N )
C
C Meaning of parameters:
C *********************
C
C       E(-M:M,N)       (I/O)   Matrix to be decomposed.
C       M, N            ( I )   Matrix dimensioning parameters,
C                               M >= 0, N >= 2*M.
C
C Remark:
C ******
C
C       No checking on the validity of the input data is performed.
C       If (M.le.0), no action is taken.
C
C***********************************************************************
C
      SUBROUTINE BANDET ( E, M, N )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION E(-M:M,N)
C
      IF (M.LE.0) RETURN
      DO 40 I=1,N
         DI = E(0,I)
         MI = MIN0(M,I-1)
         IF (MI.GE.1) THEN
            DO 10 K=1,MI
               DI = DI - E(-K,I)*E(K,I-K)
   10       CONTINUE
            E(0,I) = DI
         ENDIF
         LM = MIN0(M,N-I)
         IF (LM.GE.1) THEN
            DO 30 L=1,LM
               DL = E(-L,I+L)
               KM = MIN0(M-L,I-1)
               IF (KM.GE.1) THEN
                  DU = E(L,I)
                  DO 20 K=1,KM
                     DU = DU - E(  -K,  I)*E(L+K,I-K)
                     DL = DL - E(-L-K,L+I)*E(  K,I-K)
   20             CONTINUE
                  E(L,I) = DU
               ENDIF
               E(-L,I+L) = DL / DI
   30       CONTINUE
         ENDIF
   40 CONTINUE
C
C***  Ready
C
      RETURN
      END
C BANSOL.FOR, 1985-12-12
C
C***********************************************************************
C
C SUBROUTINE BANSOL (REAL*8)
C
C Purpose:
C *******
C
C       This subroutine solves systems of linear equations given an LU
C       decomposition of the design matrix. Such a decomposition is pro-
C       vided by subroutine BANDET, in vectorized form. It is assumed
C       that the design matrix is not singular. 
C
C Calling convention:
C ******************
C
C       CALL BANSOL ( E, Y, NY, C, NC, M, N, K )
C
C Meaning of parameters:
C *********************
C
C       E(-M:M,N)       ( I )   Input design matrix, in LU-decomposed,
C                               vectorized form. Element E(J,I) of the
C                               array E corresponds with element
C                               e(i,i+j) of the N*N design matrix E.
C       Y(NY,K)         ( I )   Right hand side vectors.
C       C(NC,K)         ( O )   Solution vectors.
C       NY, NC, M, N, K ( I )   Dimensioning parameters, with M >= 0,
C                               N > 2*M, and K >= 1.
C
C Remark:
C ******
C
C       This subroutine is an adaptation of subroutine BANSOL from the
C       paper by Lyche et al. (1983). No checking is performed on the
C       validity of the input parameters and data. Division by zero may
C       occur if the system is singular.
C
C Reference:
C *********
C
C       T. Lyche, L.L. Schumaker, & K. Sepehrnoori, Fortran subroutines
C       for computing smoothing and interpolating natural splines.
C       Advances in Engineering Software 5(1983)1, pp. 2-5.
C
C***********************************************************************
C
      SUBROUTINE BANSOL ( E, Y, NY, C, NC, M, N, K )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION E(-M:M,N), Y(NY,K), C(NC,K)
C
C***  Check on special cases: M=0, M=1, M>1
C
      NM1 = N - 1
      IF (M-1) 10,40,80
C
C***  M = 0: Diagonal system
C
   10 DO 30 I=1,N
         DO 20 J=1,K
            C(I,J) = Y(I,J) / E(0,I)
   20    CONTINUE
   30 CONTINUE
      RETURN
C
C***  M = 1: Tridiagonal system
C
   40 DO 70 J=1,K
         C(1,J) = Y(1,J)
         DO 50 I=2,N            !Forward sweep
            C(I,J) =  Y(I,J) - E(-1,I)*C(I-1,J)
   50      CONTINUE
         C(N,J) = C(N,J) / E(0,N)
         DO 60 I=NM1,1,-1      !Backward sweep
            C(I,J) = (C(I,J) - E( 1,I)*C(I+1,J)) / E(0,I)
   60    CONTINUE
   70 CONTINUE
      RETURN
C
C***  M > 1: General system
C
   80 DO 130 J=1,K
         C(1,J) = Y(1,J)
         DO 100 I=2,N            !Forward sweep
            MI = MIN0(M,I-1)
            D  = Y(I,J)
            DO 90 L=1,MI
               D = D - E(-L,I)*C(I-L,J)
   90       CONTINUE
            C(I,J) = D
  100    CONTINUE
         C(N,J) = C(N,J) / E(0,N)
         DO 120 I=NM1,1,-1      !Backward sweep
            MI = MIN0(M,N-I)
            D  = C(I,J)
            DO 110 L=1,MI
               D = D - E( L,I)*C(I+L,J)
  110       CONTINUE
            C(I,J) = D / E(0,I)
  120    CONTINUE
  130 CONTINUE
      RETURN
C
      END
C TRINV.FOR, 1985-06-03
C
C***********************************************************************
C
C FUNCTION TRINV (REAL*8)
C
C Purpose:
C *******
C
C       To calculate TRACE [ B * E**-1 ], where B and E are N * N
C       matrices with bandwidth 2*M+1, and where E is a regular matrix
C       in LU-decomposed form. B and E are stored in vectorized form,
C       compatible with subroutines BANDET and BANSOL.
C
C Calling convention:
C ******************
C
C       TRACE = TRINV ( B, E, M, N )
C
C Meaning of parameters:
C *********************
C
C       B(-M:M,N)       ( I ) Input array for matrix B. Element B(J,I)
C                             corresponds with element b(i,i+j) of the
C                             matrix B.
C       E(-M:M,N)       (I/O) Input array for matrix E. Element E(J,I)
C                             corresponds with element e(i,i+j) of the
C                             matrix E. This matrix is stored in LU-
C                             decomposed form, with L unit lower tri-
C                             angular, and U upper triangular. The unit
C                             diagonal of L is not stored. Upon return,
C                             the array E holds the central 2*M+1 bands
C                             of the inverse E**-1, in similar ordering.
C       M, N            ( I ) Array and matrix dimensioning parameters
C                             (M.gt.0, N.ge.2*M+1).
C       TRINV           ( O ) Output function value TRACE [ B * E**-1 ]
C
C Reference:
C *********
C
C       A.M. Erisman & W.F. Tinney, On computing certain elements of the
C       inverse of a sparse matrix. Communications of the ACM 18(1975),
C       nr. 3, pp. 177-179.
C
C***********************************************************************
C
      REAL*8 FUNCTION TRINV ( B, E, M, N )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER ( ZERO=0D0, ONE=1D0 )
      DIMENSION B(-M:M,N), E(-M:M,N)
C
C***  Assess central 2*M+1 bands of E**-1 and store in array E
C
      E(0,N) = ONE / E(0,N)      !Nth pivot
      DO 40 I=N-1,1,-1
         MI = MIN0(M,N-I)
         DD  = ONE / E(0,I)      !Ith pivot
C***     Save Ith column of L and Ith row of U, and normalize U row
         DO 10 K=1,MI
            E( K,N) = E( K,  I) * DD      !Ith row of U (normalized)
            E(-K,1) = E(-K,K+I)      !Ith column of L
   10    CONTINUE
         DD = DD + DD
C***     Invert around Ith pivot
         DO 30 J=MI,1,-1
            DU = ZERO
            DL = ZERO
            DO 20 K=1,MI
               DU = DU - E( K,N)*E(J-K,I+K)
               DL = DL - E(-K,1)*E(K-J,I+J)
   20       CONTINUE
            E( J,  I) = DU
            E(-J,J+I) = DL
            DD = DD - (E(J,N)*DL + E(-J,1)*DU)
   30    CONTINUE
         E(0,I) = 5D-1 * DD
   40 CONTINUE
C
C***  Assess TRACE [ B * E**-1 ] and clear working storage
C
      DD = ZERO
      DO 60 I=1,N
         MN = -MIN0(M,I-1)
         MP =  MIN0(M,N-I)
         DO 50 K=MN,MP
            DD = DD + B(K,I)*E(-K,K+I)
   50    CONTINUE
   60 CONTINUE
      TRINV = DD
      DO 70 K=1,M
         E( K,N) = ZERO
         E(-K,1) = ZERO
   70 CONTINUE
C
C***  Ready
C
      RETURN
      END
C SPLDER.FOR, 1985-06-11
C
C***********************************************************************
C
C FUNCTION SPLDER (REAL*8)
C
C Purpose:
C *******
C
C       To produce the value of the function (IDER.eq.0) or of the
C       IDERth derivative (IDER.gt.0) of a 2M-th order B-spline at
C       the point T. The spline is described in terms of the half
C       order M, the knot sequence X(N), N.ge.2*M, and the spline
C       coefficients C(N).
C
C Calling convention:
C ******************
C
C       SVIDER = SPLDER ( IDER, M, N, T, X, C, L, Q )
C
C Meaning of parameters:
C *********************
C
C       SPLDER  ( O )   Function or derivative value.
C       IDER    ( I )   Derivative order required, with 0.le.IDER
C                       and IDER.le.2*M. If IDER.eq.0, the function
C                       value is returned; otherwise, the IDER-th
C                       derivative of the spline is returned.
C       M       ( I )   Half order of the spline, with M.gt.0.
C       N       ( I )   Number of knots and spline coefficients,
C                       with N.ge.2*M.
C       T       ( I )   Argument at which the spline or its deri-
C                       vative is to be evaluated, with X(1).le.T
C                       and T.le.X(N).
C       X(N)    ( I )   Strictly increasing knot sequence array,
C                       X(I-1).lt.X(I), I=2,...,N.
C       C(N)    ( I )   Spline coefficients, as evaluated by
C                       subroutine GVCSPL.
C       L       (I/O)   L contains an integer such that:
C                       X(L).le.T and T.lt.X(L+1) if T is within
C                       the range X(1).le.T and T.lt.X(N). If
C                       T.lt.X(1), L is set to 0, and if T.ge.X(N),
C                       L is set to N. The search for L is facili-
C                       tated if L has approximately the right
C                       value on entry.
C       Q(2*M)  ( W )   Internal work array.
C
C Remark:
C ******
C
C       This subroutine is an adaptation of subroutine SPLDER of
C       the paper by Lyche et al. (1983). No checking is performed
C       on the validity of the input parameters.
C
C Reference:
C *********
C
C       T. Lyche, L.L. Schumaker, & K. Sepehrnoori, Fortran subroutines
C       for computing smoothing and interpolating natural splines.
C       Advances in Engineering Software 5(1983)1, pp. 2-5.
C
C***********************************************************************
C
      REAL*8 FUNCTION SPLDER ( IDER, M, N, T, X, C, L, Q )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER ( ZERO=0D0, ONE=1D0 )
      DIMENSION X(N), C(N), Q(2*M)
C
C***  Derivatives of IDER.ge.2*M are alway zero
C
      M2 =  2 * M
      K  = M2 - IDER
      IF (K.LT.1) THEN
         SPLDER = ZERO
         RETURN
      ENDIF
C
C***  Search for the interval value L
C
      CALL SEARCH ( N, X, T, L )
C
C***  Initialize parameters and the 1st row of the B-spline
C***  coefficients tableau
C
      TT   = T
      MP1  =  M + 1
      NPM  =  N + M
      M2M1 = M2 - 1
      K1   =  K - 1
      NK   =  N - K
      LK   =  L - K
      LK1  = LK + 1
      LM   =  L - M
      JL   =  L + 1
      JU   =  L + M2
      II   =  N - M2
      ML   = -L
      DO 2 J=JL,JU
         IF ((J.GE.MP1).AND.(J.LE.NPM)) THEN
            Q(J+ML) = C(J-M)
         ELSE
            Q(J+ML) = ZERO
         ENDIF
    2 CONTINUE
C
C***  The following loop computes differences of the B-spline
C***  coefficients. If the value of the spline is required,
C***  differencing is not necessary.
C
      IF (IDER.GT.0) THEN
         JL = JL - M2
         ML = ML + M2
         DO 6 I=1,IDER
            JL = JL + 1
            II = II + 1
            J1 = MAX0(1,JL)
            J2 = MIN0(L,II)
            MI = M2 - I
            J  = J2 + 1
            IF (J1.LE.J2) THEN
               DO 3 JIN=J1,J2
                  J  =  J - 1
                  JM = ML + J
                  Q(JM) = (Q(JM) - Q(JM-1)) / (X(J+MI) - X(J))
    3          CONTINUE
            ENDIF
            IF (JL.GE.1) GO TO 6
               I1 =  I + 1
               J  = ML + 1
               IF (I1.LE.ML) THEN
                  DO 5 JIN=I1,ML
                     J    =  J - 1
                     Q(J) = -Q(J-1)
    5             CONTINUE
               ENDIF
    6    CONTINUE
         DO 7 J=1,K
            Q(J) = Q(J+IDER)
    7    CONTINUE
      ENDIF
C
C***  Compute lower half of the evaluation tableau
C
      IF (K1.GE.1) THEN      !Tableau ready if IDER.eq.2*M-1
         DO 14 I=1,K1
            NKI  =  NK + I
            IR   =   K
            JJ   =   L
            KI   =   K - I
            NKI1 = NKI + 1
C***        Right-hand B-splines
            IF (L.GE.NKI1) THEN
               DO 9 J=NKI1,L
                  Q(IR) = Q(IR-1) + (TT-X(JJ))*Q(IR)
                  JJ    = JJ - 1
                  IR    = IR - 1
    9          CONTINUE
            ENDIF
C***        Middle B-splines
            LK1I = LK1 + I
            J1 = MAX0(1,LK1I)
            J2 = MIN0(L, NKI)
            IF (J1.LE.J2) THEN
               DO 11 J=J1,J2
                  XJKI  = X(JJ+KI)
                  Z     = Q(IR)
                  Q(IR) = Z + (XJKI-TT)*(Q(IR-1)-Z)/(XJKI-X(JJ))
                  IR    = IR - 1
                  JJ    = JJ - 1
   11          CONTINUE
            ENDIF
C***        Left-hand B-splines
            IF (LK1I.LE.0) THEN
               JJ    = KI
               LK1I1 =  1 - LK1I
               DO 13 J=1,LK1I1
                  Q(IR) = Q(IR) + (X(JJ)-TT)*Q(IR-1)
                  JJ    = JJ - 1
                  IR    = IR - 1
   13          CONTINUE
            ENDIF
   14    CONTINUE
      ENDIF
C
C***  Compute the return value
C
      Z = Q(K)
C***  Multiply with factorial if IDER.gt.0
      IF (IDER.GT.0) THEN
         DO 16 J=K,M2M1
            Z = Z * J
   16    CONTINUE
      ENDIF
      SPLDER = Z
C
C***  Ready
C
      RETURN
      END
C SEARCH.FOR, 1985-06-03
C
C***********************************************************************
C
C SUBROUTINE SEARCH (REAL*8)
C
C Purpose:
C *******
C
C       Given a strictly increasing knot sequence X(1) < ... < X(N),
C       where N >= 1, and a real number T, this subroutine finds the
C       value L such that X(L) <= T < X(L+1).  If T < X(1), L = 0;
C       if X(N) <= T, L = N.
C
C Calling convention:
C ******************
C
C       CALL SEARCH ( N, X, T, L )
C
C Meaning of parameters:
C *********************
C
C       N       ( I )   Knot array dimensioning parameter.
C       X(N)    ( I )   Stricly increasing knot array.
C       T       ( I )   Input argument whose knot interval is to
C                       be found.
C       L       (I/O)   Knot interval parameter. The search procedure
C                       is facilitated if L has approximately the
C                       right value on entry.
C
C Remark:
C ******
C
C       This subroutine is an adaptation of subroutine SEARCH from
C       the paper by Lyche et al. (1983). No checking is performed
C       on the input parameters and data; the algorithm may fail if
C       the input sequence is not strictly increasing.
C
C Reference:
C *********
C
C       T. Lyche, L.L. Schumaker, & K. Sepehrnoori, Fortran subroutines
C       for computing smoothing and interpolating natural splines.
C       Advances in Engineering Software 5(1983)1, pp. 2-5.
C
C***********************************************************************
C
      SUBROUTINE SEARCH ( N, X, T, L )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION X(N)
C
      IF (T.LT.X(1)) THEN
C***     Out of range to the left
         L = 0
         RETURN
      ENDIF
      IF (T.GE.X(N)) THEN
C***     Out of range to the right
         L = N
         RETURN
      ENDIF
C***  Validate input value of L
      L = MAX0(L,1)
      IF (L.GE.N) L = N-1
C
C***  Often L will be in an interval adjoining the interval found
C***  in a previous call to search
C
      IF (T.GE.X(L)) GO TO 5
      L = L - 1
      IF (T.GE.X(L)) RETURN
C
C***  Perform bisection
C
      IL = 1
    3 IU = L
    4 L = (IL+IU) / 2
      IF (IU-IL.LE.1) RETURN
      IF (T.LT.X(L)) GO TO 3
      IL = L
      GO TO 4
    5 IF (T.LT.X(L+1)) RETURN
      L = L + 1
      IF (T.LT.X(L+1)) RETURN
      IL = L + 1
      IU = N
      GO TO 4
C
      END
c
c
c export programs for calculating properties associated with the 
c empirical surface brightness law (Nuker law):
c
c    I(r)= 2^{(beta-gamma)/alpha}/(r^gamma(1+r^alpha)^{(beta-gamma)/alpha})
c
c    alpha:  measures curvature of profile (alpha > 0)
c    beta:   at large radii I(r) propto r^{-beta} (beta > 0)
c    gamma:  at small radii I(r) propto r^{-gamma} (gamma < 2)
c
c note: if beta.le.2 total luminosity diverges
c
c r is in units of break radius r_b, I(r) is in units of surface brightness
c at the break radius I_b, and masses and velocities assume M/L=G=1. To 
c convert to physical units multiply radii by r_b, luminosity densities by 
c I_b/r_b, mass densities by (M/L)I_b/r_b, luminosities by I_br_b^2, 
c masses by (M/L)I_br_b^2, squared velocities by G(M/L)I_bR_b.
c
c function sb (alpha,beta,gamma,r): evaluates surface brightness at radius r
c
c subroutine den (alpha,beta,gamma,r,rho,eps):
c     evaluates luminosity density rho at radius r
c     eps   = accuracy parameter for suboutine qromb
c
c subroutine threed (npt,xmax,alpha,beta,gamma,vr,vlum,vden,vsig,eps)
c     evaluates luminosity density, enclosed luminosity, and rms radial 
c     velocity at an array of radii.
c     radii are at equal log spacings between log_10(r)=-xmax, log_10(r)=xmax
c     there are npt radii, npt .le. 400
c     vr    = array of radii
c     vden  = array of densities
c     vlum  = array of enclosed luminosities
c     vsig  = array of rms radial velocities
c     eps   = accuracy parameter passed on to den and used in odeint
c
c subroutine prepare (npt,xmax,alpha,beta,gamma,eps)
c     calls threed to prepare a common block /hold/ for use by los and eval.
c     the common block contains npt values of radii and rho*v2 at radii 
c     with equal logarithmic spacing from log_10(r) = -xmax to xmax
c     eps = accuracy parameter for threed
c     prepare must be called whenever new values of alpha, beta, or 
c     gamma are used by subroutine los
c
c subroutine los (r,sbsig,eps)
c     computes surface brightness times squared line of sight velocity
c     at radius r
c     uses common block /hold/ prepared by subroutine prepare for the
c     required values of alpha,beta,gamma
c     eps = accuracy parameter for qromb
c
c subroutine eval(r,rho,sig,elum)
c     computes luminosity density, rms radial velocity, and enclosed 
c     luminosity at radius r
c     uses common block /hold/ prepared by subroutine prepare for the
c     required values of alpha,beta,gamma
c
c  WARNINGS: 
c
c  1.  most of these models are singular at r=0 and even when
c  they are not some functions are evaluated using extrapolation in
c  log r. When central quantites are desired, it's better to find them
c  using a small but non-zero value of r as input
c  2. program may fail when alpha is very small because transition from 
c  inner to outer power laws is so slow. In this case increasing parameter 
c  "big" may help 
c
c  recommended values of parameters:
c     eps = 1d-8
c     xmax= 5
c     npt = 400
c  routine threed uses power-law approximations to density between r=0
c  and first array element at log_10(r)=-xmax, and to pressure between 
c  r=infinity and last array element at log_10(r)=xmax. Thus xmax should 
c  not be chosen too small
c
c  auxiliary subroutines and functions:
c     
c  home-made:   deriv1,deriv2,func,func1,func2,func3,func4
c  Press et al: qromb,trapzd,polint,odeint,rkqc,rk4
c
c  common blocks /param/, /temp/, /hold/, /rz/:
c                 used to pass variables to associated subroutines
c                 Need not appear in calling program
c
c  NOTES:
c
c  all arguments to subroutines are real*8
c
c  version 1: November 19, 1994
c  version 2: SAVE statement added to trapzd.f so program works on SGIs
c
c****************************************************************************

      function sb (alpha,beta,gamma,r)
      implicit real*8 (a-h,o-z)
c check for erroneous parameter ranges
      if (alpha.le.0.0d0) pause 'error  : alpha.le.0'
      if (beta.le.0.0d0)  pause 'error  : beta.le.0'
      if (gamma.ge.2.0d0) pause 'error  : gamma.ge.2'
c the following parameter ranges are not erroneous but are unusual
      if (gamma.lt.0.0d0) pause 'warning: gamma<0'
      if (gamma.gt.beta)  pause 'warning: gamma>beta'
c best to use very small non-zero radius rahter than r=0 exactly
      if (r.eq.0) pause 'warning: models usually singular at r=0'
      uexp=(beta-gamma)/alpha
      sb=2.0d0**uexp/(r**gamma*(1.d0+r**alpha)**uexp)
      return
      end

      subroutine den (alpha,beta,gamma,r,rho,eps)
      implicit real*8 (a-h,o-z)
      external func,func1,func2
c variables in this common block are initialized by this subroutine
c and used by func, func1, func2
      common /param/ aexp,bexp,cexp,dexp,uexp,fact,rr
      parameter (pi=3.14159265358979d0)
c check for erroneous parameter ranges
      if (alpha.le.0.0d0) pause 'error  : alpha.le.0'
      if (beta.le.0.0d0)  pause 'error  : beta.le.0'
      if (gamma.ge.2.0d0) pause 'error  : gamma.ge.2'
c the following parameter ranges are not erroneous but are unusual
      if (gamma.lt.0.0d0) pause 'warning: gamma<0'
      if (gamma.gt.beta)  pause 'warning: gamma>beta'
c best to use very small non-zero radius rahter than r=0 exactly
      if (r.eq.0) pause 'warning: models usually singular at r=0'
      aexp=alpha
      bexp=beta
      cexp=gamma
      uexp=(beta-gamma)/alpha
      fact=2.d0**uexp
      rr=r
c
c the values below work ok but are not necessarily the best
c c1 : first part of integration goes from z=0 to z=c1*r
c ran: subsequent integrations go over a factor ran in z
c wex: if best-fit exponent is too close to -1 change to -1+wex or -1-wex
c big: set upper limit to infinity if z>big*max(1,r) 
      c1=0.3d0
      ran=5.0d0
      wex=0.3
      big=30.0d0
c
c first part of integration: z < c1*r; integrand roughly constant
c
      a=0.d0
      b=c1*r
      offset=0.d0
      call qromb(func,a,b,ss,offset,eps)
c
c second part of integration: 
c integrate over successive ranges in which z increases by a factor equal
c to parameter ran. In each range estimate typical power law slope and change
c integration variable so integrand is roughly constant over this range.
c new variable is u=z**(1-dexp)
c
 300  a=b
      b=a*ran
      a1=sqrt(a*b)
      a2=1.5d0*a1
      dexp=-(log(abs(func(a2)))
     $     -log(abs(func(a1))))/(log(a2)-log(a1))
      if (dexp.gt.1.0d0-wex.and.dexp.le.1.d0) dexp=1.0d0-wex
      if (dexp.gt.1.d0.and.dexp.le.1.0d0+wex) dexp=1.0d0+wex
      if (dexp.lt.1.d0) then
          aa=a**(1.d0-dexp)
          bb=b**(1.d0-dexp)
      else
          aa=b**(1.d0-dexp)
          bb=a**(1.d0-dexp)
      endif
      offset=ss
      call qromb(func1,aa,bb,s1,offset,eps)
      ss=ss+s1
      if(b.lt.big*max(1.d0,r)) goto 300
c
c third part of integration: assume i(r) propto r^(-beta) and 
c change integration variable so integrand is roughly constant; 
c then integrate to z = infinity. new variable is u=z**(-1-beta)
c
      aa=0.0d0
      bb=b**(-beta-1.d0)
      offset=ss
      call qromb(func2,aa,bb,s2,offset,eps)
      ss=ss+s2
      rho=-ss/pi
      return
      end

      subroutine threed(npt,xmax,alpha,beta,gamma,vr,vlum,vden,vsig,eps)
      implicit real*8 (a-h,o-z)
      dimension vr(npt),vlum(npt),vden(npt),vsig(npt)
      dimension y(2)
      external deriv1,deriv2,rkqc
c variables in this common block are initialized by this subroutine
c and used by deriv1,deriv2
      common /temp/ rl(400),denl(400),xm,np
      parameter (pi=3.14159265358979d0)
      if(npt.le.1) pause 'error: npt .le. 1'
      xm=xmax
      np=npt
c
c  fill radius vector and density vector
c
      do j=1,npt
         rl(j)=-xmax+(j-1)*2.d0*xmax/dble(npt-1)
         vr(j)=10.0**rl(j)
         r=vr(j)
         call den(alpha,beta,gamma,r,rho,eps)
         vden(j)=rho
         denl(j)=log10(rho)
       enddo
c
c  first integrate outward to determine L(r)
c
       nvar=1
c
c  assuming L(r) is a power law, we can estimate log slope 
c  of density at small radii to find starting value of L(r)
c
       gg=-(log(vden(2)/vden(1)))/(log(vr(2)/vr(1)))
       if(gg.ge.2.8) pause 'warning: central mass nearly divergent'
       y(1)=4.d0*pi*vden(1)*vr(1)**3/(3.d0-gg)
       vlum(1)=y(1)
       do j=1,npt-1
          x1=vr(j)
          x2=vr(j+1)
          h1=x2-x1
          call odeint(y,nvar,x1,x2,eps,h1,hmin,nok,nbad,deriv1,rkqc)
          vlum(j+1)=y(1)
       enddo
c
c now integrate inward to determine p(r) (NOTE: in principle P(r) and L(r) 
c could be found in a single pass, simply by subtracting off p(infinity) to
c satisfy boundary condition p->0 as r->infinity. However numerical errors
c make this procedure unsatisfactory.)
c
       nvar=2
c
c estimate log slope of p(r) at large radii to find starting value of p(r)
       x1=vr(npt)
       x2=vr(npt-1)
       y1=-vlum(npt)*vden(npt)/(x1*x1)
       y2=-vlum(npt-1)*vden(npt-1)/(x2*x2)
       gg=-log(y1/y2)/log(x1/x2)
       if(gg.le.1.2) pause 'warning: outside pressure nearly divergent'
       y(2)=-y1*x1/(gg-1.d0)
       vsig(npt)=y(2)
       do j=1,npt-1
          x1=vr(npt+1-j)
          x2=vr(npt-j)
c
c note that although we integrate both p(r) and L(r), the starting value
c of L(r) at each step is taken from the previous integration, which is
c more accurate at small radii
c
          y(1)=vlum(npt+1-j)
          h1=x2-x1
          call odeint(y,nvar,x1,x2,eps,h1,hmin,nok,nbad,deriv2,rkqc)
          vsig(npt-j)=y(2)
       enddo
       do j=1,npt
c now convert from pressure to rms velocity
          vsig(j)=sqrt(vsig(j)/vden(j))
       enddo
       return
       end

      subroutine prepare (npt,xmax,alpha,beta,gamma,eps)
      implicit real*8 (a-h,o-z)
      dimension vr(400),vlum(400),vden(400),vsig(400)
c variables in this common block are initialized in this subroutine and
c used by func3
      common /hold/ ar(400),alum(400),aden(400),asig(400),xm,np
c
      call threed(npt,xmax,alpha,beta,gamma,vr,vlum,vden,vsig,eps)
      do j=1,npt
         ar(j)=log10(vr(j))
         alum(j)=log10(vlum(j))
         aden(j)=log10(vden(j))
         asig(j)=log10(vsig(j))
      enddo
      np=npt
      xm=xmax
      return
      end

      subroutine los (r,sbsig,eps)
      implicit real*8 (a-h,o-z)
      common /rz/ rr,dexp
      external func3,func4
      parameter (pi=3.14159265358979d0)
      if (r.eq.0) pause 'warning: models usually singular at r=0'
      rr=r
c
c the values below work ok but are not necessarily the best
c c1 : first part of integration goes from z=0 to z=c1*r
c ran: subsequent integrations go over a factor ran in z
c wex: if best-fit exponent is too close to -1 change to -1+wex or -1-wex
c big: set upper limit to infinity if z>big*max(1,r) 
      c1=0.3d0
      ran=5.0d0
      wex=0.3
      big=30.0d0
c
c first part of integration: z < c1*r; integrand roughly constant
c
      a=0.d0
      b=c1*r
      offset=0.d0
      call qromb(func3,a,b,ss,offset,eps)
c
c second part of integration: 
c integrate over successive ranges in which z increases by a factor equal
c to parameter ran. In each range estimate typical power law slope and change
c integration variable so integrand is roughly constant over this range.
c new variable is u=z**(1-dexp)
c
      iflag=0
 300  a=b
      b=a*ran
      a1=sqrt(a*b)
      a2=1.5d0*a1
      dexp=-(log(func3(a2))
     $     -log(func3(a1)))/(log(a2)-log(a1))
      if (dexp.gt.1.0d0-wex.and.dexp.le.1.d0) dexp=1.0d0-wex
      if (dexp.gt.1.d0.and.dexp.le.1.0d0+wex) dexp=1.0d0+wex
      if (dexp.lt.1.d0) then
          aa=a**(1.d0-dexp)
          bb=b**(1.d0-dexp)
      else
          aa=b**(1.d0-dexp)
          bb=a**(1.d0-dexp)
          if(b.gt.big*max(1.d0,r)) then
             aa=0
             iflag=1
          endif
      endif
      offset=ss
      call qromb(func4,aa,bb,s1,offset,eps)
      ss=ss+s1
      if(iflag.eq.0) goto 300
      sbsig=2*ss
      return
      end

      subroutine eval (r,rho,sig,elum)
      implicit real*8 (a-h,o-z)
      common /hold/ ar(400),alum(400),aden(400),asig(400),xmax,npt
      if (r.eq.0) pause 'warning: models usually singular at r=0'
      rl=log10(r)
      dl=2*xmax/dble(npt-1)
      jj=(rl+xmax)/dl
      jj=max(jj,1)
      jj=min(jj,npt-2)
      rl1=ar(jj)
      rl2=ar(jj+1)
      rl3=ar(jj+2)
      yl1=alum(jj)
      yl2=alum(jj+1)
      yl3=alum(jj+2)
      yd1=aden(jj)
      yd2=aden(jj+1)
      yd3=aden(jj+2)
      ys1=asig(jj)
      ys2=asig(jj+1)
      ys3=asig(jj+2)
c  use quadratic interpolation in log-log coordinates
      yl=(0.5d0*yl1*(rl-rl2)*(rl-rl3)-yl2*(rl-rl1)*(rl-rl3)
     $  +0.5d0*yl3*(rl-rl1)*(rl-rl2))/(dl*dl)
      yd=(0.5d0*yd1*(rl-rl2)*(rl-rl3)-yd2*(rl-rl1)*(rl-rl3)
     $  +0.5d0*yd3*(rl-rl1)*(rl-rl2))/(dl*dl)
      ys=(0.5d0*ys1*(rl-rl2)*(rl-rl3)-ys2*(rl-rl1)*(rl-rl3)
     $  +0.5d0*ys3*(rl-rl1)*(rl-rl2))/(dl*dl)
c unless extrapolating, in which case use linear interpolation
      if(rl.lt.-xmax) then
         yl=(-yl1*(rl-rl2)+yl2*(rl-rl1))/dl
         yd=(-yd1*(rl-rl2)+yd2*(rl-rl1))/dl
         ys=(-ys1*(rl-rl2)+ys2*(rl-rl1))/dl
      endif
      if(rl.gt.xmax) then
         yl=(yl3*(rl-rl2)-yl2*(rl-rl3))/dl
         yd=(yd3*(rl-rl2)-yd2*(rl-rl3))/dl
         ys=(ys3*(rl-rl2)-ys2*(rl-rl3))/dl
      endif
      rho=10.0**yd
      sig=10.0**ys
      elum=10.0**yl      
      return
      end

      subroutine deriv1(r,y,dydx)
c used for first outward integration
      implicit real*8 (a-h,o-z)
      dimension y(1),dydx(1)
      parameter (pi=3.14159265358979d0)
      common /temp/ rl(400),denl(400),xmax,npt
      ul=log10(r)
      dl=2*xmax/dble(npt-1)
      jj=(ul+xmax)/dl
      jj=max(jj,1)
      jj=min(jj,npt-2)
      ul1=rl(jj)
      ul2=rl(jj+1)
      ul3=rl(jj+2)
      y1=denl(jj)
      y2=denl(jj+1)
      y3=denl(jj+2)
c  use quadratic interpolation in log-log coordinates
      yy=(0.5d0*y1*(ul-ul2)*(ul-ul3)-y2*(ul-ul1)*(ul-ul3)
     $  +0.5d0*y3*(ul-ul1)*(ul-ul2))/(dl*dl)
c unless extrapolating, in which case use linear extrapolation
      if(ul.lt.-xmax) yy=(-y1*(ul-ul2)+y2*(ul-ul1))/dl
      if(ul.gt.xmax) yy=(y3*(ul-ul2)-y2*(ul-ul3))/dl
      rho=10.0**yy
c dL/dr=4*pi*r^2*rho
      dydx(1)=4.d0*pi*r*r*rho
      return
      end

      subroutine deriv2(r,y,dydx)
c used for second inward integration
      implicit real*8 (a-h,o-z)
      dimension y(2),dydx(2)
      parameter (pi=3.14159265358979d0)
      common /temp/ rl(400),denl(400),xmax,npt
      ul=log10(r)
      dl=2*xmax/dble(npt-1)
      jj=(ul+xmax)/dl
      jj=max(jj,1)
      jj=min(jj,npt-2)
      ul1=rl(jj)
      ul2=rl(jj+1)
      ul3=rl(jj+2)
      y1=denl(jj)
      y2=denl(jj+1)
      y3=denl(jj+2)
c  use quadratic interpolation in log-log coordinates
      yy=(0.5d0*y1*(ul-ul2)*(ul-ul3)-y2*(ul-ul1)*(ul-ul3)
     $  +0.5d0*y3*(ul-ul1)*(ul-ul2))/(dl*dl)
c unless extrapolating, in which case use linear extrapolation
      if(ul.lt.-xmax) yy=(-y1*(ul-ul2)+y2*(ul-ul1))/dl
      if(ul.gt.xmax) yy=(y3*(ul-ul2)-y2*(ul-ul3))/dl
      rho=10.0**yy
      rr=r*r
c dL/dr=4*pi*r^2*rho
      dydx(1)=4.d0*pi*rr*rho
c dp/dr=-L(r)*rho/r^2 assuming M/L=1
      dydx(2)=-y(1)*rho/rr
      return
      end

      function func(z)
      implicit real*8 (a-h,o-z)
      common /param/ aexp,bexp,cexp,dexp,uexp,fact,rr
      r2=rr*rr+z*z
      ral=r2**(0.5d0*aexp)
      rga=r2**(0.5d0*cexp+1.d0)
      den=(1.d0+ral)**(uexp+1.d0)
      func=-fact*(cexp+bexp*ral)/(rga*den)
      return
      end

      function func1(u)
      implicit real*8 (a-h,o-z)
      common /param/ aexp,bexp,cexp,dexp,uexp,fact,rr
      z=u**(1.d0/(1.d0-dexp))
      func1=func(z)*z/(u*abs(dexp-1.d0))
      return
      end
      
      function func2(u)
      implicit real*8 (a-h,o-z)
      common /param/ aexp,bexp,cexp,dexp,uexp,fact,rr
      if(u.gt.1d-10) then
         z=u**(-1.d0/(1.d0+bexp))
         func2=func(z)*z/(u*(1.d0+bexp))
      else
         func2=-fact*bexp/((1.d0+bexp)*
     $     (1.d0+rr*rr*u**(2.d0/(1.d0+bexp)))**(1.d0+0.5d0*bexp))
      endif
      return
      end

      function func3(z)
      implicit real*8 (a-h,o-z)
      common /hold/ ar(400),alum(400),aden(400),asig(400),xmax,npt
      common /rz/ rr,dexp
      r2=rr*rr+z*z
      ul=0.5d0*log10(r2)
      dl=2*xmax/dble(npt-1)
      jj=(ul+xmax)/dl
      jj=max(jj,1)
      jj=min(jj,npt-2)
      ul1=ar(jj)
      ul2=ar(jj+1)
      ul3=ar(jj+2)
      y1=aden(jj)+2.0d0*asig(jj)
      y2=aden(jj+1)+2.0d0*asig(jj+1)
      y3=aden(jj+2)+2.0d0*asig(jj+2)
c  use quadratic interpolation in log-log coordinates
      y=(0.5d0*y1*(ul-ul2)*(ul-ul3)-y2*(ul-ul1)*(ul-ul3)
     $  +0.5d0*y3*(ul-ul1)*(ul-ul2))/(dl*dl)
c unless extrapolating, in which case use linear interpolation
      if(ul.lt.-xmax) y=(-y1*(ul-ul2)+y2*(ul-ul1))/dl
      if(ul.gt.xmax) y=(y3*(ul-ul2)-y2*(ul-ul3))/dl
      func3=10**y
      return
      end

      function func4(u)
      implicit real*8 (a-h,o-z)
      common /rz/ rr,dexp
      if (u.gt.1d-20) then
         z=u**(1.0d0/(1.0d0-dexp))
         func4=func3(z)*z/(u*abs(dexp-1.0d0))
      else
         func4=0
      endif
      return
      end

      subroutine qromb(func,a,b,ss,offset,eps)
c
c   qromb modified from Press et al by extra arguments offset and eps:
c   offset is defined such that integration stops when
c   relative error is less than eps or absolute error is
c   less than eps*offset
c
      implicit real*8 (a-h,o-z)
      parameter(jmax=20,jmaxp=jmax+1,k=5,km=4)
      dimension s(jmaxp),h(jmaxp)
      external func
      h(1)=1.d0
      do 11 j=1,jmax
        call trapzd(func,a,b,s(j),j)
        if (j.ge.k) then
          l=j-km
          call polint(h(l),s(l),k,0.d0,ss,dss)
          if (abs(dss).lt.eps*max(abs(ss),abs(offset))) return
        endif
        s(j+1)=s(j)
        h(j+1)=0.25d0*h(j)
11    continue
      pause 'too many steps.'
      end

      subroutine trapzd(func,a,b,s,n)
      implicit real*8 (a-h,o-z)
      external func
      save it
      if (n.eq.1) then
        s=0.5d0*(b-a)*(func(a)+func(b))
        it=1
      else
        tnm=it
        del=(b-a)/tnm
        x=a+0.5d0*del
        sum=0.d0
        do 11 j=1,it
          sum=sum+func(x)
          x=x+del
11      continue
        s=0.5d0*(s+(b-a)*sum/tnm)
        it=2*it
      endif
      return
      end

      subroutine polint(xa,ya,n,x,y,dy)
      implicit real*8 (a-h,o-z)
      parameter (nmax=10) 
      dimension xa(n),ya(n),c(nmax),d(nmax)
      ns=1
      dif=abs(x-xa(1))
      do 11 i=1,n 
        dift=abs(x-xa(i))
        if (dift.lt.dif) then
          ns=i
          dif=dift
        endif
        c(i)=ya(i)
        d(i)=ya(i)
11    continue
      y=ya(ns)
      ns=ns-1
      do 13 m=1,n-1
        do 12 i=1,n-m
          ho=xa(i)-x
          hp=xa(i+m)-x
          w=c(i+1)-d(i)
          den=ho-hp
          if(den.eq.0.)pause
          den=w/den
          d(i)=hp*den
          c(i)=ho*den
12      continue
        if (2*ns.lt.n-m)then
          dy=c(ns+1)
        else
          dy=d(ns)
          ns=ns-1
        endif
        y=y+dy
13    continue
      return
      end

      subroutine odeint(ystart,nvar,x1,x2,eps,h1,hmin,nok,nbad,derivs,
     $  rkqc)
      integer nbad,nok,nvar,maxstp,nmax
      real*8 eps,h1,hmin,x1,x2,ystart(nvar),two,zero,tiny
      external derivs,rkqc
      parameter (maxstp=10000,nmax=10,two=2.0,zero=0.0,tiny=1.d-30)
      common /path/ kmax,kount,dxsav,xp,yp
      integer i,kmax,kount,nstp
      real*8 dxsav,h,hdid,hnext,x,xsav,dydx(nmax),xp(400),y(nmax),
     $  yp(10,400),yscal(nmax)
      x=x1
      h=sign(h1,x2-x1)
      nok=0
      nbad=0
      kount=0
      do 11 i=1,nvar
        y(i)=ystart(i)
   11 continue
      if (kmax.gt.0) xsav=x-dxsav*two
      do 16 nstp=1,maxstp
        call derivs(x,y,dydx)
        do 12 i=1,nvar
          yscal(i)=abs(y(i))+abs(h*dydx(i))+tiny
   12   continue
        if(kmax.gt.0)then
          if(abs(x-xsav).gt.abs(dxsav)) then
            if(kount.lt.kmax-1)then
              kount=kount+1
              xp(kount)=x
              do 13 i=1,nvar
                yp(i,kount)=y(i)
   13         continue
              xsav=x
            endif
          endif
        endif
        if((x+h-x2)*(x+h-x1).gt.zero) h=x2-x
        call rkqc(y,dydx,nvar,x,h,eps,yscal,hdid,hnext,derivs)
        if(hdid.eq.h)then
          nok=nok+1
        else
          nbad=nbad+1
        endif
        if((x-x2)*(x2-x1).ge.zero)then
          do 14 i=1,nvar
            ystart(i)=y(i)
   14     continue
          if(kmax.ne.0)then
            kount=kount+1
            xp(kount)=x
            do 15 i=1,nvar
              yp(i,kount)=y(i)
   15       continue
          endif
          return
        endif
        if(abs(hnext).lt.hmin) pause 'stepsize smaller than minimum.'
        h=hnext
   16 continue
      pause 'too many steps.'
      return
      end

      subroutine rkqc(y,dydx,n,x,htry,eps,yscal,hdid,hnext,derivs)
      integer n,nmax
      real*8 eps,hdid,hnext,htry,x,dydx(n),y(n),yscal(n),pgrow,
     $  pshrnk,fcor,one,safety,errcon
      external derivs
      parameter (nmax=10,pgrow=-0.20,pshrnk=-0.25,fcor=1./15.,
     $  one=1.,safety=0.9,errcon=6.e-4)
      integer i
      real*8 errmax,h,hh,xsav,dysav(nmax),ysav(10),ytemp(nmax)
      xsav=x
      do 11 i=1,n
        ysav(i)=y(i)
        dysav(i)=dydx(i)
   11 continue
      h=htry
    1 hh=0.5d0*h
      call rk4(ysav,dysav,n,xsav,hh,ytemp,derivs)
      x=xsav+hh
      call derivs(x,ytemp,dydx)
      call rk4(ytemp,dydx,n,x,hh,y,derivs)
      x=xsav+h
      if(x.eq.xsav)pause 'stepsize not significant in rkqc.'
      call rk4(ysav,dysav,n,xsav,h,ytemp,derivs)
      errmax=0.
      do 12 i=1,n
        ytemp(i)=y(i)-ytemp(i)
        errmax=max(errmax,abs(ytemp(i)/yscal(i)))
   12 continue
      errmax=errmax/eps
      if(errmax.gt.one) then
        h=safety*h*(errmax**pshrnk)
        goto 1
      else
        hdid=h
        if(errmax.gt.errcon)then
          hnext=safety*h*(errmax**pgrow)
        else
          hnext=4.*h
        endif
      endif
      do 13 i=1,n
        y(i)=y(i)+ytemp(i)*fcor
   13 continue
      return
      end

      subroutine rk4(y,dydx,n,x,h,yout,derivs)
      integer n,nmax
      real*8 h,x,dydx(n),y(n),yout(n)
      external derivs
      parameter (nmax=10)
      integer i
      real*8 h6,hh,xh,dym(nmax),dyt(nmax),yt(10)
      hh=h*0.5d0
      h6=h/6.d0
      xh=x+hh
      do 11 i=1,n
        yt(i)=y(i)+hh*dydx(i)
   11 continue
      call derivs(xh,yt,dyt)
      do 12 i=1,n
        yt(i)=y(i)+hh*dyt(i)
   12 continue
      call derivs(xh,yt,dym)
      do 13 i=1,n
        yt(i)=y(i)+h*dym(i)
        dym(i)=dyt(i)+dym(i)
   13 continue
      call derivs(x+h,yt,dyt)
      do 14 i=1,n
        yout(i)=y(i)+h6*(dydx(i)+dyt(i)+2.*dym(i))
   14 continue
      return
      end
      subroutine qags(f,a,b,epsabs,epsrel,result,abserr,neval,ier,
     *   limit,lenw,last,iwork,work)
c***begin prologue  qags
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a1
c***keywords  automatic integrator, general-purpose,
c             (end-point) singularities, extrapolation,
c             globally adaptive
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & prog. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            definite integral  i = integral of f over (a,b),
c            hopefully satisfying following claim for accuracy
c            abs(i-result).le.max(epsabs,epsrel*abs(i)).
c***description
c
c        computation of a definite integral
c        standard fortran subroutine
c        real version
c
c
c        parameters
c         on entry
c            f      - real
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - real
c                     lower limit of integration
c
c            b      - real
c                     upper limit of integration
c
c            epsabs - real
c                     absolute accuracy requested
c            epsrel - real
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c         on return
c            result - real
c                     approximation to the integral
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine
c                             the estimates for integral and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c            error messages
c                     ier = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more sub-
c                             divisions by increasing the value of limit
c                             (and taking the according dimension
c                             adjustments into account. however, if
c                             this yields no improvement it is advised
c                             to analyze the integrand in order to
c                             determine the integration difficulties. if
c                             the position of a local difficulty can be
c                             determined (e.g. singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling the
c                             integrator on the subranges. if possible,
c                             an appropriate special-purpose integrator
c                             should be used, which is designed for
c                             handling the type of difficulty involved.
c                         = 2 the occurrence of roundoff error is detec-
c                             ted, which prevents the requested
c                             tolerance from being achieved.
c                             the error may be under-estimated.
c                         = 3 extremely bad integrand behaviour
c                             occurs at some points of the integration
c                             interval.
c                         = 4 the algorithm does not converge.
c                             roundoff error is detected in the
c                             extrapolation table. it is presumed that
c                             the requested tolerance cannot be
c                             achieved, and that the returned result is
c                             the best which can be obtained.
c                         = 5 the integral is probably divergent, or
c                             slowly convergent. it must be noted that
c                             divergence can occur with any other value
c                             of ier.
c                         = 6 the input is invalid, because
c                             (epsabs.le.0 and
c                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28)
c                             or limit.lt.1 or lenw.lt.limit*4.
c                             result, abserr, neval, last are set to
c                             zero.except when limit or lenw is invalid,
c                             iwork(1), work(limit*2+1) and
c                             work(limit*3+1) are set to zero, work(1)
c                             is set to a and work(limit+1) to b.
c
c         dimensioning parameters
c            limit - integer
c                    dimensioning parameter for iwork
c                    limit determines the maximum number of subintervals
c                    in the partition of the given integration interval
c                    (a,b), limit.ge.1.
c                    if limit.lt.1, the routine will end with ier = 6.
c
c            lenw  - integer
c                    dimensioning parameter for work
c                    lenw must be at least limit*4.
c                    if lenw.lt.limit*4, the routine will end
c                    with ier = 6.
c
c            last  - integer
c                    on return, last equals the number of subintervals
c                    produced in the subdivision process, detemines the
c                    number of significant elements actually in the work
c                    arrays.
c
c         work arrays
c            iwork - integer
c                    vector of dimension at least limit, the first k
c                    elements of which contain pointers
c                    to the error estimates over the subintervals
c                    such that work(limit*3+iwork(1)),... ,
c                    work(limit*3+iwork(k)) form a decreasing
c                    sequence, with k = last if last.le.(limit/2+2),
c                    and k = limit+1-last otherwise
c
c            work  - real
c                    vector of dimension at least lenw
c                    on return
c                    work(1), ..., work(last) contain the left
c                     end-points of the subintervals in the
c                     partition of (a,b),
c                    work(limit+1), ..., work(limit+last) contain
c                     the right end-points,
c                    work(limit*2+1), ..., work(limit*2+last) contain
c                     the integral approximations over the subintervals,
c                    work(limit*3+1), ..., work(limit*3+last)
c                     contain the error estimates.
c
c
c
c***references  (none)
c***routines called  qagse,xerror
c***end prologue  qags
c
c
      real a,abserr,b,epsabs,epsrel,f,result,work
      integer ier,iwork,lenw,limit,lvl,l1,l2,l3,neval
c
      dimension iwork(limit),work(lenw)
c
      external f
c
c         check validity of limit and lenw.
c
c***first executable statement  qags
      ier = 6
      neval = 0
      last = 0
      result = 0.0e+00
      abserr = 0.0e+00
      if(limit.lt.1.or.lenw.lt.limit*4) go to 10
c
c         prepare call for qagse.
c
      l1 = limit+1
      l2 = limit+l1
      l3 = limit+l2
c
      call qagse(f,a,b,epsabs,epsrel,limit,result,abserr,neval,
     *  ier,work(1),work(l1),work(l2),work(l3),iwork,last)
c
c         call error handler if necessary.
c
      lvl = 0
10    if(ier.eq.6) lvl = 1
c      if(ier.ne.0) pause 'abnormal return from  qags'

c     call xerror(26habnormal return from  qags,
c     *  26,ier,lvl)

      return
      end
      subroutine qagse(f,a,b,epsabs,epsrel,limit,result,abserr,neval,
     *   ier,alist,blist,rlist,elist,iord,last)
c***begin prologue  qagse
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a1
c***keywords  automatic integrator, general-purpose,
c             (end point) singularities, extrapolation,
c             globally adaptive
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            definite integral i = integral of f over (a,b),
c            hopefully satisfying following claim for accuracy
c            abs(i-result).le.max(epsabs,epsrel*abs(i)).
c***description
c
c        computation of a definite integral
c        standard fortran subroutine
c        real version
c
c        parameters
c         on entry
c            f      - real
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - real
c                     lower limit of integration
c
c            b      - real
c                     upper limit of integration
c
c            epsabs - real
c                     absolute accuracy requested
c            epsrel - real
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c            limit  - integer
c                     gives an upperbound on the number of subintervals
c                     in the partition of (a,b)
c
c         on return
c            result - real
c                     approximation to the integral
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine
c                             the estimates for integral and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c            error messages
c                         = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more sub-
c                             divisions by increasing the value of limit
c                             (and taking the according dimension
c                             adjustments into account). however, if
c                             this yields no improvement it is advised
c                             to analyze the integrand in order to
c                             determine the integration difficulties. if
c                             the position of a local difficulty can be
c                             determined (e.g. singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling the
c                             integrator on the subranges. if possible,
c                             an appropriate special-purpose integrator
c                             should be used, which is designed for
c                             handling the type of difficulty involved.
c                         = 2 the occurrence of roundoff error is detec-
c                             ted, which prevents the requested
c                             tolerance from being achieved.
c                             the error may be under-estimated.
c                         = 3 extremely bad integrand behaviour
c                             occurs at some points of the integration
c                             interval.
c                         = 4 the algorithm does not converge.
c                             roundoff error is detected in the
c                             extrapolation table.
c                             it is presumed that the requested
c                             tolerance cannot be achieved, and that the
c                             returned result is the best which can be
c                             obtained.
c                         = 5 the integral is probably divergent, or
c                             slowly convergent. it must be noted that
c                             divergence can occur with any other value
c                             of ier.
c                         = 6 the input is invalid, because
c                             epsabs.le.0 and
c                             epsrel.lt.max(50*rel.mach.acc.,0.5d-28).
c                             result, abserr, neval, last, rlist(1),
c                             iord(1) and elist(1) are set to zero.
c                             alist(1) and blist(1) are set to a and b
c                             respectively.
c
c            alist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the left end points
c                     of the subintervals in the partition of the
c                     given integration range (a,b)
c
c            blist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the right end points
c                     of the subintervals in the partition of the given
c                     integration range (a,b)
c
c            rlist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the integral
c                     approximations on the subintervals
c
c            elist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the moduli of the
c                     absolute error estimates on the subintervals
c
c            iord   - integer
c                     vector of dimension at least limit, the first k
c                     elements of which are pointers to the
c                     error estimates over the subintervals,
c                     such that elist(iord(1)), ..., elist(iord(k))
c                     form a decreasing sequence, with k = last
c                     if last.le.(limit/2+2), and k = limit+1-last
c                     otherwise
c
c            last   - integer
c                     number of subintervals actually produced in the
c                     subdivision process
c
c***references  (none)
c***routines called  qelg,qk21,qpsrt,r1mach
c***end prologue  qagse
c
      real a,abseps,abserr,alist,area,area1,area12,area2,a1,
     *  a2,b,blist,b1,b2,correc,defabs,defab1,defab2,r1mach,
     *  dres,elist,epmach,epsabs,epsrel,erlarg,erlast,errbnd,
     *  errmax,error1,error2,erro12,errsum,ertest,f,oflow,resabs,
     *  reseps,result,res3la,rlist,rlist2,small,uflow
      integer id,ier,ierro,iord,iroff1,iroff2,iroff3,jupbnd,k,ksgn,
     *  ktmin,last,limit,maxerr,neval,nres,nrmax,numrl2
      logical extrap,noext
c
      dimension alist(limit),blist(limit),elist(limit),iord(limit),
     * res3la(3),rlist(limit),rlist2(52)
c
      external f
c
c            the dimension of rlist2 is determined by the value of
c            limexp in subroutine qelg (rlist2 should be of dimension
c            (limexp+2) at least).
c
c            list of major variables
c            -----------------------
c
c           alist     - list of left end points of all subintervals
c                       considered up to now
c           blist     - list of right end points of all subintervals
c                       considered up to now
c           rlist(i)  - approximation to the integral over
c                       (alist(i),blist(i))
c           rlist2    - array of dimension at least limexp+2
c                       containing the part of the epsilon table
c                       which is still needed for further computations
c           elist(i)  - error estimate applying to rlist(i)
c           maxerr    - pointer to the interval with largest error
c                       estimate
c           errmax    - elist(maxerr)
c           erlast    - error on the interval currently subdivided
c                       (before that subdivision has taken place)
c           area      - sum of the integrals over the subintervals
c           errsum    - sum of the errors over the subintervals
c           errbnd    - requested accuracy max(epsabs,epsrel*
c                       abs(result))
c           *****1    - variable for the left interval
c           *****2    - variable for the right interval
c           last      - index for subdivision
c           nres      - number of calls to the extrapolation routine
c           numrl2    - number of elements currently in rlist2. if an
c                       appropriate approximation to the compounded
c                       integral has been obtained it is put in
c                       rlist2(numrl2) after numrl2 has been increased
c                       by one.
c           small     - length of the smallest interval considered
c                       up to now, multiplied by 1.5
c           erlarg    - sum of the errors over the intervals larger
c                       than the smallest interval considered up to now
c           extrap    - logical variable denoting that the routine
c                       is attempting to perform extrapolation
c                       i.e. before subdividing the smallest interval
c                       we try to decrease the value of erlarg.
c           noext     - logical variable denoting that extrapolation
c                       is no longer allowed (true value)
c
c            machine dependent constants
c            ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c           oflow is the largest positive magnitude.
c
c***first executable statement  qagse
      epmach = r1mach(4)
c
c            test on validity of parameters
c            ------------------------------
      ier = 0
      neval = 0
      last = 0
      result = 0.0e+00
      abserr = 0.0e+00
      alist(1) = a
      blist(1) = b
      rlist(1) = 0.0e+00
      elist(1) = 0.0e+00
      if(epsabs.le.0.0e+00.and.epsrel.lt.amax1(0.5e+02*epmach,0.5e-14))
     *   ier = 6
      if(ier.eq.6) go to 999
c
c           first approximation to the integral
c           -----------------------------------
c
      uflow = r1mach(1)
      oflow = r1mach(2)
      ierro = 0
      call qk21(f,a,b,result,abserr,defabs,resabs)
c
c           test on accuracy.
c
      dres = abs(result)
      errbnd = amax1(epsabs,epsrel*dres)
      last = 1
      rlist(1) = result
      elist(1) = abserr
      iord(1) = 1
      if(abserr.le.1.0e+02*epmach*defabs.and.abserr.gt.
     *  errbnd) ier = 2
      if(limit.eq.1) ier = 1
      if(ier.ne.0.or.(abserr.le.errbnd.and.abserr.ne.resabs).or.
     *  abserr.eq.0.0e+00) go to 140
c
c           initialization
c           --------------
c
      rlist2(1) = result
      errmax = abserr
      maxerr = 1
      area = result
      errsum = abserr
      abserr = oflow
      nrmax = 1
      nres = 0
      numrl2 = 2
      ktmin = 0
      extrap = .false.
      noext = .false.
      iroff1 = 0
      iroff2 = 0
      iroff3 = 0
      ksgn = -1
      if(dres.ge.(0.1e+01-0.5e+02*epmach)*defabs) ksgn = 1
c
c           main do-loop
c           ------------
c
      do 90 last = 2,limit
c
c           bisect the subinterval with the nrmax-th largest
c           error estimate.
c
        a1 = alist(maxerr)
        b1 = 0.5e+00*(alist(maxerr)+blist(maxerr))
        a2 = b1
        b2 = blist(maxerr)
        erlast = errmax
        call qk21(f,a1,b1,area1,error1,resabs,defab1)
        call qk21(f,a2,b2,area2,error2,resabs,defab2)
c
c           improve previous approximations to integral
c           and error and test for accuracy.
c
        area12 = area1+area2
        erro12 = error1+error2
        errsum = errsum+erro12-errmax
        area = area+area12-rlist(maxerr)
        if(defab1.eq.error1.or.defab2.eq.error2) go to 15
        if(abs(rlist(maxerr)-area12).gt.0.1e-04*abs(area12)
     *  .or.erro12.lt.0.99e+00*errmax) go to 10
        if(extrap) iroff2 = iroff2+1
        if(.not.extrap) iroff1 = iroff1+1
   10   if(last.gt.10.and.erro12.gt.errmax) iroff3 = iroff3+1
   15   rlist(maxerr) = area1
        rlist(last) = area2
        errbnd = amax1(epsabs,epsrel*abs(area))
c
c           test for roundoff error and eventually
c           set error flag.
c
        if(iroff1+iroff2.ge.10.or.iroff3.ge.20) ier = 2
        if(iroff2.ge.5) ierro = 3
c
c           set error flag in the case that the number of
c           subintervals equals limit.
c
        if(last.eq.limit) ier = 1
c
c           set error flag in the case of bad integrand behaviour
c           at a point of the integration range.
c
        if(amax1(abs(a1),abs(b2)).le.(0.1e+01+0.1e+03*epmach)*
     *  (abs(a2)+0.1e+04*uflow)) ier = 4
c
c           append the newly-created intervals to the list.
c
        if(error2.gt.error1) go to 20
        alist(last) = a2
        blist(maxerr) = b1
        blist(last) = b2
        elist(maxerr) = error1
        elist(last) = error2
        go to 30
   20   alist(maxerr) = a2
        alist(last) = a1
        blist(last) = b1
        rlist(maxerr) = area2
        rlist(last) = area1
        elist(maxerr) = error2
        elist(last) = error1
c
c           call subroutine qpsrt to maintain the descending ordering
c           in the list of error estimates and select the
c           subinterval with nrmax-th largest error estimate (to be
c           bisected next).
c
   30   call qpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
c ***jump out of do-loop
        if(errsum.le.errbnd) go to 115
c ***jump out of do-loop
        if(ier.ne.0) go to 100
        if(last.eq.2) go to 80
        if(noext) go to 90
        erlarg = erlarg-erlast
        if(abs(b1-a1).gt.small) erlarg = erlarg+erro12
        if(extrap) go to 40
c
c           test whether the interval to be bisected next is the
c           smallest interval.
c
        if(abs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
        extrap = .true.
        nrmax = 2
   40   if(ierro.eq.3.or.erlarg.le.ertest) go to 60
c
c           the smallest interval has the largest error.
c           before bisecting decrease the sum of the errors
c           over the larger intervals (erlarg) and perform
c           extrapolation.
c
        id = nrmax
        jupbnd = last
        if(last.gt.(2+limit/2)) jupbnd = limit+3-last
        do 50 k = id,jupbnd
          maxerr = iord(nrmax)
          errmax = elist(maxerr)
c ***jump out of do-loop
          if(abs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
          nrmax = nrmax+1
   50   continue
c
c           perform extrapolation.
c
   60   numrl2 = numrl2+1
        rlist2(numrl2) = area
        call qelg(numrl2,rlist2,reseps,abseps,res3la,nres)
        ktmin = ktmin+1
        if(ktmin.gt.5.and.abserr.lt.0.1e-02*errsum) ier = 5
        if(abseps.ge.abserr) go to 70
        ktmin = 0
        abserr = abseps
        result = reseps
        correc = erlarg
        ertest = amax1(epsabs,epsrel*abs(reseps))
c ***jump out of do-loop
        if(abserr.le.ertest) go to 100
c
c           prepare bisection of the smallest interval.
c
   70   if(numrl2.eq.1) noext = .true.
        if(ier.eq.5) go to 100
        maxerr = iord(1)
        errmax = elist(maxerr)
        nrmax = 1
        extrap = .false.
        small = small*0.5e+00
        erlarg = errsum
        go to 90
   80   small = abs(b-a)*0.375e+00
        erlarg = errsum
        ertest = errbnd
        rlist2(2) = area
   90 continue
c
c           set final result and error estimate.
c           ------------------------------------
c
  100 if(abserr.eq.oflow) go to 115
      if(ier+ierro.eq.0) go to 110
      if(ierro.eq.3) abserr = abserr+correc
      if(ier.eq.0) ier = 3
      if(result.ne.0.0e+00.and.area.ne.0.0e+00) go to 105
      if(abserr.gt.errsum) go to 115
      if(area.eq.0.0e+00) go to 130
      go to 110
  105 if(abserr/abs(result).gt.errsum/abs(area)) go to 115
c
c           test on divergence.
c
  110 if(ksgn.eq.(-1).and.amax1(abs(result),abs(area)).le.
     * defabs*0.1e-01) go to 130
      if(0.1e-01.gt.(result/area).or.(result/area).gt.0.1e+03
     * .or.errsum.gt.abs(area)) ier = 6
      go to 130
c
c           compute global integral sum.
c
  115 result = 0.0e+00
      do 120 k = 1,last
         result = result+rlist(k)
  120 continue
      abserr = errsum
  130 if(ier.gt.2) ier = ier-1
  140 neval = 42*last-21
  999 return
      end
      subroutine qelg(n,epstab,result,abserr,res3la,nres)
c***begin prologue  qelg
c***refer to  qagie,qagoe,qagpe,qagse
c***routines called  r1mach
c***revision date  830518   (yymmdd)
c***keywords  epsilon algorithm, convergence acceleration,
c             extrapolation
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math & progr. div. - k.u.leuven
c***purpose  the routine determines the limit of a given sequence of
c            approximations, by means of the epsilon algorithm of
c            p. wynn. an estimate of the absolute error is also given.
c            the condensed epsilon table is computed. only those
c            elements needed for the computation of the next diagonal
c            are preserved.
c***description
c
c           epsilon algorithm
c           standard fortran subroutine
c           real version
c
c           parameters
c              n      - integer
c                       epstab(n) contains the new element in the
c                       first column of the epsilon table.
c
c              epstab - real
c                       vector of dimension 52 containing the elements
c                       of the two lower diagonals of the triangular
c                       epsilon table. the elements are numbered
c                       starting at the right-hand corner of the
c                       triangle.
c
c              result - real
c                       resulting approximation to the integral
c
c              abserr - real
c                       estimate of the absolute error computed from
c                       result and the 3 previous results
c
c              res3la - real
c                       vector of dimension 3 containing the last 3
c                       results
c
c              nres   - integer
c                       number of calls to the routine
c                       (should be zero at first call)
c
c***end prologue  qelg
c
      real abserr,delta1,delta2,delta3,r1mach,
     *  epmach,epsinf,epstab,error,err1,err2,err3,e0,e1,e1abs,e2,e3,
     *  oflow,res,result,res3la,ss,tol1,tol2,tol3
      integer i,ib,ib2,ie,indx,k1,k2,k3,limexp,n,newelm,nres,num
      dimension epstab(52),res3la(3)
c
c           list of major variables
c           -----------------------
c
c           e0     - the 4 elements on which the
c           e1       computation of a new element in
c           e2       the epsilon table is based
c           e3                 e0
c                        e3    e1    new
c                              e2
c           newelm - number of elements to be computed in the new
c                    diagonal
c           error  - error = abs(e1-e0)+abs(e2-e1)+abs(new-e2)
c           result - the element in the new diagonal with least value
c                    of error
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           oflow is the largest positive magnitude.
c           limexp is the maximum number of elements the epsilon
c           table can contain. if this number is reached, the upper
c           diagonal of the epsilon table is deleted.
c
c***first executable statement  qelg
      epmach = r1mach(4)
      oflow = r1mach(2)
      nres = nres+1
      abserr = oflow
      result = epstab(n)
      if(n.lt.3) go to 100
      limexp = 50
      epstab(n+2) = epstab(n)
      newelm = (n-1)/2
      epstab(n) = oflow
      num = n
      k1 = n
      do 40 i = 1,newelm
        k2 = k1-1
        k3 = k1-2
        res = epstab(k1+2)
        e0 = epstab(k3)
        e1 = epstab(k2)
        e2 = res
        e1abs = abs(e1)
        delta2 = e2-e1
        err2 = abs(delta2)
        tol2 = amax1(abs(e2),e1abs)*epmach
        delta3 = e1-e0
        err3 = abs(delta3)
        tol3 = amax1(e1abs,abs(e0))*epmach
        if(err2.gt.tol2.or.err3.gt.tol3) go to 10
c
c           if e0, e1 and e2 are equal to within machine
c           accuracy, convergence is assumed.
c           result = e2
c           abserr = abs(e1-e0)+abs(e2-e1)
c
        result = res
        abserr = err2+err3
c ***jump out of do-loop
        go to 100
   10   e3 = epstab(k1)
        epstab(k1) = e1
        delta1 = e1-e3
        err1 = abs(delta1)
        tol1 = amax1(e1abs,abs(e3))*epmach
c
c           if two elements are very close to each other, omit
c           a part of the table by adjusting the value of n
c
        if(err1.le.tol1.or.err2.le.tol2.or.err3.le.tol3) go to 20
        ss = 0.1e+01/delta1+0.1e+01/delta2-0.1e+01/delta3
        epsinf = abs(ss*e1)
c
c           test to detect irregular behaviour in the table, and
c           eventually omit a part of the table adjusting the value
c           of n.
c
        if(epsinf.gt.0.1e-03) go to 30
   20   n = i+i-1
c ***jump out of do-loop
        go to 50
c
c           compute a new element and eventually adjust
c           the value of result.
c
   30   res = e1+0.1e+01/ss
        epstab(k1) = res
        k1 = k1-2
        error = err2+abs(res-e2)+err3
        if(error.gt.abserr) go to 40
        abserr = error
        result = res
   40 continue
c
c           shift the table.
c
   50 if(n.eq.limexp) n = 2*(limexp/2)-1
      ib = 1
      if((num/2)*2.eq.num) ib = 2
      ie = newelm+1
      do 60 i=1,ie
        ib2 = ib+2
        epstab(ib) = epstab(ib2)
        ib = ib2
   60 continue
      if(num.eq.n) go to 80
      indx = num-n+1
      do 70 i = 1,n
        epstab(i)= epstab(indx)
        indx = indx+1
   70 continue
   80 if(nres.ge.4) go to 90
      res3la(nres) = result
      abserr = oflow
      go to 100
c
c           compute error estimate
c
   90 abserr = abs(result-res3la(3))+abs(result-res3la(2))
     *  +abs(result-res3la(1))
      res3la(1) = res3la(2)
      res3la(2) = res3la(3)
      res3la(3) = result
  100 abserr = amax1(abserr,0.5e+01*epmach*abs(result))
      return
      end
      subroutine qk21(f,a,b,result,abserr,resabs,resasc)
c***begin prologue  qk21
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  21-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b), with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           real version
c
c           parameters
c            on entry
c              f      - real
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the driver program.
c
c              a      - real
c                       lower limit of integration
c
c              b      - real
c                       upper limit of integration
c
c            on return
c              result - real
c                       approximation to the integral i
c                       result is computed by applying the 21-point
c                       kronrod rule (resk) obtained by optimal addition
c                       of abscissae to the 10-point gauss rule (resg).
c
c              abserr - real
c                       estimate of the modulus of the absolute error,
c                       which should not exceed abs(i-result)
c
c              resabs - real
c                       approximation to the integral j
c
c              resasc - real
c                       approximation to the integral of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  r1mach
c***end prologue  qk21
c
      real a,absc,abserr,b,centr,dhlgth,epmach,f,fc,fsum,fval1,fval2,
     *  fv1,fv2,hlgth,resabs,resg,resk,reskh,result,r1mach,uflow,wg,wgk,
     *  xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(10),fv2(10),wg(5),wgk(11),xgk(11)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 21-point kronrod rule
c                    xgk(2), xgk(4), ...  abscissae of the 10-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 10-point gauss rule
c
c           wgk    - weights of the 21-point kronrod rule
c
c           wg     - weights of the 10-point gauss rule
c
      data xgk(1),xgk(2),xgk(3),xgk(4),xgk(5),xgk(6),xgk(7),
     *  xgk(8),xgk(9),xgk(10),xgk(11)/
     *         0.9956571630258081e+00,     0.9739065285171717e+00,
     *     0.9301574913557082e+00,     0.8650633666889845e+00,
     *     0.7808177265864169e+00,     0.6794095682990244e+00,
     *     0.5627571346686047e+00,     0.4333953941292472e+00,
     *     0.2943928627014602e+00,     0.1488743389816312e+00,
     *     0.0000000000000000e+00/
c
      data wgk(1),wgk(2),wgk(3),wgk(4),wgk(5),wgk(6),wgk(7),
     *  wgk(8),wgk(9),wgk(10),wgk(11)/
     *     0.1169463886737187e-01,     0.3255816230796473e-01,
     *     0.5475589657435200e-01,     0.7503967481091995e-01,
     *     0.9312545458369761e-01,     0.1093871588022976e+00,
     *     0.1234919762620659e+00,     0.1347092173114733e+00,
     *     0.1427759385770601e+00,     0.1477391049013385e+00,
     *     0.1494455540029169e+00/
c
      data wg(1),wg(2),wg(3),wg(4),wg(5)/
     *     0.6667134430868814e-01,     0.1494513491505806e+00,
     *     0.2190863625159820e+00,     0.2692667193099964e+00,
     *     0.2955242247147529e+00/
c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 10-point gauss formula
c           resk   - result of the 21-point kronrod formula
c           reskh  - approximation to the mean value of f over (a,b),
c                    i.e. to i/(b-a)
c
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  qk21
      epmach = r1mach(4)
      uflow = r1mach(1)
c
      centr = 0.5e+00*(a+b)
      hlgth = 0.5e+00*(b-a)
      dhlgth = abs(hlgth)
c
c           compute the 21-point kronrod approximation to
c           the integral, and estimate the absolute error.
c
      resg = 0.0e+00
      fc = f(centr)
      resk = wgk(11)*fc
      resabs = abs(resk)
      do 10 j=1,5
        jtw = 2*j
        absc = hlgth*xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(abs(fval1)+abs(fval2))
   10 continue
      do 15 j = 1,5
        jtwm1 = 2*j-1
        absc = hlgth*xgk(jtwm1)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(abs(fval1)+abs(fval2))
   15 continue
      reskh = resk*0.5e+00
      resasc = wgk(11)*abs(fc-reskh)
      do 20 j=1,10
        resasc = resasc+wgk(j)*(abs(fv1(j)-reskh)+abs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = abs((resk-resg)*hlgth)
      if(resasc.ne.0.0e+00.and.abserr.ne.0.0e+00)
     *  abserr = resasc*amin1(0.1e+01,
     *  (0.2e+03*abserr/resasc)**1.5e+00)
      if(resabs.gt.uflow/(0.5e+02*epmach)) abserr = amax1
     *  ((epmach*0.5e+02)*resabs,abserr)
      return
      end
      subroutine qpsrt(limit,last,maxerr,ermax,elist,iord,nrmax)
c***begin prologue  qpsrt
c***refer to  qage,qagie,qagpe,qagse,qawce,qawse,qawoe
c***routines called  (none)
c***keywords  sequential sorting
c***description
c
c 1.        qpsrt
c           ordering routine
c              standard fortran subroutine
c              real version
c
c 2.        purpose
c              this routine maintains the descending ordering
c              in the list of the local error estimates resulting from
c              the interval subdivision process. at each call two error
c              estimates are inserted using the sequential search
c              method, top-down for the largest error estimate
c              and bottom-up for the smallest error estimate.
c
c 3.        calling sequence
c              call qpsrt(limit,last,maxerr,ermax,elist,iord,nrmax)
c
c           parameters (meaning at output)
c              limit  - integer
c                       maximum number of error estimates the list
c                       can contain
c
c              last   - integer
c                       number of error estimates currently
c                       in the list
c
c              maxerr - integer
c                       maxerr points to the nrmax-th largest error
c                       estimate currently in the list
c
c              ermax  - real
c                       nrmax-th largest error estimate
c                       ermax = elist(maxerr)
c
c              elist  - real
c                       vector of dimension last containing
c                       the error estimates
c
c              iord   - integer
c                       vector of dimension last, the first k
c                       elements of which contain pointers
c                       to the error estimates, such that
c                       elist(iord(1)),... , elist(iord(k))
c                       form a decreasing sequence, with
c                       k = last if last.le.(limit/2+2), and
c                       k = limit+1-last otherwise
c
c              nrmax  - integer
c                       maxerr = iord(nrmax)
c
c 4.        no subroutines or functions needed
c***end prologue  qpsrt
c
      real elist,ermax,errmax,errmin
      integer i,ibeg,ido,iord,isucc,j,jbnd,jupbn,k,last,limit,maxerr,
     *  nrmax
      dimension elist(last),iord(last)
c
c           check whether the list contains more than
c           two error estimates.
c
c***first executable statement  qpsrt
      if(last.gt.2) go to 10
      iord(1) = 1
      iord(2) = 2
      go to 90
c
c           this part of the routine is only executed
c           if, due to a difficult integrand, subdivision
c           increased the error estimate. in the normal case
c           the insert procedure should start after the
c           nrmax-th largest error estimate.
c
   10 errmax = elist(maxerr)
      if(nrmax.eq.1) go to 30
      ido = nrmax-1
      do 20 i = 1,ido
        isucc = iord(nrmax-1)
c ***jump out of do-loop
        if(errmax.le.elist(isucc)) go to 30
        iord(nrmax) = isucc
        nrmax = nrmax-1
   20    continue
c
c           compute the number of elements in the list to
c           be maintained in descending order. this number
c           depends on the number of subdivisions still
c           allowed.
c
   30 jupbn = last
      if(last.gt.(limit/2+2)) jupbn = limit+3-last
      errmin = elist(last)
c
c           insert errmax by traversing the list top-down,
c           starting comparison from the element elist(iord(nrmax+1)).
c
      jbnd = jupbn-1
      ibeg = nrmax+1
      if(ibeg.gt.jbnd) go to 50
      do 40 i=ibeg,jbnd
        isucc = iord(i)
c ***jump out of do-loop
        if(errmax.ge.elist(isucc)) go to 60
        iord(i-1) = isucc
   40 continue
   50 iord(jbnd) = maxerr
      iord(jupbn) = last
      go to 90
c
c           insert errmin by traversing the list bottom-up.
c
   60 iord(i-1) = maxerr
      k = jbnd
      do 70 j=i,jbnd
        isucc = iord(k)
c ***jump out of do-loop
        if(errmin.lt.elist(isucc)) go to 80
        iord(k+1) = isucc
        k = k-1
   70 continue
      iord(i) = last
      go to 90
   80 iord(k+1) = last
c
c           set maxerr and ermax.
c
   90 maxerr = iord(nrmax)
      ermax = elist(maxerr)
      return
      end
      REAL FUNCTION R1MACH(I)
C
C  SINGLE-PRECISION MACHINE CONSTANTS
C
C  R1MACH(1) = B**(EMIN-1), THE SMALLEST POSITIVE MAGNITUDE.
C
C  R1MACH(2) = B**EMAX*(1 - B**(-T)), THE LARGEST MAGNITUDE.
C
C  R1MACH(3) = B**(-T), THE SMALLEST RELATIVE SPACING.
C
C  R1MACH(4) = B**(1-T), THE LARGEST RELATIVE SPACING.
C
C  R1MACH(5) = LOG10(B)
C
C  TO ALTER THIS FUNCTION FOR A PARTICULAR ENVIRONMENT,
C  THE DESIRED SET OF DATA STATEMENTS SHOULD BE ACTIVATED BY
C  REMOVING THE C FROM COLUMN 1.
C
C  FOR IEEE-ARITHMETIC MACHINES (BINARY STANDARD), THE FIRST
C  SET OF CONSTANTS BELOW SHOULD BE APPROPRIATE.
C
C  WHERE POSSIBLE, DECIMAL, OCTAL OR HEXADECIMAL CONSTANTS ARE USED
C  TO SPECIFY THE CONSTANTS EXACTLY.  SOMETIMES THIS REQUIRES USING
C  EQUIVALENT INTEGER ARRAYS.  IF YOUR COMPILER USES HALF-WORD
C  INTEGERS BY DEFAULT (SOMETIMES CALLED INTEGER*2), YOU MAY NEED TO
C  CHANGE INTEGER TO INTEGER*4 OR OTHERWISE INSTRUCT YOUR COMPILER
C  TO USE FULL-WORD INTEGERS IN THE NEXT 5 DECLARATIONS.
C
C  COMMENTS JUST BEFORE THE END STATEMENT (LINES STARTING WITH *)
C  GIVE C SOURCE FOR R1MACH.
C
      INTEGER SMALL(2)
      INTEGER LARGE(2)
      INTEGER RIGHT(2)
      INTEGER DIVER(2)
      INTEGER LOG10(2)
      INTEGER SC
C/6S
C/7S
      SAVE SMALL, LARGE, RIGHT, DIVER, LOG10, SC
C/
      REAL RMACH(5)
C
      EQUIVALENCE (RMACH(1),SMALL(1))
      EQUIVALENCE (RMACH(2),LARGE(1))
      EQUIVALENCE (RMACH(3),RIGHT(1))
      EQUIVALENCE (RMACH(4),DIVER(1))
      EQUIVALENCE (RMACH(5),LOG10(1))
C
C     MACHINE CONSTANTS FOR IEEE ARITHMETIC MACHINES, SUCH AS THE AT&T
C     3B SERIES, MOTOROLA 68000 BASED MACHINES (E.G. SUN 3 AND AT&T
C     PC 7300), AND 8087 BASED MICROS (E.G. IBM PC AND AT&T 6300).
C
C      DATA SMALL(1) /     8388608 /
C      DATA LARGE(1) /  2139095039 /
C      DATA RIGHT(1) /   864026624 /
C      DATA DIVER(1) /   872415232 /
C      DATA LOG10(1) /  1050288283 /, SC/987/
C
C     MACHINE CONSTANTS FOR AMDAHL MACHINES.
C
C      DATA SMALL(1) /    1048576 /
C      DATA LARGE(1) / 2147483647 /
C      DATA RIGHT(1) /  990904320 /
C      DATA DIVER(1) / 1007681536 /
C      DATA LOG10(1) / 1091781651 /, SC/987/
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 1700 SYSTEM.
C
C      DATA RMACH(1) / Z400800000 /
C      DATA RMACH(2) / Z5FFFFFFFF /
C      DATA RMACH(3) / Z4E9800000 /
C      DATA RMACH(4) / Z4EA800000 /
C      DATA RMACH(5) / Z500E730E8 /, SC/987/
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 5700/6700/7700 SYSTEMS.
C
C      DATA RMACH(1) / O1771000000000000 /
C      DATA RMACH(2) / O0777777777777777 /
C      DATA RMACH(3) / O1311000000000000 /
C      DATA RMACH(4) / O1301000000000000 /
C      DATA RMACH(5) / O1157163034761675 /, SC/987/
C
C     MACHINE CONSTANTS FOR FTN4 ON THE CDC 6000/7000 SERIES.
C
C      DATA RMACH(1) / 00564000000000000000B /
C      DATA RMACH(2) / 37767777777777777776B /
C      DATA RMACH(3) / 16414000000000000000B /
C      DATA RMACH(4) / 16424000000000000000B /
C      DATA RMACH(5) / 17164642023241175720B /, SC/987/
C
C     MACHINE CONSTANTS FOR FTN5 ON THE CDC 6000/7000 SERIES.
C
C      DATA RMACH(1) / O"00564000000000000000" /
C      DATA RMACH(2) / O"37767777777777777776" /
C      DATA RMACH(3) / O"16414000000000000000" /
C      DATA RMACH(4) / O"16424000000000000000" /
C      DATA RMACH(5) / O"17164642023241175720" /, SC/987/
C
C     MACHINE CONSTANTS FOR CONVEX C-1.
C
C      DATA RMACH(1) / '00800000'X /
C      DATA RMACH(2) / '7FFFFFFF'X /
C      DATA RMACH(3) / '34800000'X /
C      DATA RMACH(4) / '35000000'X /
C      DATA RMACH(5) / '3F9A209B'X /, SC/987/
C
C     MACHINE CONSTANTS FOR THE CRAY 1, XMP, 2, AND 3.
C
C      DATA RMACH(1) / 200034000000000000000B /
C      DATA RMACH(2) / 577767777777777777776B /
C      DATA RMACH(3) / 377224000000000000000B /
C      DATA RMACH(4) / 377234000000000000000B /
C      DATA RMACH(5) / 377774642023241175720B /, SC/987/
C
C     MACHINE CONSTANTS FOR THE DATA GENERAL ECLIPSE S/200.
C
C     NOTE - IT MAY BE APPROPRIATE TO INCLUDE THE FOLLOWING LINE -
C     STATIC RMACH(5)
C
C      DATA SMALL/20K,0/,LARGE/77777K,177777K/
C      DATA RIGHT/35420K,0/,DIVER/36020K,0/
C      DATA LOG10/40423K,42023K/, SC/987/
C
C     MACHINE CONSTANTS FOR THE HARRIS SLASH 6 AND SLASH 7.
C
C      DATA SMALL(1),SMALL(2) / '20000000, '00000201 /
C      DATA LARGE(1),LARGE(2) / '37777777, '00000177 /
C      DATA RIGHT(1),RIGHT(2) / '20000000, '00000352 /
C      DATA DIVER(1),DIVER(2) / '20000000, '00000353 /
C      DATA LOG10(1),LOG10(2) / '23210115, '00000377 /, SC/987/
C
C     MACHINE CONSTANTS FOR THE HONEYWELL DPS 8/70 SERIES.
C
C      DATA RMACH(1) / O402400000000 /
C      DATA RMACH(2) / O376777777777 /
C      DATA RMACH(3) / O714400000000 /
C      DATA RMACH(4) / O716400000000 /
C      DATA RMACH(5) / O776464202324 /, SC/987/
C
C     MACHINE CONSTANTS FOR THE IBM 360/370 SERIES,
C     THE XEROX SIGMA 5/7/9 AND THE SEL SYSTEMS 85/86.
C
C      DATA RMACH(1) / Z00100000 /
C      DATA RMACH(2) / Z7FFFFFFF /
C      DATA RMACH(3) / Z3B100000 /
C      DATA RMACH(4) / Z3C100000 /
C      DATA RMACH(5) / Z41134413 /, SC/987/
C
C     MACHINE CONSTANTS FOR THE INTERDATA 8/32
C     WITH THE UNIX SYSTEM FORTRAN 77 COMPILER.
C
C     FOR THE INTERDATA FORTRAN VII COMPILER REPLACE
C     THE Z'S SPECIFYING HEX CONSTANTS WITH Y'S.
C
C      DATA RMACH(1) / Z'00100000' /
C      DATA RMACH(2) / Z'7EFFFFFF' /
C      DATA RMACH(3) / Z'3B100000' /
C      DATA RMACH(4) / Z'3C100000' /
C      DATA RMACH(5) / Z'41134413' /, SC/987/
C
C     MACHINE CONSTANTS FOR THE PDP-10 (KA OR KI PROCESSOR).
C
C      DATA RMACH(1) / "000400000000 /
C      DATA RMACH(2) / "377777777777 /
C      DATA RMACH(3) / "146400000000 /
C      DATA RMACH(4) / "147400000000 /
C      DATA RMACH(5) / "177464202324 /, SC/987/
C
C     MACHINE CONSTANTS FOR PDP-11 FORTRANS SUPPORTING
C     32-BIT INTEGERS (EXPRESSED IN INTEGER AND OCTAL).
C
C      DATA SMALL(1) /    8388608 /
C      DATA LARGE(1) / 2147483647 /
C      DATA RIGHT(1) /  880803840 /
C      DATA DIVER(1) /  889192448 /
C      DATA LOG10(1) / 1067065499 /, SC/987/
C
C      DATA RMACH(1) / O00040000000 /
C      DATA RMACH(2) / O17777777777 /
C      DATA RMACH(3) / O06440000000 /
C      DATA RMACH(4) / O06500000000 /
C      DATA RMACH(5) / O07746420233 /, SC/987/
C
C     MACHINE CONSTANTS FOR PDP-11 FORTRANS SUPPORTING
C     16-BIT INTEGERS  (EXPRESSED IN INTEGER AND OCTAL).
C
C      DATA SMALL(1),SMALL(2) /   128,     0 /
C      DATA LARGE(1),LARGE(2) / 32767,    -1 /
C      DATA RIGHT(1),RIGHT(2) / 13440,     0 /
C      DATA DIVER(1),DIVER(2) / 13568,     0 /
C      DATA LOG10(1),LOG10(2) / 16282,  8347 /, SC/987/
C
C      DATA SMALL(1),SMALL(2) / O000200, O000000 /
C      DATA LARGE(1),LARGE(2) / O077777, O177777 /
C      DATA RIGHT(1),RIGHT(2) / O032200, O000000 /
C      DATA DIVER(1),DIVER(2) / O032400, O000000 /
C      DATA LOG10(1),LOG10(2) / O037632, O020233 /, SC/987/
C
C     MACHINE CONSTANTS FOR THE SEQUENT BALANCE 8000.
C
C      DATA SMALL(1) / $00800000 /
C      DATA LARGE(1) / $7F7FFFFF /
C      DATA RIGHT(1) / $33800000 /
C      DATA DIVER(1) / $34000000 /
C      DATA LOG10(1) / $3E9A209B /, SC/987/
C
C     MACHINE CONSTANTS FOR THE UNIVAC 1100 SERIES.
C
C      DATA RMACH(1) / O000400000000 /
C      DATA RMACH(2) / O377777777777 /
C      DATA RMACH(3) / O146400000000 /
C      DATA RMACH(4) / O147400000000 /
C      DATA RMACH(5) / O177464202324 /, SC/987/
C
C     MACHINE CONSTANTS FOR THE VAX UNIX F77 COMPILER.
C
      DATA SMALL(1) /       128 /
      DATA LARGE(1) /    -32769 /
      DATA RIGHT(1) /     13440 /
      DATA DIVER(1) /     13568 /
      DATA LOG10(1) / 547045274 /, SC/987/
C
C     MACHINE CONSTANTS FOR THE VAX-11 WITH
C     FORTRAN IV-PLUS COMPILER.
C
C      DATA RMACH(1) / Z00000080 /
C      DATA RMACH(2) / ZFFFF7FFF /
C      DATA RMACH(3) / Z00003480 /
C      DATA RMACH(4) / Z00003500 /
C      DATA RMACH(5) / Z209B3F9A /, SC/987/
C
C     MACHINE CONSTANTS FOR VAX/VMS VERSION 2.2.
C
C      DATA RMACH(1) /       '80'X /
C      DATA RMACH(2) / 'FFFF7FFF'X /
C      DATA RMACH(3) /     '3480'X /
C      DATA RMACH(4) /     '3500'X /
C      DATA RMACH(5) / '209B3F9A'X /, SC/987/
C
C  ***  ISSUE STOP 777 IF ALL DATA STATEMENTS ARE COMMENTED...
      IF (SC .NE. 987) THEN
*        *** CHECK FOR AUTODOUBLE ***
         SMALL(2) = 0
         RMACH(1) = 1E13
         IF (SMALL(2) .NE. 0) THEN
*           *** AUTODOUBLED ***
            IF (      SMALL(1) .EQ. 1117925532
     *          .AND. SMALL(2) .EQ. -448790528) THEN
*              *** IEEE BIG ENDIAN ***
               SMALL(1) = 1048576
               SMALL(2) = 0
               LARGE(1) = 2146435071
               LARGE(2) = -1
               RIGHT(1) = 1017118720
               RIGHT(2) = 0
               DIVER(1) = 1018167296
               DIVER(2) = 0
               LOG10(1) = 1070810131
               LOG10(2) = 1352628735
            ELSE IF ( SMALL(2) .EQ. 1117925532
     *          .AND. SMALL(1) .EQ. -448790528) THEN
*              *** IEEE LITTLE ENDIAN ***
               SMALL(2) = 1048576
               SMALL(1) = 0
               LARGE(2) = 2146435071
               LARGE(1) = -1
               RIGHT(2) = 1017118720
               RIGHT(1) = 0
               DIVER(2) = 1018167296
               DIVER(1) = 0
               LOG10(2) = 1070810131
               LOG10(1) = 1352628735
            ELSE IF ( SMALL(1) .EQ. -2065213935
     *          .AND. SMALL(2) .EQ. 10752) THEN
*              *** VAX WITH D_FLOATING ***
               SMALL(1) = 128
               SMALL(2) = 0
               LARGE(1) = -32769
               LARGE(2) = -1
               RIGHT(1) = 9344
               RIGHT(2) = 0
               DIVER(1) = 9472
               DIVER(2) = 0
               LOG10(1) = 546979738
               LOG10(2) = -805796613
            ELSE IF ( SMALL(1) .EQ. 1267827943
     *          .AND. SMALL(2) .EQ. 704643072) THEN
*              *** IBM MAINFRAME ***
               SMALL(1) = 1048576
               SMALL(2) = 0
               LARGE(1) = 2147483647
               LARGE(2) = -1
               RIGHT(1) = 856686592
               RIGHT(2) = 0
               DIVER(1) = 873463808
               DIVER(2) = 0
               LOG10(1) = 1091781651
               LOG10(2) = 1352628735
            ELSE
               WRITE(*,9010)
               STOP 777
               END IF
         ELSE
            RMACH(1) = 1234567.
            IF (SMALL(1) .EQ. 1234613304) THEN
*              *** IEEE ***
               SMALL(1) = 8388608
               LARGE(1) = 2139095039
               RIGHT(1) = 864026624
               DIVER(1) = 872415232
               LOG10(1) = 1050288283
            ELSE IF (SMALL(1) .EQ. -1271379306) THEN
*              *** VAX ***
               SMALL(1) = 128
               LARGE(1) = -32769
               RIGHT(1) = 13440
               DIVER(1) = 13568
               LOG10(1) = 547045274
            ELSE IF (SMALL(1) .EQ. 1175639687) THEN
*              *** IBM ***
               SMALL(1) = 1048576
               LARGE(1) = 2147483647
               RIGHT(1) = 990904320
               DIVER(1) = 1007681536
               LOG10(1) = 1091781651
            ELSE
               WRITE(*,9020)
               STOP 777
               END IF
            END IF
         SC = 987
         END IF
C
C  ***  ISSUE STOP 776 IF ALL DATA STATEMENTS ARE OBVIOUSLY WRONG...
      IF (RMACH(4) .GE. 1.0) STOP 776
*C/6S
*C     IF (I .LT. 1  .OR.  I .GT. 5)
*C    1   CALL SETERR(24HR1MACH - I OUT OF BOUNDS,24,1,2)
*C/7S
*      IF (I .LT. 1  .OR.  I .GT. 5)
*     1   CALL SETERR('R1MACH - I OUT OF BOUNDS',24,1,2)
*C/
C
      IF (I .LT. 1 .OR. I .GT. 5) THEN
         WRITE(*,*) 'R1MACH(I): I =',I,' is out of bounds.'
         STOP
         END IF
      R1MACH = RMACH(I)
      RETURN
 9010 FORMAT(/42H Adjust autodoubled R1MACH by getting data/
     *42H appropriate for your machine from D1MACH.)
 9020 FORMAT(/46H Adjust R1MACH by uncommenting data statements/
     *30H appropriate for your machine.)
C
* /* C source for R1MACH -- remove the * in column 1 */
*#include <stdio.h>
*#include <float.h>
*#include <math.h>
*
*float r1mach_(long *i)
*{
*	switch(*i){
*	  case 1: return FLT_MIN;
*	  case 2: return FLT_MAX;
*	  case 3: return FLT_EPSILON/FLT_RADIX;
*	  case 4: return FLT_EPSILON;
*	  case 5: return log10(FLT_RADIX);
*	  }
*
*	fprintf(stderr, "invalid argument: r1mach(%ld)\n", *i);
*	exit(1);
*	return 0; /* for compilers that complain of missing return values */
*	}
      END
      SUBROUTINE odeint2(ystart,nvar,x1,x2,eps,h1,hmin,nok,nbad,derivs,
     *rkqs)
      INTEGER nbad,nok,nvar,KMAXX,MAXSTP,NMAX
      REAL eps,h1,hmin,x1,x2,ystart(nvar),TINY
      EXTERNAL derivs,rkqs
      PARAMETER (MAXSTP=10000,NMAX=50,KMAXX=200,TINY=1.e-30)
      INTEGER i,kmax,kount,nstp
      REAL dxsav,h,hdid,hnext,x,xsav,dydx(NMAX),xp(KMAXX),y(NMAX),
     *yp(NMAX,KMAXX),yscal(NMAX)
      COMMON /path/ kmax,kount,dxsav,xp,yp
      x=x1
      h=sign(h1,x2-x1)
      nok=0
      nbad=0
      kount=0
      do 11 i=1,nvar
        y(i)=ystart(i)
11    continue
      if (kmax.gt.0) xsav=x-2.*dxsav
      do 16 nstp=1,MAXSTP
        call derivs(x,y,dydx)
        do 12 i=1,nvar
          yscal(i)=abs(y(i))+abs(h*dydx(i))+TINY
12      continue
        if(kmax.gt.0)then
          if(abs(x-xsav).gt.abs(dxsav)) then
            if(kount.lt.kmax-1)then
              kount=kount+1
              xp(kount)=x
              do 13 i=1,nvar
                yp(i,kount)=y(i)
13            continue
              xsav=x
            endif
          endif
        endif
        if((x+h-x2)*(x+h-x1).gt.0.) h=x2-x
        call rkqs(y,dydx,nvar,x,h,eps,yscal,hdid,hnext,derivs)
        if(hdid.eq.h)then
          nok=nok+1
        else
          nbad=nbad+1
        endif
        if((x-x2)*(x2-x1).ge.0.)then
          do 14 i=1,nvar
            ystart(i)=y(i)
14        continue
          if(kmax.ne.0)then
            kount=kount+1
            xp(kount)=x
            do 15 i=1,nvar
              yp(i,kount)=y(i)
15          continue
          endif
          return
        endif
        if(abs(hnext).lt.hmin) pause
     *'stepsize smaller than minimum in odeint'
        h=hnext
16    continue
      pause 'too many steps in odeint'
      return
      END

      SUBROUTINE bsstep(y,dydx,nv,x,htry,eps,yscal,hdid,hnext,derivs)
      INTEGER nv,NMAX,KMAXX,IMAX
      REAL eps,hdid,hnext,htry,x,dydx(nv),y(nv),yscal(nv),SAFE1,SAFE2,
     *REDMAX,REDMIN,TINY,SCALMX
      PARAMETER (NMAX=50,KMAXX=8,IMAX=KMAXX+1,SAFE1=.25,SAFE2=.7,
     *REDMAX=1.e-5,REDMIN=.7,TINY=1.e-30,SCALMX=.1)
CU    USES derivs,mmid,pzextr
      INTEGER i,iq,k,kk,km,kmax,kopt,nseq(IMAX)
      REAL eps1,epsold,errmax,fact,h,red,scale,work,wrkmin,xest,xnew,
     *a(IMAX),alf(KMAXX,KMAXX),err(KMAXX),yerr(NMAX),ysav(NMAX),
     *yseq(NMAX)
      LOGICAL first,reduct
      SAVE a,alf,epsold,first,kmax,kopt,nseq,xnew
      EXTERNAL derivs
      DATA first/.true./,epsold/-1./
      DATA nseq /2,4,6,8,10,12,14,16,18/
      if(eps.ne.epsold)then
        hnext=-1.e29
        xnew=-1.e29
        eps1=SAFE1*eps
        a(1)=nseq(1)+1
        do 11 k=1,KMAXX
          a(k+1)=a(k)+nseq(k+1)
11      continue
        do 13 iq=2,KMAXX
          do 12 k=1,iq-1
            alf(k,iq)=eps1**((a(k+1)-a(iq+1))/((a(iq+1)-a(1)+1.)*(2*k+
     *1)))
12        continue
13      continue
        epsold=eps
        do 14 kopt=2,KMAXX-1
          if(a(kopt+1).gt.a(kopt)*alf(kopt-1,kopt))goto 1
14      continue
1       kmax=kopt
      endif
      h=htry
      do 15 i=1,nv
        ysav(i)=y(i)
15    continue
      if(h.ne.hnext.or.x.ne.xnew)then
        first=.true.
        kopt=kmax
      endif
      reduct=.false.
2     do 17 k=1,kmax
        xnew=x+h
        if(xnew.eq.x)pause 'step size underflow in bsstep'
        call mmid2(ysav,dydx,nv,x,h,nseq(k),yseq,derivs)
        xest=(h/nseq(k))**2
        call pzextr2(k,xest,yseq,y,yerr,nv)
        if(k.ne.1)then
          errmax=TINY
          do 16 i=1,nv
            errmax=max(errmax,abs(yerr(i)/yscal(i)))
16        continue
          errmax=errmax/eps
          km=k-1
          err(km)=(errmax/SAFE1)**(1./(2*km+1))
        endif
        if(k.ne.1.and.(k.ge.kopt-1.or.first))then
          if(errmax.lt.1.)goto 4
          if(k.eq.kmax.or.k.eq.kopt+1)then
            red=SAFE2/err(km)
            goto 3
          else if(k.eq.kopt)then
            if(alf(kopt-1,kopt).lt.err(km))then
              red=1./err(km)
              goto 3
            endif
          else if(kopt.eq.kmax)then
            if(alf(km,kmax-1).lt.err(km))then
              red=alf(km,kmax-1)*SAFE2/err(km)
              goto 3
            endif
          else if(alf(km,kopt).lt.err(km))then
            red=alf(km,kopt-1)/err(km)
            goto 3
          endif
        endif
17    continue
3     red=min(red,REDMIN)
      red=max(red,REDMAX)
      h=h*red
      reduct=.true.
      goto 2
4     x=xnew
      hdid=h
      first=.false.
      wrkmin=1.e35
      do 18 kk=1,km
        fact=max(err(kk),SCALMX)
        work=fact*a(kk+1)
        if(work.lt.wrkmin)then
          scale=fact
          wrkmin=work
          kopt=kk+1
        endif
18    continue
      hnext=h/scale
      if(kopt.ge.k.and.kopt.ne.kmax.and..not.reduct)then
        fact=max(scale/alf(kopt-1,kopt),SCALMX)
        if(a(kopt+1)*fact.le.wrkmin)then
          hnext=h/fact
          kopt=kopt+1
        endif
      endif
      return
      END
      SUBROUTINE pzextr2(iest,xest,yest,yz,dy,nv)
      INTEGER iest,nv,IMAX,NMAX
      REAL xest,dy(nv),yest(nv),yz(nv)
      PARAMETER (IMAX=13,NMAX=50)
      INTEGER j,k1
      REAL delta,f1,f2,q,d(NMAX),qcol(NMAX,IMAX),x(IMAX)
      SAVE qcol,x
      x(iest)=xest
      do 11 j=1,nv
        dy(j)=yest(j)
        yz(j)=yest(j)
11    continue
      if(iest.eq.1) then
        do 12 j=1,nv
          qcol(j,1)=yest(j)
12      continue
      else
        do 13 j=1,nv
          d(j)=yest(j)
13      continue
        do 15 k1=1,iest-1
          delta=1./(x(iest-k1)-xest)
          f1=xest*delta
          f2=x(iest-k1)*delta
          do 14 j=1,nv
            q=qcol(j,k1)
            qcol(j,k1)=dy(j)
            delta=d(j)-q
            dy(j)=f1*delta
            d(j)=f2*delta
            yz(j)=yz(j)+dy(j)
14        continue
15      continue
        do 16 j=1,nv
          qcol(j,iest)=dy(j)
16      continue
      endif
      return
      END
      SUBROUTINE mmid2(y,dydx,nvar,xs,htot,nstep,yout,derivs)
      INTEGER nstep,nvar,NMAX
      REAL htot,xs,dydx(nvar),y(nvar),yout(nvar)
      EXTERNAL derivs
      PARAMETER (NMAX=50)
      INTEGER i,n
      REAL h,h2,swap,x,ym(NMAX),yn(NMAX)
      h=htot/nstep
      do 11 i=1,nvar
        ym(i)=y(i)
        yn(i)=y(i)+h*dydx(i)
11    continue
      x=xs+h
      call derivs(x,yn,yout)
      h2=2.*h
      do 13 n=2,nstep
        do 12 i=1,nvar
          swap=ym(i)+h2*yout(i)
          ym(i)=yn(i)
          yn(i)=swap
12      continue
        x=x+h
        call derivs(x,yn,yout)
13    continue
      do 14 i=1,nvar
        yout(i)=0.5*(ym(i)+yn(i)+h*yout(i))
14    continue
      return
      END

c----------------------------------------------------------------------------- 
      subroutine biwgt (x,n,xbl,xbs)
c----------------------------------------------------------------------------- 

c  This subroutine calculates an estimator of the location and scale
c  of the data set X.  The scale uses the Biweight function in the
c  general formula of "A-estimators." This formula is given on page 416
c  in UREDA (formula 4). The BIWEIGHT function is given by:
c
c                                  u((1-u*u)**2)     abs(u) <= 1
c                         f(u) =
c                                  0                 abs(u) >  1
c
c  where u is defined by
c
c                         u = (X(I) - Med) / (c * Mad)  .
c
c  Med, Mad, and c are the median, the median absolute deviation from
c  the median, and the tuning constant respectively. The tuning
c  constant is a parameter which is chosen depending on the sample
c  size and the specific function being used for the scale estimate.
c  (See page 417 in UREDA).  The tuning constant c is set to 6.0 for 
c  calculation of the location, as recommended by Tukey.
c
c  The biweght location is found iteratively using the formula:
c
c                         T = Med + (sums)
c
c  where (sums) are as given on page 421 in UREDA
c
c  ( some comments on the scale   Here we take c = 9.0.)
c
c  WARNING: This version of the Biweight routine has the side effect of
c  sorting the input array X, and returning the sorted data in place.


      integer n, i
      real x(n), xbl, xbs, xmed, xmad
      real cmad, cmadsq, delta
      real*8 sum1,sum2,t0,t1

      if (n .lt. 1) then
         xbl =  0.0
         xbs = -2.0
         return
      else if (n .eq. 1) then
         xbl =  x(1)
         xbs = -1.0
         return
      else if (n .eq. 2) then
         xbl = (x(1) + x(2)) / 2.0
         xbs = abs(x(1) - xbl)
         return
      end if

      call medmad (x, n, xmed, xmad)

      if (xmad .lt. 1.0e-6 * max(1.0, abs(xmed))) then
         xbl = xmed
         xbs = xmad
         return
      end if

      xbl = xmed
      delta = max(2.0e-5,abs(xmed))
      cmad = 6.0 * xmad
      cmadsq = cmad * cmad

      icnt=0
c      do while (abs(delta) .ge. 1.0e-5 * max(1.0,abs(xbl)))
c      do while (abs(delta) .ge. 1.0e-4 * xmad)
      do while ((abs(delta) .ge. 1.0e-4 * xmad).and.(icnt.lt.20))
         
         icnt=icnt+1
         sum1 = 0.0
         sum2 = 0.0

         do i=1,n
            t0 = x(i) - xbl
            if (abs(t0) .lt. cmad) then
               t1 = cmadsq - t0 * t0
               t1 = t1 * t1
               sum1 = sum1  + (t0 * t1)
               sum2 = sum2 + t1
            endif
         end do
         
         delta = sum1 / sum2
         xbl = xbl + delta
      end do

      sum1 = 0.0
      sum2 = 0.0
      cmad = 9.0 * xmad
      cmadsq = cmad * cmad
      
      do i=1,n
         t0 = x(i) - xbl
         if (abs(t0) .lt. cmad) then
            t0 = t0 * t0
            t1 = cmadsq - t0
            sum1 = sum1 + (t0 * t1 * t1 * t1 * t1)
            sum2 = sum2 + t1 * (cmadsq - 5.0 * t0)
         endif
      end do

      xbs = n * sqrt(sum1/(n-1.0)) / abs(sum2)

      return
      end



c------------------------------------------------------------------------------
      subroutine medmad (x,n,xmed,xmad)
c------------------------------------------------------------------------------

c  This routine calculates the median and the median absolute deviation of
c  a data set.  The data are supplied in the array X, of length N. The
c  median is returned in XMED, and the median absolute deviation in XMAD.
c
c  If N is odd, the median is the value from the data set that has equal
c  numbers of values greater than it and less than it.  If N is even, the
c  median is the mean of the two central values of the data set.
c
c  The median absolute deviation, as the name implies, is the median of
c  the distribution of absolute values of the deviation of the data about
c  the median.
c
c  WARNING: This routine has the side effect of sorting the input array X.


      integer i, j, k, n, n2
      real x(n), xmed, xmad, xi, xj
      logical even

      even = mod(n,2) .eq. 0
      n2 = n / 2

c  sort the data

      call sort (n,x)

c  calculate the median

      if (even) then
         xmed = 0.5 * (x(n2) + x(n2+1))
      else
         xmed = x(n2+1)
      endif

c  calculate the mad

      i = n2
      j = n2 + 2
      if (even) j = n2 + 1
      xi = xmed - x(i)
      xj = x(j) - xmed

      do k=1,n2
         if (xi .lt. xj) then
            xmad = xi
            if (i .gt. 1) then
               i = i - 1
               xi = xmed - x(i)
            else
               j = j + 1
               xj = x(j) - xmed
            end if
         else
            xmad = xj
            if (j .lt. n) then
               j = j + 1
               xj = x(j) - xmed
            else
               i = i - 1
               xi = xmed - x(i)
            end if
         end if
      end do

      if (even) then
         if (xi .lt. xj) then
            xmad = 0.5 * (xmad + xi)
         else
            xmad = 0.5 * (xmad + xj)
         end if
      end if

      return
      end


c------------------------------------------------------------------------------
      subroutine sort (n,x)
c------------------------------------------------------------------------------
 
c  This routine performs a heapsort of the data in array X, of length N.
c  The data are sorted into ascending order, and returned in the array X.
c
c  Stolen (unabashedly) from NUMERICAL RECIPES.



      integer i, j, k, m, n
      real x(n), temp
 
      m = n / 2 + 1
      k = n

 10   if (m .gt. 1) then
         m = m - 1
         temp = x(m)
      else
         temp = x(k)
         x(k) = x(1)
         k = k - 1
         if (k .eq. 1) then
            x(1) = temp
            return
         endif
      endif

      i = m
      j = m + m

 20   if (j .le. k) then
         if (j .lt. k) then
            if (x(j) .lt. x(j+1)) j = j + 1
         endif
         if (temp .lt. x(j)) then
            x(i) = x(j)
            i = j
            j = j + j
         else
            j = k + 1
         endif
         go to 20
      endif

      x(i) = temp
      go to 10

      end

