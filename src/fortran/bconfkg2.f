C-----------------------------------------------------------------------
C  IMSL Name:  BCONF/DBCONF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    December 16, 1985
C
C  Purpose:    Minimize a function of N variables subject to bounds on
C              the variables using a quasi-Newton method and a
C              finite-difference gradient.
C
C  Usage:      CALL BCONF (FCN, N, XGUESS, IBTYPE, XLB, XUB, XSCALE,
C                          FSCALE, IPARAM, RPARAM, X, FVALUE)
C
C  Arguments:
C     FCN    - User-supplied SUBROUTINE to evaluate the function to be
C              minimized.  The usage is
C              CALL FCN (N, X, F), where
C              N      - Length of X.  (Input)
C              X      - Vector of length N at which point the function
C                       is evaluated.  (Input)
C                       X should not be changed by FCN.
C              F      - The computed function value at the point X.
C                       (Output)
C              FCN must be declared EXTERNAL in the calling program.
C     N      - Dimension of the problem.  (Input)
C     XGUESS - Vector of length N containing an initial guess of the
C              computed solution.  (Input)
C     IBTYPE - Scalar indicating the types of bounds on variables.
C              (Input)
C              IBTYPE  Action
C                 0    User will supply all the bounds.
C                 1    All variables are nonnegative.
C                 2    All variables are nonpositive.
C                 3    User supplies only the bounds on 1st variable,
C                      all other variables will have the same bounds.
C     XLB    - Vector of length N containing the lower bounds on
C              variables.  (Input, if IBTYPE = 0; output, if IBTYPE = 1
C              or 2; input/output, if IBTYPE = 3)
C     XUB    - Vector of length N containing the upper bounds on
C              variables.  (Input, if IBTYPE = 0; output, if IBTYPE = 1
C              or 2; input/output, if IBTYPE = 3)
C     XSCALE - Vector of length N containing the diagonal scaling matrix
C              for the variables.  (Input)
C              In the absence of other information, set all entries to
C              1.0.
C     FSCALE - Scalar containing the function scaling.  (Input)
C              In the absence of other information, set FSCALE to 1.0.
C     IPARAM - Parameter vector of length 7.  (Input/Output)
C              See Remarks.
C     RPARAM - Parameter vector of length 7.  (Input/Output)
C              See Remarks.
C     X      - Vector of length N containing the computed solution.
C              (Output)
C     FVALUE - Scalar containing the value of the function at the
C              computed solution.  (Output)
C
C  Remarks:
C  1. Automatic workspace usage is
C              BCONF    N*(2*N+8)+N   units, or
C              DBCONF   2*N*(2*N+8)+N units.
C     Workspace may be explicitly provided, if desired, by use of
C     B2ONF/DB2ONF.  The reference is
C              CALL B2ONF (FCN, N, XGUESS, IBTYPE, XLB, XUB, XSCALE,
C                          FSCALE, IPARAM, RPARAM, X, FVALUE, WK, IWK)
C     The additional arguments are as follows:
C     WK     - Real work vector of length N*(2*N+8).  WK contains the
C              the following information on output:
C              The second N locations contain the last step taken.
C              The third N locations contain the last Newton step.
C              The fourth N locations contain an estimate of the
C                  gradient at the solution.
C              The final N*N locations contain a BFGS approximation
C                  to the Hessian at the solution.
C     IWK    - Work vector of length N.
C
C  2. Informational errors
C     Type Code
C       3   1  Both the actual and predicted relative reductions in the
C              function are less than or equal to the relative function
C              convergence tolerance.
C       4   2  The iterates appear to be converging to a noncritical
C              point.
C       4   3  Maximum number of iterations exceeded.
C       4   4  Maximum number of function evaluations exceeded.
C       4   5  Maximum number of gradient evaluations exceeded.
C       4   6  Five consecutive steps have been taken with the maximum
C              step length.
C
C  3. The first stopping criterion for BCONF occurs when the norm of the
C     gradient is less than the given gradient tolerance (RPARAM(1)).
C     The second stopping criterion for BCONF occurs when the scaled
C     distance between the last two steps is less than the step
C     tolerance (RPARAM(2)).
C
C  4. If the default parameters are desired for BCONF, then set
C     IPARAM(1) to zero and call the optimization program omitting the
C     call to U4INF.  Otherwise, if the nondefault parameters are
C     desired for IPARAM or RPARAM, then U4INF is called and the
C     corresponding parameters are set to the desired value before
C     calling the optimization program.  The use of U4INF would be as
C     follows:
C
C              CALL U4INF (IPARAM, RPARAM).
C              Set the desired parameters.
C
C     The following is a list of the parameters and the default values:
C     IPARAM - Integer vector of length 7.
C              IPARAM(1) = Initialization flag. (0)
C              IPARAM(2) = Number of good digits in the function.
C                          (Machine dependent)
C              IPARAM(3) = Maximum number of iterations. (100)
C              IPARAM(4) = Maximum number of function evaluations. (400)
C              IPARAM(5) = Maximum number of gradient evaluations. (400)
C              IPARAM(6) = Hessian initialization parameter. (0)
C                          If IPARAM(6) = 0 the Hessian is initialized
C                          to the identity matrix, otherwise it is
C                          initialized to a diagonal matrix where
C                          H(I,I) =
C                          MAX (FSCALE, ABS(F(X(I)))) * XSCALE(I)**2
C              IPARAM(7) = Maximum number of Hessian evaluations. (100)
C                          (Not required when using BCONF/DBCONF)
C     RPARAM - Real vector of length 7.
C              RPARAM(1) = Scaled gradient tolerance. (eps**(2/3))
C              RPARAM(2) = Scaled step tolerance. (eps**(2/3))
C              RPARAM(3) = Relative function tolerance.
C                          (MAX(1.0E-10,eps**(2/3)))
C              RPARAM(4) = Absolute function tolerance.
C                          (MAX(1.0E-10,eps**(2/3)))
C              RPARAM(5) = False convergence tolerance. (100*eps)
C              RPARAM(6) = Maximum allowable step size.
C                          (1000*MAX(TOL1,TOL2)) where,
C                          TOL1 = SQRT(sum of (XSCALE(I)*XGUESS(I))**2)
C                                 for I = 1,...,N
C                          TOL2 = 2-norm of XSCALE.
C              RPARAM(7) = Size of initial trust region radius.
C                          (Based on the initial scaled Cauchy step)
C     eps is machine epsilon.
C     If double precision is desired, then DU4INF is called and RPARAM
C     is declared double precision.
C
C  Keywords:   Factored BFGS update; Line search
C
C  GAMS:       G2h1a1
C
C  Chapter:    MATH/LIBRARY Optimization
C
C  Copyright:  1985 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE BCONFkg (FCN, N, XGUESS, IBTYPE, XLB, XUB, XSCALE,
     &                  FSCALE, IPARAM, RPARAM, X, FVALUE)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, IBTYPE, IPARAM(*)
      REAL       FSCALE, FVALUE, XGUESS(*), XLB(*), XUB(*), XSCALE(*),
     &           RPARAM(*), X(*)
      EXTERNAL   FCN
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    INDI, INDR, LWK
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                                  SPECIFICATIONS FOR COMMON /WORKSP/
      REAL       RWKSP(5000)
      REAL       RDWKSP(5000)
      DOUBLE PRECISION DWKSP(2500)
      COMPLEX    CWKSP(2500)
      COMPLEX    CZWKSP(2500)
      COMPLEX    *16 ZWKSP(1250)
      INTEGER    IWKSP(5000)
      LOGICAL    LWKSP(5000)
      EQUIVALENCE (DWKSP(1), RWKSP(1))
      EQUIVALENCE (CWKSP(1), RWKSP(1)), (ZWKSP(1), RWKSP(1))
      EQUIVALENCE (IWKSP(1), RWKSP(1)), (LWKSP(1), RWKSP(1))
      EQUIVALENCE (RDWKSP(1), RWKSP(1)), (CZWKSP(1), RWKSP(1))
      COMMON     /WORKSP/ DWKSP
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, E1STI, B2ONF
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   I1KGT, N1RCD
      INTEGER    I1KGT, N1RCD
C
      CALL E1PSH ('BCONF ')
C
      IF (N .LE. 0) THEN
         CALL E1STI (1, N)
         CALL E1MES (5, 1, 'The number of variables must be '//
     &               'positive while N = %(I1) is given.')
      ELSE
C                                  Allocate workspace
         LWK = N*(2*N+8)
         INDR = I1KGT(LWK,3)
         INDI = I1KGT(N,2)
         IF (N1RCD(0) .NE. 0) THEN
            CALL E1MES (5, 2, ' ')
            CALL E1STI (1, N)
            CALL E1MES (5, 2, 'The workspace is based on N, '//
     &                  'where N = %(I1) is given.')
         ELSE
            CALL B2ONFkg (FCN, N, XGUESS, IBTYPE, XLB, XUB, XSCALE,
     &                  FSCALE, IPARAM, RPARAM, X, FVALUE,
     &                  RDWKSP(INDR), IWKSP(INDI))
         END IF
      END IF
C
      CALL E1POP ('BCONF ')
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  B2ONF/DB2ONF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Minimize a function of N variables subject to bounds on
C              the variables using a quasi-Newton method and a finite
C              difference gradient.
C
C  Usage:      CALL B2ONF (FCN, N, XGUESS, IBTYPE, XLB, XUB, XSCALE,
C                          FSCALE, IPARAM, RPARAM, X, FVALUE, WK, IWK)
C
C  Arguments:  See BCONF/DBCONF.
C
C  Remarks:    See BCONF/DBCONF.
C
C  Chapter:    MATH/LIBRARY Optimization
C
C  Copyright:  1985 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE B2ONFkg (FCN, N, XGUESS, IBTYPE, XLB, XUB, XSCALE,
     &                  FSCALE, IPARAM, RPARAM, X, FVALUE, WK, IWK)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, IBTYPE, IPARAM(*), IWK(*)
      REAL       FSCALE, FVALUE, XGUESS(*), XLB(*), XUB(*), XSCALE(*),
     &           RPARAM(*), X(*), WK(*)
      EXTERNAL   FCN
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, E1STI, SCOPY, SSET, B3ONF
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   N1RCD
      INTEGER    N1RCD
C
      CALL E1PSH ('B2ONF ')
C
      IF (N .LE. 0) THEN
         CALL E1STI (1, N)
         CALL E1MES (5, 1, 'The number of variables must be '//
     &               'positive while N = %(I1) is given.')
      ELSE
         CALL SCOPY (N, XGUESS, 1, X, 1)
         IF (IBTYPE .EQ. 1) THEN
            CALL SSET (N, 0.0E0, XLB, 1)
            CALL SSET (N, 1.0E20, XUB, 1)
         ELSE IF (IBTYPE .EQ. 2) THEN
            CALL SSET (N, -1.0E20, XLB, 1)
            CALL SSET (N, 0.0E0, XUB, 1)
         ELSE IF (IBTYPE .EQ. 3) THEN
            CALL SSET (N-1, XLB(1), XLB(2), 1)
            CALL SSET (N-1, XUB(1), XUB(2), 1)
         ELSE IF (IBTYPE.LT.0 .OR. IBTYPE.GE.4) THEN
            CALL E1STI (1, IBTYPE)
            CALL E1MES (5, 2, 'The value for the type of bounds on '//
     &                  'the variables, IBTYPE, must be greater than '//
     &                  'or equal to zero and less than or equal to '//
     &                  'three while IBTYPE = %(I1) is given.')
         END IF
C                                  Call unconstrained minimization
C                                  solver using F.D. gradient
         IF (N1RCD(0) .EQ. 0) CALL B3ONFkg (FCN, N, X, XLB, XUB, XSCALE,
     &       FSCALE, IPARAM, RPARAM, IWK, FVALUE, WK(1), WK(N+1),
     &       WK(2*N+1), WK(3*N+1), WK(4*N+1), WK((N+8)*N+1), N,
     &       WK(5*N+1), WK(6*N+1), WK(7*N+1), WK(8*N+1))
      END IF
C
      CALL E1POP ('B2ONF ')
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  B3ONF/DB3ONF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Driver for unconstrained minimization solver using finite
C              difference or analytic gradient.
C
C  Usage:      CALL B3ONF (FCN, N, XC, XLB, XUB, XSCALE, FSCALE, IPARAM,
C                          RPARAM, IACT, FVALUE, XP, SC, SNWTN, GC, GP,
C                          H, LDH, WK1, WK2, WK3, HCH)
C
C  Arguments:
C     FCN    - User-supplied SUBROUTINE to evaluate the function to be
C              minimized.  The usage is
C              CALL FCN (N, X, F), where
C              N      - Length of X.  (Input)
C              X      - The point at which the function is evaluated.
C                       (Input)
C                       X should not be changed by FCN.
C              F      - The computed function value at the point X.
C                       (Output)
C              FCN must be declared EXTERNAL in the calling program.
C     N      - Dimension of the problem.  (Input)
C     XC     - Real vector of length N containing initial guess on input
C              and approximate solution on output.  (Input / Output)
C     XLB    - Real vector of length N containing lower bounds of the
C              variables.  (Input)
C     XUB    - Real vector of length N containing upper bounds of the
C              variables.  (Input)
C     XSCALE - Real vector of length N containing the diagonal scaling
C              matrix for the variables.  (Input)
C     FSCALE - Real scalar containing the function scaling.  (Input)
C     IPARAM - Integer parameters vector of length 7.  (Input)
C              See U4INF for details.
C     RPARAM - Real parameters vector of length 7.  (Input)
C              See U4INF for details.
C     IACT   - Integer vector of length N indicating if XC(I) had to be
C              moved to an upper or lower bound.  (Input)
C     FVALUE - Real scalar containing the value of the function at the
C              solution.  (Output)
C     XP     - Real vector of length N containing the updated point.
C                 (Output)
C     SC     - Real vector of length N containing the last step taken.
C                 (Output)
C     SNWTN  - Real vector of length N containing the last Newton step.
C                 (Output)
C     GC     - Real vector of length N containing an estimate of the
C              gradient at the current point.  (Output)
C     GP     - Real vector of length N containing an estimate of the
C              gradient at the updated point.  (Output)
C     H      - Real N by N matrix containing an estimate of the Hessian
C              at the approximate solution.  (Output)
C     LDH    - Leading dimension of H exactly as specified in the
C              dimension statement of the calling program.  (Input)
C     WK1    - Real work vector of length N.  (Output)
C     WK2    - Real work vector of length N.  (Output)
C     WK3    - Real work vector of length N.  (Output)
C     HCH    - Real N by N matrix containing an estimate of the Cholesky
C              decomposition at the approximate solution.  (Output)
C
C  Chapter:    MATH/LIBRARY Optimization
C
C  Copyright:  1985 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE B3ONFkg (FCN, N, XC, XLB, XUB, XSCALE, FSCALE, IPARAM,
     &                  RPARAM, IACT, FVALUE, XP, SC, SNWTN, GC, GP,
     &                  H, LDH, WK1, WK2, WK3, HCH)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, LDH, IPARAM(*), IACT(*)
      REAL       FSCALE, FVALUE, XC(*), XLB(*), XUB(*), XSCALE(*),
     &           RPARAM(*), XP(*), SC(*), SNWTN(*), GC(*), GP(*),
     &           H(LDH,*), WK1(*), WK2(*), WK3(*), HCH(LDH,*)
      EXTERNAL   FCN
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, ICODE, IHESS, ITER, JJ, NACT, NFCN, NGRAD, NHESS,
     &           NNA
      REAL       DELTA, EPS, EPSFCN, FC, FDIGIT, FP, RNWTNL, SCGRAD,
     &           STEPMX, VALMAX
      LOGICAL    CHANGE, DONE, FDIFF, MXTAKE, SAMEP
C                                  SPECIFICATIONS FOR COMMON /U16NF/
      COMMON     /U16NF/ GRADTL, STEPTL, RFTOL, AFTOL, FALSTL, MXITER,
     &           MAXFCN, MAXGRD, MAXHES
      INTEGER    MXITER, MAXFCN, MAXGRD, MAXHES
      REAL       GRADTL, STEPTL, RFTOL, AFTOL, FALSTL
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS,AMAX1
      INTRINSIC  ABS, AMAX1
      REAL       ABS, AMAX1
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1POP, E1PSH, E1USR, SCOPY, B4ONF, B5ONF, B6ONF,
     &           B7ONF, B8ONF, CDGRD, CRGRG, FDGRD, TRNRR, U10NF,
     &           U11NF, U4IAH, U4INF, U5INF, U6INF, U9INF
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   AMACH, N1RCD, N1RTY, SNRM2
      INTEGER    N1RCD, N1RTY
      REAL       AMACH, SNRM2
C
      CALL E1PSH ('B3ONF ')
C                                  SET UP DEFAULT PARAMETERS
      IF (IPARAM(1) .EQ. 0) CALL U4INF (IPARAM, RPARAM)
C                                  CHECK THE VALIDITY OF THE USER
C                                  SPECIFIED PARAMETERS
      CALL U5INF (N, XC, XSCALE, FSCALE, .FALSE., IPARAM, RPARAM)
      IF (N1RTY(1) .EQ. 5) GO TO 9000
C
      FDIGIT = IPARAM(2)
      MXITER = IPARAM(3)
      MAXFCN = IPARAM(4)
      MAXGRD = IPARAM(5)
      IHESS = IPARAM(6)
      MAXHES = IPARAM(7)
C
      GRADTL = RPARAM(1)
      STEPTL = RPARAM(2)
      RFTOL = RPARAM(3)
      AFTOL = RPARAM(4)
      FALSTL = RPARAM(5)
      STEPMX = RPARAM(6)
      DELTA = RPARAM(7)
C                                  SET EPSFCN TO ESTIMATE OF RELATIVE
C                                  NOISE IN FUNCTION
      EPS = AMACH(4)
      EPSFCN = AMAX1(EPS,10.0E0**(-FDIGIT))
C
C                                  Check infeasibility of variables,
C                                  and set them to the bound
      DO 10  I=1, N
         IF (XC(I) .LT. XLB(I)) THEN
            XC(I) = XLB(I)
         ELSE IF (XC(I) .GT. XUB(I)) THEN
            XC(I) = XUB(I)
         END IF
   10 CONTINUE
C                                  INITIALIZE ITERATION, FUNCTION &
C                                  GRADIENT EVALUATION COUNTER
      ITER = 0
      NFCN = 0
      NGRAD = 0
      NHESS = 0
C                                  EVALUATE THE FUNCTION & GRADIENT AT
C                                  THE INITIAL POINT; ALSO SET FVALUE
C                                  = FUNCTION VALUE AT INITIAL GUESS
C                                  FOR THE CASE THAT THE INITIAL
C                                  GUESS IS THE SOLUTION.
      CALL E1USR ('ON')
      CALL FCN (N, XC, FC)
      CALL E1USR ('OFF')
      NFCN = NFCN + 1
      FVALUE = FC
      CALL FDGRD (FCN, N, XC, XSCALE, FC, EPSFCN, GC)
      FDIFF = .TRUE.
      NGRAD = NGRAD + 1
C                                  CHECK STOPPING CRITERIA AT THE
C                                  INITIAL POINT
      ICODE = 0
      MXTAKE = .FALSE.
      CALL U6INF (N, XC, SC, FC, GC, XSCALE, FSCALE, ICODE, ITER,
     &            NFCN, NGRAD, NHESS, .FALSE., MXTAKE)
      IF (N1RCD(1).NE.0 .OR. ICODE.EQ.-999) GO TO 140
C                                  SET UP IACT
      CHANGE = .FALSE.
      SAMEP = .FALSE.
      CALL B6ONF (N, XC, XLB, XUB, GC, IACT, EPS, CHANGE)
C                                  GET THE (APPROXIMATE) HESSIAN AT THE
C                                  INITAL POINT
      CALL U9INF (N, FC, FSCALE, XSCALE, IHESS, H, LDH)
      CALL CRGRG (N, H, LDH, HCH, LDH)
C                                  MAIN ITERATION LOOP
   20 CONTINUE
      ITER = ITER + 1
C                                  CHANGE DIMENSION OF H
      IF (CHANGE) THEN
         CALL B7ONF (N, IACT, HCH, LDH, NACT)
         NNA = N - NACT
      ELSE
         NNA = N
      END IF
C
      IF (NNA .EQ. 0) THEN
         DONE = .TRUE.
         CALL B8ONF (N, IACT, GC, DONE)
         IF (DONE) GO TO 130
         NNA = 1
      END IF
C                                  PERFORM CHOLESKY FACTORIZATION
      CALL U4IAH (NNA, HCH, LDH, XSCALE, WK2)
   30 JJ = 1
      DO 40  I=1, N
         IF (IACT(I) .EQ. 0) THEN
            WK3(JJ) = GC(I)
            JJ = JJ + 1
         END IF
   40 CONTINUE
C                                  COMPUTE NEWTON STEP AND LENGTH OF
C                                  SCALED NEWTON STEP
   50 CALL U10NF (NNA, HCH, LDH, WK3, SC)
      JJ = 1
      DO 60  I=1, N
         IF (IACT(I) .NE. 0) THEN
            SNWTN(I) = 0.0
         ELSE
            SNWTN(I) = SC(JJ)
            JJ = JJ + 1
         END IF
   60 CONTINUE
      CALL U11NF (N, XSCALE, 1, SNWTN, WK1)
      RNWTNL = SNRM2(N,WK1,1)
C                                  COMPUTE THE DOUBLE DOGLEG STEP
   70 CALL B4ONF (FCN, N, XC, FC, GC, XLB, XUB, SNWTN, XSCALE, STEPMX,
     &            STEPTL, IACT, ICODE, XP, FP, SC, MXTAKE, RNWTNL,
     &            EPSFCN, NFCN)
C                                  USE CENTRAL DIFFERENCE IF
C                                  NECESSARY
      IF ((ICODE.EQ.1) .AND. (FDIFF)) THEN
         CALL CDGRD (FCN, N, XC, XSCALE, EPSFCN, GC)
         FDIFF = .FALSE.
         VALMAX = 0.0E0
         DO 80  I=1, N
            IF (IACT(I) .LE. 0) THEN
               SCGRAD = ABS(GC(I))*AMAX1(ABS(XC(I)),1.0E0/XSCALE(I))/
     &                  AMAX1(ABS(FC),FSCALE)
               VALMAX = AMAX1(SCGRAD,VALMAX)
            END IF
   80    CONTINUE
         IF (VALMAX .LE. GRADTL) GO TO 110
         GO TO 30
      ELSE IF (ICODE .EQ. 1) THEN
         SAMEP = .TRUE.
      END IF
C                                  EVALUATE THE GRADIENT AT
C                                  NEW POINT
      IF (FDIFF) THEN
         CALL FDGRD (FCN, N, XP, XSCALE, FP, EPSFCN, GP)
         CALL B6ONF (N, XP, XLB, XUB, GP, IACT, EPS, CHANGE)
         VALMAX = 0.0E0
C
         DO 90  I=1, N
            IF (IACT(I) .LE. 0) THEN
               SCGRAD = ABS(GP(I))*AMAX1(ABS(XP(I)),1.0E0/XSCALE(I))/
     &                  AMAX1(ABS(FP),FSCALE)
               VALMAX = AMAX1(SCGRAD,VALMAX)
            END IF
   90    CONTINUE
C
         IF (VALMAX .LE. 0.1E0) THEN
            CALL CDGRD (FCN, N, XP, XSCALE, EPSFCN, GP)
            FDIFF = .FALSE.
            NGRAD = NGRAD + 1
         END IF
      ELSE
         CALL CDGRD (FCN, N, XP, XSCALE, EPSFCN, GP)
         CALL B6ONF (N, XP, XLB, XUB, GP, IACT, EPS, CHANGE)
      END IF
      NGRAD = NGRAD + 1
      DO 100  I=1, N
         IF (IACT(I) .LE. 0) THEN
            WK1(I) = GP(I)
         ELSE
            WK1(I) = 0.0
         END IF
  100 CONTINUE
C                                  CHECK STOPPING CRITERIA AT NEW POINT
  110 CALL U6INFkg (N, XP, SC, FP, WK1, XSCALE, FSCALE, ICODE, ITER,
     &            NFCN, NGRAD, NHESS, .FALSE., MXTAKE)
      IF (N1RCD(1) .EQ. 0) THEN
         DONE = .FALSE.
         IF (ICODE .EQ. -999) THEN
            DONE = .TRUE.
            IF (N .NE. NNA) THEN
               CALL B8ONF (N, IACT, GP, DONE)
            END IF
         END IF
  120    IF (.NOT.DONE) THEN
C                                  UPDATE THE HESSIAN APPROXIMATION,
C                                  XC, GC, FC; NEXT ITERATION
            IF (.NOT.SAMEP) THEN
               CALL SCOPY (N, H(1,1), LDH+1, WK1, 1)
               CALL TRNRR (N, N, H, LDH, N, N, H, LDH)
               CALL B5ONF (N, SC, GC, GP, EPSFCN, .FALSE., H, LDH,
     &                     WK1, WK2, WK3)
            ELSE
               SAMEP = .FALSE.
            END IF
            CALL CRGRG (N, H, LDH, HCH, LDH)
            CALL SCOPY (N, XP, 1, XC, 1)
            CALL SCOPY (N, GP, 1, GC, 1)
            FC = FP
            GO TO 20
         END IF
      END IF
C                                  OTHERWISE THE STOPPING CRITERIA IS
C                                  SATISFIED; RETURN
  130 CALL SCOPY (N, XP, 1, XC, 1)
      CALL SCOPY (N, GP, 1, GC, 1)
      FVALUE = FP
  140 IPARAM(3) = ITER
      IPARAM(4) = NFCN
      IPARAM(5) = NGRAD
C
 9000 CALL E1POP ('B3ONF ')
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U6INF/DU6INF  (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Stopping conditions for unconstrained minimization.
C
C  Usage:      CALL U6INF (N, XP, SC, FP, GP, XSCALE, FSCALE, ICODE,
C                          ITER, NFCN, NGRAD, NHESS, USRHES, MXTAKE)
C
C  Arguments:
C     N      - Dimension of the problem.  (Input)
C     XP     - Vector of length N containing the new iterate.
C              (Input)
C     SC     - Vector of length N containing step taken.  (Input)
C     FP     - Scalar containing the function value at XP.  (Input)
C     GP     - Vector of length N containing the gradient at XP.
C              (Input)
C     XSCALE - Vector of length N containing the diagonal scaling
C              matrix for the variables.  (Input)
C     FSCALE - Scalar containing the function scaling.  (Input)
C     ICODE  - Return code from the global strategy algorithm.  (Input)
C     ITER   - Number of iterations.  (Input)
C     NFCN   - Number of function evaluations.  (Input)
C     NGRAD  - Number of gradient evaluations.  (Input)
C     NHESS  - Number of Hessian evaluations.   (Input)
C     USRHES - Logical variable.  (Input)
C              USRHES = .TRUE. if Newton's method is used.
C              USRHES = .FALSE. otherwise.
C     MXTAKE - Logical variable indicating a step of maximum length was
C              taken.  (Input)
C
C  Chapter:    MATH/LIBRARY Optimization
C
C  Copyright:  1985 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE U6INFkg (N, XP, SC, FP, GP, XSCALE, FSCALE, ICODE,
     &                  ITER, NFCN, NGRAD, NHESS, USRHES, MXTAKE)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, ICODE, ITER, NFCN, NGRAD, NHESS
      REAL       FP, FSCALE, XP(*), SC(*), GP(*), XSCALE(*)
      LOGICAL    USRHES, MXTAKE
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I
      REAL       SCGRAD, SCSTEP, VALMAX
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    NMAXS
      SAVE       NMAXS
C                                  SPECIFICATIONS FOR COMMON /U16NF/
      COMMON     /U16NF/ GRADTL, STEPTL, RFTOL, AFTOL, FALSTL, MXITER,
     &           MAXFCN, MAXGRD, MAXHES
      INTEGER    MXITER, MAXFCN, MAXGRD, MAXHES
      REAL       GRADTL, STEPTL, RFTOL, AFTOL, FALSTL
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS,AMAX1
      INTRINSIC  ABS, AMAX1
      REAL       ABS, AMAX1
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, E1STR, U18NF
C
      CALL E1PSH ('U6INF ')
c      print *,icode,iter,mxiter,nfcn,maxfcn
C
      IF (ITER .GE. MXITER) THEN
C                                  Print error message
Ckg         CALL U18NF (3)
c         CALL E1MES (3, 3, ' Maximum number of iterations exceeded.')
         call e1mes(3,1,'')
         goto 9000
      endif
      IF (NFCN .GE. MAXFCN) THEN
C                                  Print error message
ckg         CALL U18NF (4)
c         CALL E1MES (3, 8, ' Maximum number of function evaluations'//
c     &               ' exceeded.')
         call e1mes(3,1,'')
         go to 9000
      endif
C                                  TEST OF NORM OF SCALED GRADIENT
      VALMAX = 0.0E0
      DO 10  I=1, N
         SCGRAD = ABS(GP(I))*AMAX1(ABS(XP(I)),1.0E0/XSCALE(I))/
     &            AMAX1(ABS(FP),FSCALE)
         VALMAX = AMAX1(SCGRAD,VALMAX)
   10 CONTINUE
      IF (VALMAX .LE. GRADTL) THEN
ckg
c         ICODE = -999
c         print *,'hello'
         call e1mes(3,1,'')
         GO TO 9000
      END IF
C                                  IF FIRST ITER., INITIALIZE COUNTER
C                                  FOR MX. STEP TAKEN AND RETURN
      IF (ITER .EQ. 0) THEN
         NMAXS = 0
         GO TO 9000
      END IF
C                                  CHECK LAST GLOBAL STEP
      IF (ICODE .EQ. 1) THEN
         CALL E1STR (1, STEPTL)
ckg         CALL E1MES (3, 8, 'The last global step failed to '//
ckg     &               'locate a lower point than the current X '//
ckg     &               'value.  The current X may be an approximate '//
ckg     &               'local minimizer and no more accuracy is '//
ckg     &               'possible or the step tolerance may be too '//
ckg     &               'large where STEPTL = %(R1) is given.')
c         call e1mes(3,8,'Close enough')
         call e1mes(3,1,'')
         GO TO 9000
      END IF
C                                  TEST NORM OF SCALED STEP
      VALMAX = 0.0E0
      DO 20  I=1, N
         SCSTEP = ABS(SC(I))/AMAX1(ABS(XP(I)),1.0E0/XSCALE(I))
         VALMAX = AMAX1(SCSTEP,VALMAX)
   20 CONTINUE
      IF (VALMAX .LE. STEPTL) THEN
ckg         ICODE = -999
c         call e1mes(3,8,'I do not know what is going on')
         call e1mes(3,1,'')
         GO TO 9000
      END IF
C                                  CHECK RELATIVE FUNCTION CONVERGENCE
C                                  TOLERANCE
      IF (ICODE .EQ. 2) THEN
         CALL E1STR (1, RFTOL)
         CALL E1MES (3, 1, 'RELATIVE FUNCTION CONVERGENCE - '//
     &               'Both the actual and predicted relative '//
     &               'reductions in the function are less '//
     &               'than or equal to the relative function '//
     &               'convergence tolerance RFTOL = %(R1).')
         GO TO 9000
      END IF
C                                  CHECK FALSE CONVERGENCE TOLERANCE
      IF (ICODE .EQ. 3) THEN
         CALL E1MES (4, 2, 'FALSE CONVERGENCE - The iterates '//
     &               'appear to be converging to a noncritical '//
     &               'point.  Possibly incorrect gradient '//
     &               'information is used, or the function is '//
     &               'discontinuous, or the other stopping '//
     &               'tolerances are too tight.')
         GO TO 9000
      END IF
C                                  CHECK ITERATION, FUNCTION, GRADIENT
C                                  & HESSIAN EVALUATIONS LIMIT
      IF (ITER .GE. MXITER) THEN
C                                  Print error message
Ckg         CALL U18NF (3)
c         CALL E1MES (3, 3, ' Maximum number of iterations exceeded.')
         call e1mes(3,1,'')
         goto 9000
C
      ELSE IF (NFCN .GE. MAXFCN) THEN
C                                  Print error message
ckg         CALL U18NF (4)
c         CALL E1MES (3, 4, ' Maximum number of function evaluations'//
c     &               ' exceeded.')
         call e1mes(3,1,'')
         goto 9000
C
      ELSE IF (NGRAD .GE. MAXGRD) THEN
C                                  Print error message
ckg         CALL U18NF (5)
c         CALL E1MES (3, 5, ' Maximum number of gradient evaluations'//
c     &               ' exceeded.')
         call e1mes(3,1,'')
         goto 9000
C
      ELSE IF (USRHES .AND. (NHESS.GE.MAXHES)) THEN
C                                  Print error message
ckg         CALL U18NF (7)
c         CALL E1MES (3, 7, 'Maximum number of Hessian evaluations '//
c     &               'exceeded.')
         call e1mes(3,1,'')
         goto 9000
C
      ELSE IF (MXTAKE) THEN
         NMAXS = NMAXS + 1
         IF (NMAXS .EQ. 5) THEN
C                                  Print error message
ckg            CALL U18NF (6)
c         call e1mes(3,8,'Five consecutive steps of....get out')
         call e1mes(3,1,'')
         goto 9000
CC            CALL E1MES (4, 6, ' Five consecutive steps of '//
CC     &                  'length STEPMX have been taken; either the '//
CC     &                  'function is unbounded below, or has a '//
CC     &                  'finite asymptote in some direction or the '//
CC     &                  'maximum allowable step size STEPMX is too '//
CC     &                  'small.')
         END IF
      END IF
C
 9000 CALL E1POP ('U6INF ')
      RETURN
      END
