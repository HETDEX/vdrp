C-----------------------------------------------------------------------
C  IMSL Name:  U4IAH/DU4IAH (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Driver for the perturbed Cholesky factorization routine.
C
C  Usage:      CALL U4IAH (N, A, LDA, XSCALE, DIAG)
C
C  Arguments:
C     N      - Dimension of the problem.  (Input)
C     A      - Real N by N matrix.  (Input/Output)
C              On input, A is the symmetric positive definite matrix to
C                 be factored (with only the lower triangular part and
C                 the diagonal stored).
C              On output, A contains L of the Cholesky factorization of
C                 the perturbed matrix in the lower triangular part and
C                 diagonal and the perturbed matrix in the upper triang-
C                 ular part and DIAG respectively.
C     LDA    - Row dimension of A exactly as specified in the dimension
C              statement of the calling program.  (Input)
C     XSCALE - Real vector of length N containing the diagonal scaling
C              matrix.  (Input)
C     DIAG   - Real vector of length N containing the diagonal of the
C              perturbed symmetric positive definite matrix.  (Output)
C
C  Remark:
C     This is based on Algorithm A5.5.1, page 315, Dennis-Schnabel book.
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
      SUBROUTINE U4IAH (N, A, LDA, XSCALE, DIAG)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, LDA
      REAL       A(LDA,*), XSCALE(*), DIAG(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, J
      REAL       ADDMAX, AMU, DI1MAX, DIAMIN, EVMAX, EVMIN, MACHEP,
     &           OFFMAX, OFFROW, OFLMAX, POSMAX, SDD, TOL
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS,AMAX1,AMIN1,FLOAT,SQRT
      INTRINSIC  ABS, AMAX1, AMIN1, FLOAT, SQRT
      REAL       ABS, AMAX1, AMIN1, FLOAT, SQRT
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   SADD, SCOPY, SSCAL, CSFRG, U6IAH
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   AMACH, SASUM
      REAL       AMACH, SASUM
C
      MACHEP = AMACH(4)
C                                  SCALE HESSIAN BY PRE- AND POST-
C                                  MULTIPLY "A" BY XSCALE**-1.
      DO 20  J=1, N
         DO 10  I=J, N
            A(I,J) = A(I,J)/(XSCALE(I)*XSCALE(J))
   10    CONTINUE
   20 CONTINUE
C                                  STEP 1: IF "A" HAS ANY NEGATIVE
C                                  DIAGONAL ELEMENTS OR THE ABSOLUTE
C                                  ====== VALUE OF THE LARGEST
C                                  OFF-DIAGONAL ELEMENT OF "A" IS
C                                  .GT. LARGEST DIAGONAL ELEMENT OF
C                                  "A" THEN SET "A" = "A" + MU*I
C                                  WHERE MU.GT.0 IS CHOSEN SO THAT
C                                  THE NEW DIAGONAL IS ALL POSITIVE,
C                                  WITH RATIO OF SMALLEST TO LARGEST
C                                  ELEMENT .GE. SQRT(MACHEP), AND
C                                  RATIO OF LARGEST ELEMENT TO
C                                  LARGEST ABSOLUTE OFF-DIAGONAL .GE.
C                                  (1 + 2*SQRT(MACHEP)).
      TOL = SQRT(MACHEP)
      DI1MAX = A(1,1)
      DIAMIN = A(1,1)
      DO 30  I=2, N
         DI1MAX = AMAX1(DI1MAX,A(I,I))
         DIAMIN = AMIN1(DIAMIN,A(I,I))
   30 CONTINUE
      POSMAX = AMAX1(0.0E0,DI1MAX)
C                                  AMU WILL CONTAIN AMOUNT TO ADD TO
C                                  DIAGONAL OF A BEFOR ATTEMPTING THE
C                                  CHOLESKY FACTORIZATION.
      IF (DIAMIN .LE. (TOL*POSMAX)) THEN
         AMU = 2.0E0*(POSMAX-DIAMIN)*TOL - DIAMIN
         DI1MAX = DI1MAX + AMU
      ELSE
         AMU = 0.0E0
      END IF
C                                  GET MAXIMUM OFF-DIAGONAL ELEMENT
      OFFMAX = 0.0E0
      DO 50  I=2, N
         DO 40  J=1, I - 1
            OFFMAX = AMAX1(OFFMAX,ABS(A(I,J)))
   40    CONTINUE
   50 CONTINUE
C
      IF ((OFFMAX*(1.0E0+2.0E0*TOL)) .GT. DI1MAX) THEN
         AMU = AMU + (OFFMAX-DI1MAX) + (2.0E0*TOL*OFFMAX)
         DI1MAX = OFFMAX*(1.0E0+2.0E0*TOL)
      END IF
C                                  "A" IS EQUAL TO THE ZERO MATRIX
      IF (DI1MAX .EQ. 0.0E0) THEN
         AMU = 1.0E0
         DI1MAX = 1.0E0
      END IF
C                                  "A" = "A" + AMU*I
      IF (AMU .GT. 0.0E0) CALL SADD (N, AMU, A(1,1), LDA+1)
C                                  STEP 2: CALL THE PERTURBED CHOLESKY
C                                  DECOMPOSITION ROUTINE. ====== KEEP
C                                  A COPY OF ORIGINAL "A" IN UPPER
C                                  TRIANGLE AND DIAG. OFLMAX = BOUND
C                                  ON THE SIZE OF OFF-DIAGONAL
C                                  ELEMENT OF L.
      CALL SCOPY (N, A(1,1), LDA+1, DIAG, 1)
      DO 60  J=1, N - 1
         CALL SCOPY (N-J, A(J+1,J), 1, A(J,J+1), LDA)
   60 CONTINUE
      OFLMAX = SQRT(AMAX1(DI1MAX,(OFFMAX/FLOAT(N))))
      CALL U6IAH (N, A, LDA, OFLMAX, ADDMAX)
C                                  STEP 3: IF ADDMAX = 0, "A" IS
C                                  POSITIVE DEFINITE GOING INTO STEP
C                                  2 ==== THE CHOLESKY FACTORIZATION
C                                  HAS BEEN DONE, AND WE RETURN.
C                                  OTHERWISE, ADDMAX > 0. PRETURB "A"
C                                  SO THAT IT IS SAFELY DIAGONAL
C                                  DOMINANT AND FIND THE CHOLESKY
C                                  FACTORIZATION.
      IF (ADDMAX .GT. 0.0E0) THEN
C                                  RESTORE ORIGINAL "A"
         CALL SCOPY (N, DIAG, 1, A, LDA+1)
         CALL CSFRG (N, A, LDA)
C                                  FIND SDD SUCH THAT A + SDD*I IS
C                                  SAFELY POSITIVE DEFINITE
         EVMAX = A(1,1)
         EVMIN = A(1,1)
         DO 70  I=1, N
            OFFROW = SASUM(I-1,A(1,I),1) + SASUM(N-I,A(I,I+1),LDA)
            EVMAX = AMAX1(EVMAX,DIAG(I)+OFFROW)
            EVMIN = AMIN1(EVMIN,DIAG(I)-OFFROW)
   70    CONTINUE
         SDD = (EVMAX-EVMIN)*TOL - EVMIN
         SDD = AMAX1(SDD,0.0E0)
C                                  PERTURB "A" AND DECOMPOSE AGAIN
         AMU = AMIN1(ADDMAX,SDD)
         CALL SADD (N, AMU, A(1,1), LDA+1)
         CALL SCOPY (N, A(1,1), LDA+1, DIAG, 1)
C                                  "A" IS NOW GUARANTEED TO BE SAFELY
C                                  POSITIVE DEFINITE
         OFLMAX = 0.0E0
         CALL U6IAH (N, A, LDA, OFLMAX, ADDMAX)
      END IF
C                                  UNSCALE THE HESSIAN AND ITS FACTORS
      DO 90  I=1, N
         DO 80  J=I + 1, N
            A(I,J) = XSCALE(I)*A(I,J)*XSCALE(J)
   80    CONTINUE
         DIAG(I) = XSCALE(I)*DIAG(I)*XSCALE(I)
         CALL SSCAL (I, XSCALE(I), A(I,1), LDA)
   90 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1MES
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 2, 1984
C
C  Purpose:    Set an error state for the current level in the stack.
C              The message is printed immediately if the error type is
C              5, 6, or 7 and the print attribute for that type is YES.
C
C  Usage:      CALL E1MES(IERTYP,IERCOD,MSGPKD)
C
C  Arguments:
C     IERTYP - Integer specifying the error type.  (Input)
C                IERTYP=1,  informational/note
C                IERTYP=2,  informational/alert
C                IERTYP=3,  informational/warning
C                IERTYP=4,  informational/fatal
C                IERTYP=5,  terminal
C                IERTYP=6,  PROTRAN/warning
C                IERTYP=7,  PROTRAN/fatal
C     IERCOD - Integer specifying the error code.  (Input)
C     MSGPKD - A character string containing the message.
C              (Input)  Within the message, any of following may appear
C                %(A1),%(A2),...,%(A9) for character arrays
C                %(C1),%(C2),...,%(C9) for complex numbers
C                %(D1),%(D2),...,%(D9) for double precision numbers
C                %(I1),%(I2),...,%(I9) for integer numbers
C                %(K1),%(K2),...,%(K9) for keywords
C                %(L1),%(L2),...,%(L9) for literals (strings)
C                %(R1),%(R2),...,%(R9) for real numbers
C                %(Z1),%(Z2),...,%(Z9) for double complex numbers
C              This provides a way to insert character arrays, strings,
C              numbers, and keywords into the message.  See remarks
C              below.
C
C  Remarks:
C     The number of characters in the message after the insertion of
C     the corresponding strings, etc. should not exceed 255.  If the
C     limit is exceeded, only the first 255 characters will be used.
C     The appropriate strings, etc. need to have been previously stored
C     in common via calls to E1STA, E1STD, etc.  Line breaks may be
C     specified by inserting the two characters '%/' into the message
C     at the desired locations.
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1MES (IERTYP, IERCOD, MSGPKD)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    IERTYP, IERCOD
      CHARACTER  MSGPKD*(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    ERTYP2, I, IER, IPLEN, ISUB, LAST, LEN2, LOC, M, MS,
     &           NLOC, NUM, PBEG
      CHARACTER  MSGTMP(255)
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    IFINIT, NFORMS
      CHARACTER  BLNK, DBB(3), FIND(4), FORMS(9), INREF(25), LPAR,
     &           NCHECK(3), PERCNT, RPAR
      SAVE       BLNK, DBB, FIND, FORMS, IFINIT, INREF, LPAR, NCHECK,
     &           NFORMS, PERCNT, RPAR
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  LEN,MIN0
      INTRINSIC  LEN, MIN0
      INTEGER    LEN, MIN0
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   C1TCI, E1INIT, E1PRT, E1UCS, M1VE, M1VECH
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   I1DX
      INTEGER    I1DX
C
      DATA FORMS/'A', 'C', 'D', 'I', 'K', 'L', 'R', 'S', 'Z'/,
     &     NFORMS/9/
      DATA PERCNT/'%'/, LPAR/'('/, RPAR/')'/, BLNK/' '/
      DATA INREF/' ', 'i', 'n', ' ', 'r', 'e', 'f', 'e', 'r',
     &     'e', 'n', 'c', 'e', ' ', 't', 'o', ' ', 'k', 'e',
     &     'y', 'w', 'o', 'r', 'd', ' '/
      DATA NCHECK/'N', '1', '*'/, DBB/'.', ' ', ' '/
      DATA FIND/'*', ' ', ' ', '*'/
      DATA IFINIT/0/
C                                  INITIALIZE ERROR TABLE IF NECESSARY
      IF (IFINIT .EQ. 0) THEN
         CALL E1INIT
         IFINIT = 1
      END IF
C                                  CHECK AND SET ERROR TYPE IF NECESSARY
      IF (IERTYP .NE. -1) THEN
         ERTYPE(CALLVL) = IERTYP
      ELSE IF (IERTYP.LT.-1 .OR. IERTYP.GT.7) THEN
         MSGLEN = 51
         CALL M1VECH ('.  Error from E1MES.  Illegal error type'//
     &                ' specified. ', MSGLEN, MSGSAV, MSGLEN)
         CALL E1PRT
         STOP
      END IF
C
      ERTYP2 = ERTYPE(CALLVL)
C                                  SET ERROR CODE IF NECESSARY
      IF (IERCOD .GT. -1) ERCODE(CALLVL) = IERCOD
      LEN2 = LEN(MSGPKD)
C
      IF (IERTYP.EQ.0 .OR. IERCOD.EQ.0) THEN
C                                  REMOVE THE ERROR STATE
         MSGLEN = 0
      ELSE IF (LEN2.EQ.0 .OR. (LEN2.EQ.1.AND.MSGPKD(1:1).EQ.BLNK)) THEN
         IF (ERTYP2 .EQ. 6) IFERR6 = 1
         IF (ERTYP2 .EQ. 7) IFERR7 = 1
C                                  UPDATE CHECKSUM PARAMETER ERCKSM
         CALL E1UCS
C                                  PRINT MESSAGE IF NECESSARY
         IF (ERTYP2.GE.5 .AND. PRINTB(ERTYP2).EQ.1) CALL E1PRT
      ELSE
C                                  FILL UP MSGSAV WITH EXPANDED MESSAGE
         LEN2 = MIN0(LEN2,255)
         DO 10  I=1, LEN2
            MSGTMP(I) = MSGPKD(I:I)
   10    CONTINUE
         MS = 0
         M = 0
C                                  CHECK PLIST FOR KEYWORD NAME
         NLOC = I1DX(PLIST,PLEN,NCHECK,3)
         IF (NLOC.GT.0 .AND. HDRFMT(ERTYP2).EQ.3) THEN
C                                  M1VE INREF INTO MSGSAV
            CALL M1VE (INREF, 1, 25, 25, MSGSAV, 1, 25, 25, IER)
C                                  GET LENGTH OF KEYWORD NAME
            CALL C1TCI (PLIST(NLOC+3), 3, IPLEN, IER)
            PBEG = NLOC + 3 + IER
C                                  M1VE KEYWORD NAME INTO MSGSAV
            CALL M1VE (PLIST, PBEG, PBEG+IPLEN-1, PLEN, MSGSAV, 26,
     &                 IPLEN+25, 255, IER)
C                                  UPDATE POINTER
            MS = IPLEN + 25
         END IF
C                                  INSERT DOT, BLANK, BLANK
         CALL M1VE (DBB, 1, 3, 3, MSGSAV, MS+1, MS+3, 255, IER)
         MS = MS + 3
C                                  LOOK AT NEXT CHARACTER
   20    M = M + 1
         ISUB = 0
         IF (M .GT. LEN2-4) THEN
            LAST = LEN2 - M + 1
            DO 30  I=1, LAST
   30       MSGSAV(MS+I) = MSGTMP(M+I-1)
            MSGLEN = MS + LAST
            GO TO 40
         ELSE IF (MSGTMP(M).EQ.PERCNT .AND. MSGTMP(M+1).EQ.LPAR .AND.
     &           MSGTMP(M+4).EQ.RPAR) THEN
            CALL C1TCI (MSGTMP(M+3), 1, NUM, IER)
            IF (IER.EQ.0 .AND. NUM.NE.0 .AND. I1DX(FORMS,NFORMS,
     &          MSGTMP(M+2),1).NE.0) THEN
C                                  LOCATE THE ITEM IN THE PARAMETER LIST
               CALL M1VE (MSGTMP(M+2), 1, 2, 2, FIND, 2, 3, 4, IER)
               LOC = I1DX(PLIST,PLEN,FIND,4)
               IF (LOC .GT. 0) THEN
C                                  SET IPLEN = LENGTH OF STRING
                  CALL C1TCI (PLIST(LOC+4), 4, IPLEN, IER)
                  PBEG = LOC + 4 + IER
C                                  ADJUST IPLEN IF IT IS TOO BIG
                  IPLEN = MIN0(IPLEN,255-MS)
C                                  M1VE STRING FROM PLIST INTO MSGSAV
                  CALL M1VE (PLIST, PBEG, PBEG+IPLEN-1, PLEN, MSGSAV,
     &                       MS+1, MS+IPLEN, 255, IER)
                  IF (IER.GE.0 .AND. IER.LT.IPLEN) THEN
C                                  UPDATE POINTERS
                     M = M + 4
                     MS = MS + IPLEN - IER
C                                  BAIL OUT IF NO MORE ROOM
                     IF (MS .GE. 255) THEN
                        MSGLEN = 255
                        GO TO 40
                     END IF
C                                  SET FLAG TO SHOW SUBSTITION WAS MADE
                     ISUB = 1
                  END IF
               END IF
            END IF
         END IF
         IF (ISUB .EQ. 0) THEN
            MS = MS + 1
            MSGSAV(MS) = MSGTMP(M)
         END IF
         GO TO 20
   40    ERTYP2 = ERTYPE(CALLVL)
         IF (ERTYP2 .EQ. 6) IFERR6 = 1
         IF (ERTYP2 .EQ. 7) IFERR7 = 1
C                                  UPDATE CHECKSUM PARAMETER ERCKSM
         CALL E1UCS
C                                  PRINT MESSAGE IF NECESSARY
         IF (ERTYP2.GE.5 .AND. PRINTB(ERTYP2).EQ.1) CALL E1PRT
      END IF
C                                  CLEAR PARAMETER LIST
      PLEN = 1
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1STI
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 6, 1984
C
C  Purpose:    To store an integer for subsequent use within an error
C              message.
C
C  Usage:      CALL E1STI(II, IVALUE)
C
C  Arguments:
C     II     - Integer specifying the substitution index.  II must be
C              between 1 and 9.  (Input)
C     IVALUE - The integer to be stored.  (Input)
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1STI (II, IVALUE)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    II, IVALUE
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    IBEG, IER, ILEN
      CHARACTER  ARRAY(14)
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    IFINIT
      CHARACTER  BLANK(1)
      SAVE       BLANK, IFINIT
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   C1TIC, E1INIT, E1INPL
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   I1ERIF
      INTEGER    I1ERIF
C
      DATA BLANK/' '/, IFINIT/0/
C                                  INITIALIZE IF NECESSARY
      IF (IFINIT .EQ. 0) THEN
         CALL E1INIT
         IFINIT = 1
      END IF
      CALL C1TIC (IVALUE, ARRAY, 14, IER)
      IBEG = I1ERIF(ARRAY,14,BLANK,1)
      IF (II.GE.1 .AND. II.LE.9 .AND. IER.EQ.0) THEN
         ILEN = 15 - IBEG
         CALL E1INPL ('I', II, ILEN, ARRAY(IBEG))
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1USR
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    November 2, 1984
C
C  Purpose:    Set USER CODE switch.
C
C  Usage:      CALL E1USR(SWITCH)
C
C  Arguments:
C     SWITCH - Character string.  (Input)
C                'ON'  Indicates that USER CODE mode is being entered.
C                'OFF' Indicates that USER CODE mode is being exited.
C  Remarks:
C     When E1POP is called from a routine while in USER CODE mode,
C     then an error message of type 1-4 will be printed (if an error
C     condition is in effect and the print table allows it).
C     However, an error message of type 1-4 will never be printed
C     if USER CODE mode is not in effect.
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1USR (SWITCH)
C                                  SPECIFICATIONS FOR ARGUMENTS
      CHARACTER  SWITCH*(*)
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    IFINIT
      SAVE       IFINIT
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1INIT, E1MES, E1STL
C
      DATA IFINIT/0/
C                                  INITIALIZE ERROR TABLE IF NECESSARY
      IF (IFINIT .EQ. 0) THEN
         CALL E1INIT
         IFINIT = 1
      END IF
      IF (SWITCH.EQ.'ON' .OR. SWITCH.EQ.'on') THEN
         ISUSER(CALLVL) = .TRUE.
      ELSE IF (SWITCH.EQ.'OFF' .OR. SWITCH.EQ.'off') THEN
         ISUSER(CALLVL) = .FALSE.
      ELSE
         CALL E1STL (1, SWITCH)
         CALL E1MES (5, 1, 'Invalid value for SWITCH in call to'//
     &               ' E1USR.  SWITCH must be set to ''ON'' or '//
     &               '''OFF''.  SWITCH = ''%(L1)'' ')
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U10NF/DU10NF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Solve (L*TRANS(L))*s = -g for s.
C
C  Usage:      CALL U10NF (N, H, LDH, GC, SNWTN)
C
C  Arguments:
C     N      - Length of the vectors GC, SNWTN. (Input)
C     H      - N by N matrix containing the Cholesky factor of the
C              Hessian in the lower triangle and diagonal.  (Input)
C     LDH    - Leading dimension of H exactly as specified in the
C              dimension statement of the calling program.  (Input)
C     GC     - Vector of length N containing the current gradient.
C              (Input)
C     SNWTN  - Vector of length N containing Newton's step.  (Output)
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
      SUBROUTINE U10NF (N, H, LDH, GC, SNWTN)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, LDH
      REAL       H(LDH,*), GC(*), SNWTN(N)
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   SSCAL, U13NF, U14NF
C
      CALL U13NF (N, H, LDH, GC, SNWTN)
C                                  Solve L**Ts=y
      CALL U14NF (N, H, LDH, SNWTN, SNWTN)
      CALL SSCAL (N, -1.0, SNWTN, 1)
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U9INF/DU9INF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Compute the Hessian at the initial point.
C
C  Usage:      CALL U9INF (N, FX0, FSCALE, XSCALE, IHESS, H, LDH)
C
C  Arguments:
C     N      - Order of H.  (Input)
C     FX0    - Function value at the initial guess.  (Input)
C     FSCALE - Scaling for the function.  (Input)
C     XSCALE - Real vector of length N containing the diagonal scaling
C              matrix for the variables.  (Input)
C     IHESS  - Hessian initialization parameter.  (Input)
C              If IHESS = 0 the Hessian is initialized to the identity
C              matrix, otherwise it is initialized to a diagonal matrix
C              containing the Cholesky factor on the diagonal.
C     H      - N by N matrix containing the initialized Hessian.
C              (Output)
C     LDH    - Row dimension of H exactly as specified in the dimension
C              statement of the calling program.  (Input)
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
      SUBROUTINE U9INF (N, FX0, FSCALE, XSCALE, IHESS, H, LDH)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, IHESS, LDH
      REAL       FX0, FSCALE, XSCALE(*), H(LDH,*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    J
      REAL       TEMP
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS,AMAX1,SQRT
      INTRINSIC  ABS, AMAX1, SQRT
      REAL       ABS, AMAX1, SQRT
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   SSET
C
      TEMP = SQRT(AMAX1(ABS(FX0),FSCALE))
      DO 10  J=1, N
         CALL SSET (N, 0.0E0, H(1,J), 1)
         IF (IHESS .EQ. 0) THEN
            H(J,J) = 1.0E0
         ELSE
            H(J,J) = TEMP*XSCALE(J)
         END IF
   10 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U11NF/DU11NF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Compute X = Y**K * Z.
C
C  Usage:      CALL U11NF (N, Y, K, Z, X)
C
C  Arguments:
C     N      - Length of the vectors X, Y and Z.  (Input)
C     Y      - Vector of length N.  (Input)
C     K      - Integer specifying the exponent.  (Input)
C     Z      - Vector of length N.  (Input)
C     X      - Vector of length N.  (Output)
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
      SUBROUTINE U11NF (N, Y, K, Z, X)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, K
      REAL       Y(*), Z(*), X(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I
C
      IF (K .LT. 0) THEN
         IF (K .EQ. -1) THEN
            DO 10  I=1, N
               X(I) = Z(I)/Y(I)
   10       CONTINUE
         ELSE
            DO 20  I=1, N
               X(I) = Z(I)/(Y(I)**(-K))
   20       CONTINUE
         END IF
      ELSE
         IF (K .EQ. 1) THEN
            DO 30  I=1, N
               X(I) = Z(I)*Y(I)
   30       CONTINUE
         ELSE
            DO 40  I=1, N
   40       X(I) = Y(I)**K*Z(I)
         END IF
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1STR
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 2, 1984
C
C  Purpose:    To store a real number for subsequent use within an error
C              message.
C
C  Usage:      CALL E1STR(IR,RVALUE)
C
C  Arguments:
C     IR     - Integer specifying the substitution index.  IR must be
C              between 1 and 9.  (Input)
C     RVALUE - The real number to be stored.  (Input)
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1STR (IR, RVALUE)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    IR
      REAL       RVALUE
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IBEG, ILEN
      CHARACTER  ARRAY(14), SAVE*14
      REAL       RNINF
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    IFINIT
      CHARACTER  BLANK(1)
      SAVE       BLANK, IFINIT
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1INIT, E1INPL
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   AMACH, I1ERIF
      REAL       AMACH
      INTEGER    I1ERIF
C
      DATA BLANK/' '/, IFINIT/0/
C                                  INITIALIZE IF NECESSARY
      IF (IFINIT .EQ. 0) THEN
         CALL E1INIT
         RNINF = AMACH(8)
         IFINIT = 1
      END IF
      IF (RVALUE .EQ. 0.0) THEN
         WRITE (SAVE,'(E14.6)') RVALUE
      ELSE
         IF (RVALUE .EQ. RNINF) THEN
            WRITE (SAVE,'(E4.4)') RVALUE
         ELSE
            WRITE (SAVE,'(1PE14.6)') RVALUE
         END IF
      END IF
      DO 40  I=1, 14
   40 ARRAY(I) = SAVE(I:I)
      IBEG = I1ERIF(ARRAY,14,BLANK,1)
      IF (IR.GE.1 .AND. IR.LE.9) THEN
         ILEN = 15 - IBEG
         CALL E1INPL ('R', IR, ILEN, ARRAY(IBEG))
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  CRGRG/DCRGRG (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    June 5, 1985
C
C  Purpose:    Copy a real general matrix.
C
C  Usage:      CALL CRGRG (N, A, LDA, B, LDB)
C
C  Arguments:
C     N      - Order of the matrices.  (Input)
C     A      - Matrix of order N.  (Input)
C     LDA    - Leading dimension of A exactly as specified in the
C              dimension statement of the calling program.  (Input)
C     B      - Matrix of order N containing a copy of A.  (Output)
C     LDB    - Leading dimension of B exactly as specified in the
C              dimension statement of the calling program.  (Input)
C
C  GAMS:       D1b8
C
C  Chapter:    MATH/LIBRARY Basic Matrix/Vector Operations
C
C  Copyright:  1985 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE CRGRG (N, A, LDA, B, LDB)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, LDA, LDB
      REAL       A(LDA,*), B(LDB,*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    J
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, E1STI, SCOPY
C
      CALL E1PSH ('CRGRG ')
C                                  Check N
      IF (N .LT. 1) THEN
         CALL E1STI (1, N)
         CALL E1MES (5, 1, 'The argument N = %(I1).  It must be at '//
     &               'least 1.')
         GO TO 9000
      END IF
C                                  Check LDA
      IF (LDA .LT. N) THEN
         CALL E1STI (1, LDA)
         CALL E1STI (2, N)
         CALL E1MES (5, 2, 'The argument LDA = %(I1).  It must be at '//
     &               'least as large as N = %(I2).')
         GO TO 9000
      END IF
C                                  Check LDB
      IF (LDB .LT. N) THEN
         CALL E1STI (1, LDB)
         CALL E1STI (2, N)
         CALL E1MES (5, 3, 'The argument LDB = %(I1).  It must be at '//
     &               'least as large as N = %(I2).')
         GO TO 9000
      END IF
C                                  Copy
      IF (LDA.EQ.N .AND. LDB.EQ.N) THEN
         CALL SCOPY (N*N, A, 1, B, 1)
      ELSE IF (LDA .GE. LDB) THEN
         DO 10  J=1, N
            CALL SCOPY (N, A(1,J), 1, B(1,J), 1)
   10    CONTINUE
      ELSE
         DO 20  J=N, 1, -1
            CALL SCOPY (N, A(1,J), -1, B(1,J), -1)
   20    CONTINUE
      END IF
C
 9000 CONTINUE
      CALL E1POP ('CRGRG ')
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
      SUBROUTINE U6INF (N, XP, SC, FP, GP, XSCALE, FSCALE, ICODE,
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
C
C                                  TEST OF NORM OF SCALED GRADIENT
      VALMAX = 0.0E0
      DO 10  I=1, N
         SCGRAD = ABS(GP(I))*AMAX1(ABS(XP(I)),1.0E0/XSCALE(I))/
     &            AMAX1(ABS(FP),FSCALE)
         VALMAX = AMAX1(SCGRAD,VALMAX)
   10 CONTINUE
      IF (VALMAX .LE. GRADTL) THEN
         ICODE = -999
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
         CALL E1MES (3, 8, 'The last global step failed to '//
     &               'locate a lower point than the current X '//
     &               'value.  The current X may be an approximate '//
     &               'local minimizer and no more accuracy is '//
     &               'possible or the step tolerance may be too '//
     &               'large where STEPTL = %(R1) is given.')
         GO TO 9000
      END IF
C                                  TEST NORM OF SCALED STEP
      VALMAX = 0.0E0
      DO 20  I=1, N
         SCSTEP = ABS(SC(I))/AMAX1(ABS(XP(I)),1.0E0/XSCALE(I))
         VALMAX = AMAX1(SCSTEP,VALMAX)
   20 CONTINUE
      IF (VALMAX .LE. STEPTL) THEN
         ICODE = -999
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
         CALL U18NF (3)
CC         CALL E1MES (4, 3, ' Maximum number of iterations '//
CC     &               'exceeded.')
C
      ELSE IF (NFCN .GE. MAXFCN) THEN
C                                  Print error message
         CALL U18NF (4)
CC         CALL E1MES (4, 4, ' Maximum number of function evaluations'//
CC     &               ' exceeded.')
C
      ELSE IF (NGRAD .GE. MAXGRD) THEN
C                                  Print error message
         CALL U18NF (5)
CC         CALL E1MES (4, 5, ' Maximum number of gradient evaluations'//
CC     &               ' exceeded.')
C
      ELSE IF (USRHES .AND. (NHESS.GE.MAXHES)) THEN
C                                  Print error message
         CALL U18NF (7)
CC         CALL E1MES (4, 7, 'Maximum number of Hessian evaluations '//
CC     &               'exceeded.')
C
      ELSE IF (MXTAKE) THEN
         NMAXS = NMAXS + 1
         IF (NMAXS .EQ. 5) THEN
C                                  Print error message
            CALL U18NF (6)
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
C-----------------------------------------------------------------------
C  IMSL Name:  B8ONF/DB8ONF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Determine if the free variables have satisified the
C              stopping criterion.
C
C  Usage:      CALL B8ONF (N, IACT, G, DONE)
C
C  Arguments:
C     N      - The size fo the problem. (Input)
C     IACT   - Integer vector of length N indicating whether X(I) has
C              been moved to an upper or lower bound.  (Input)
C     G      - Vector of length N containing the current gradient.
C              (Input)
C     DONE   - Logical variable indicating if the stopping criterion for
C              the free variables has been satisfied.  (Output)
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
      SUBROUTINE B8ONF (N, IACT, G, DONE)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, IACT(*)
      REAL       G(*)
      LOGICAL    DONE
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, ITMP
      REAL       GTMP
C
      DO 10  I=1, N
         ITMP = IACT(I)
         GTMP = G(I)
         IF (ITMP.EQ.1 .AND. GTMP.LT.0.0E0) THEN
            IACT(I) = 0
            DONE = .FALSE.
            GO TO 9000
         ELSE IF (ITMP.EQ.2 .AND. GTMP.GT.0.0E0) THEN
            IACT(I) = 0
            DONE = .FALSE.
            GO TO 9000
         END IF
   10 CONTINUE
C
 9000 RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  N1RTY
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 6, 1984
C
C  Purpose:    Retrieve an error type.
C
C  Usage:      N1RTY(IOPT)
C
C  Arguments:
C     IOPT   - Integer specifying the level.  (Input)
C              If IOPT=0 the error type for the current level is
C              returned.  If IOPT=1 the error type for the most
C              recently called routine (last pop) is returned.
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION N1RTY (IOPT)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    IOPT
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1PRT, M1VECH
C
      IF (IOPT.NE.0 .AND. IOPT.NE.1) THEN
         ERTYPE(CALLVL) = 5
         ERCODE(CALLVL) = 1
         MSGLEN = 47
         CALL M1VECH ('.  The argument passed to N1RTY must be 0 or '//
     &                '1. ', MSGLEN, MSGSAV, MSGLEN)
         CALL E1PRT
         STOP
      ELSE
         N1RTY = ERTYPE(CALLVL+IOPT)
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U5INF/DU5INF  (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Check validity of input to unconstrained minimization.
C
C  Usage:      CALL U5INF (N, XC, XSCALE, FSCALE, USRHES, IPARAM,
C                          RPARAM)
C
C  Arguments:
C     N      - Size of the problem.  (Input)
C     XC     - Vector of length N containing the initial point.  (Input)
C     XSCALE - Vector of length N containing the diagonal scaling matrix
C              for the variables.  (Input/Output)
C     FSCALE - Estimate of the scale of the objective function.
C              (Input/Output)
C     USRHES - Logical variable.  (Input)
C              USRHES = .TRUE. if analytic Hessian or finite difference
C                       Hessian is used.
C              USRHES = .FALSE. otherwise.
C     IPARAM - Integer parameters vector of length 6.  (Input/Output)
C              See UMINF or UMIDH for details.
C     RPARAM - Real parameters vector of length 7.  (Input/Output)
C              See UMINF or UMIDH for details.
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
      SUBROUTINE U5INF (N, XC, XSCALE, FSCALE, USRHES, IPARAM, RPARAM)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, IPARAM(*)
      REAL       FSCALE, XC(*), XSCALE(*), RPARAM(*)
      LOGICAL    USRHES
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I
      REAL       MACHEP, TEMP, TEMP1, TEMP2
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  AMAX1,SQRT
      INTRINSIC  AMAX1, SQRT
      REAL       AMAX1, SQRT
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, E1STI, E1STR, SSET, U19NF
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   AMACH, IMACH, SNRM2
      INTEGER    IMACH
      REAL       AMACH, SNRM2
C
      CALL E1PSH ('U5INF ')
      MACHEP = AMACH(4)
C                                  FDIGIT  good digits in F
C                                  MXITER  Min number of iterations
C                                  MAXFCN  Max fcn. evaluations
C                                  MAXGRD  Max grad. evaluations
C                                  IHESS   Hessian initial. parameter
C                                  MAXHES  Max Hess. evaluations
C                                  GRADTL  Scaled gradient tolerance
C                                  STEPTL  Scaled step tolerance
C                                  RFTOL   Relative function tolerance
C                                  AFTOL   Absolute function tolerance
C                                  FALSTL  False convergence tolerance
C                                  STEPMX  Maximum allowable step size
C                                  DELTA   Size of initial trust region
      IF (N .LE. 0) THEN
         CALL E1STI (1, N)
C                                  Print error message
         CALL U19NF (0)
CC         CALL E1MES (5, 1, 'The size of the problem must be '//
CC     &               'positive while N = %(I1) is given.')
         GO TO 9000
      ELSE IF (N .EQ. 1) THEN
C                                  Print error message
         CALL U19NF (1)
CC         CALL E1MES (6, 1, 'This routine may be inefficient '//
CC     &               'for a problem of size N = 1.')
      END IF
C                                  CHECK VARIABLE SCALING MATRIX
      DO 10  I=1, N
         IF (XSCALE(I) .LE. 0.0E0) GO TO 20
   10 CONTINUE
      GO TO 30
C                                  Print error message
   20 CALL U19NF (2)
CC   20 CALL E1MES (6, 2, 'The diagonal scaling matrix for '//
CC     &            'the variables must be positive while some '//
CC     &            'of the entries are less than or equal to zero.'//
CC     &            '  The algorithm will use the identity scaling '//
CC     &            'matrix for XSCALE.')
      CALL SSET (N, 1.0E0, XSCALE, 1)
C                                  CHECK FUNCTION SCALING
   30 IF (FSCALE .LE. 0.0E0) THEN
         CALL E1STR (1, FSCALE)
         CALL E1MES (6, 3, 'The estimate of the scale of the '//
     &               'objective function must be positive while '//
     &               'FSCALE = %(R1) is given.  The algorithm will '//
     &               'use FSCALE = 1.0.')
         FSCALE = 1.0E0
      END IF
C                                  CHECK ACCURACY OF THE PROBLEM
      IF (IPARAM(2) .LE. 0) THEN
         CALL E1STI (1, IPARAM(2))
C                                  Print error message
         CALL U19NF (4)
CC         CALL E1MES (6, 4, 'The estimate of the number of '//
CC     &               'good digits in the function must be positive '//
CC     &               'while FDIGIT = %(I1) is given.  The algorithm'//
CC     &               ' will assume that the function is accurate to'//
CC     &               ' the precision of the arithmetic.')
         IPARAM(2) = IMACH(7)
      END IF
C                                  CHECK MAXIMUM NUMBER OF ITERATIONS
      IF (IPARAM(3) .LE. 0) THEN
         CALL E1STI (1, IPARAM(3))
C                                  Print error message
         CALL U19NF (5)
CC         CALL E1MES (6, 5, 'The maximum number of iterations '//
CC     &               'must be positive while MXITER = %(I1) is '//
CC     &               'given.  The algorithm will use MXITER = 100.')
         IPARAM(3) = 100
      END IF
C                                  CHECK MAXIMUM FUNCTION EVALUATIONS
      IF (IPARAM(4) .LE. 0) THEN
         CALL E1STI (1, IPARAM(4))
C                                  Print error message
         CALL U19NF (6)
CC         CALL E1MES (6, 6, 'The maximum number of function '//
CC     &               'evaluations must be positive while MAXFCN = '//
CC     &               '%(I1) is given.  The algorithm will use '//
CC     &               'MAXFCN = 400.')
         IPARAM(4) = 400
      END IF
C                                  CHECK MAXIMUM GRADIENT EVALUATIONS
      IF (IPARAM(5) .LE. 0) THEN
         CALL E1STI (1, IPARAM(5))
C                                  Print error message
         CALL U19NF (7)
CC         CALL E1MES (6, 7, 'The maximum number of gradient '//
CC     &               'evaluations must be positive while MAXGRD = '//
CC     &               '%(I1) is given.  The algorithm will use '//
CC     &               'MAXGRD = 400.')
         IPARAM(5) = 400
      END IF
C                                  CHECK MAXIMUM HESSIAN EVALUATIONS IF
C                                  A NEWTON METHOD IS USED
      IF (USRHES) THEN
         IF (IPARAM(7) .LE. 0) THEN
            CALL E1STI (1, IPARAM(7))
C                                  Print error message
            CALL U19NF (8)
CC            CALL E1MES (6, 8, 'The maximum number of Hessian '//
CC     &                  'evaluations must be positive while MAXHES '//
CC     &                  '= %(I1) is given.  The algorithm will use '//
CC     &                  'MAXHES = 100.')
            IPARAM(7) = 100
         END IF
      END IF
C
      TEMP1 = MACHEP**(2.0E0/3.0E0)
      TEMP2 = MACHEP**(2.0E0/3.0E0)
C                                  CHECK THE GRADIENT TOLERANCE
      IF (RPARAM(1) .LT. 0.0E0) THEN
         CALL E1STR (1, RPARAM(1))
         CALL E1STR (2, TEMP1)
         CALL E1MES (6, 9, 'The gradient tolerance must be '//
     &               'nonnegative while GRADTL = %(R1) is given.  '//
     &               'The algorithm will use GRADTL = %(R2).')
         RPARAM(1) = TEMP1
      END IF
C                                  CHECK THE STEP TOLERANCE
      IF (RPARAM(2) .LT. 0.0E0) THEN
         CALL E1STR (1, RPARAM(2))
         CALL E1STR (2, TEMP2)
         CALL E1MES (6, 10, 'The step tolerance must be nonnegative '//
     &               'while STEPTL = %(R1) is given.  The algorithm '//
     &               'will use STEPTL = %(R2).')
         RPARAM(2) = TEMP2
      END IF
C                                  CHECK RELATIVE FUNCTION TOLERANCE
      IF (RPARAM(3) .LT. 0.0E0) THEN
         TEMP = AMAX1(1.0E-10,TEMP2)
         CALL E1STR (1, RPARAM(3))
         CALL E1STR (2, TEMP)
         CALL E1MES (6, 11, 'The relative function tolerance '//
     &               'must be nonnegative while RFTOL = %(R1) is '//
     &               'given.  The algorithm will use RFTOL = %(R2).')
         RPARAM(3) = TEMP
      END IF
C                                  CHECK ABSOLUTE FUNCTION TOLERANCE
      IF (RPARAM(4) .LT. 0.0E0) THEN
         TEMP = AMAX1(1.0E-20,MACHEP*MACHEP)
         CALL E1STR (1, RPARAM(4))
         CALL E1STR (2, TEMP)
         CALL E1MES (6, 12, 'The absolute function tolerance '//
     &               'must be nonnegative while AFTOL = %(R1) is '//
     &               'given.  The algorithm will use AFTOL = %(R2).')
         RPARAM(4) = TEMP
      END IF
C                                  CHECK FALSE CONVERGENCE TOLERANCE
      IF (RPARAM(5) .LT. 0.0E0) THEN
         TEMP = 1.0E2*MACHEP
         CALL E1STR (1, RPARAM(5))
         CALL E1STR (2, TEMP)
         CALL E1MES (6, 13, 'The false convergence tolerance '//
     &               'must be nonnegative while FALSTL = %(R1) is '//
     &               'given.  The algorithm will use FALSTL = %(R2).')
         RPARAM(5) = TEMP
      END IF
C                                  CHECK MAXIMUM ALLOWED STEP SIZE
      IF (RPARAM(6) .LE. 0.0E0) THEN
         TEMP1 = 0.0E0
         DO 40  I=1, N
            TEMP1 = TEMP1 + (XSCALE(I)*XC(I))**2
   40    CONTINUE
         TEMP1 = SQRT(TEMP1)
         TEMP2 = SNRM2(N,XSCALE,1)
         TEMP = 1.0E3*AMAX1(TEMP1,TEMP2)
         IF (IPARAM(1).NE.0 .AND. RPARAM(6).NE.-9999.0E0) THEN
            CALL E1STR (1, RPARAM(6))
            CALL E1STR (2, TEMP)
            CALL E1MES (6, 14, 'The maximum allowable scaled '//
     &                  'step length must be positive while STEPMX = '//
     &                  '%(R1) is given.  The algorithm will use '//
     &                  'STEPMX = %(R2).')
         END IF
         RPARAM(6) = TEMP
      END IF
C                                  CHECK INITIAL TRUST REGION RADIUS
      IF (RPARAM(7) .LE. 0.0E0) THEN
         IF (IPARAM(1).NE.0 .AND. RPARAM(7).NE.-9999.0E0) THEN
            CALL E1STR (1, RPARAM(7))
            CALL E1MES (6, 15, 'The initial trust region '//
     &                  'radius must be positive while DELTA = %(R1) '//
     &                  'is given.  The algorithm will use the '//
     &                  'length of the initial scaled Cauchy step '//
     &                  'for DELTA.')
         END IF
         RPARAM(7) = -1.0E0
      END IF
C
 9000 CALL E1POP ('U5INF ')
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  B7ONF/DB7ONF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Change Hessian and dimension for Cholesky factorization.
C
C  Usage:      CALL B7ONF (N, IACT, H, LDH, NACT)
C
C  Arguments:
C     N      - The size fo the problem.  (Input)
C     IACT   - Integer vector of length N indicating whether X(I) has
C              been moved to an upper or lower bound.  (Input)
C     H      - Matrix of dimension N by N containing the current
C              Hessian.  (Input/Output)
C              On output H will be in the proper form for the Cholesky
C              factorization.
C     LDH    - Leading dimension of H exactly as specified in the
C              dimension statement of the calling program.  (Input)
C     NACT   - Integer scalar indicating the final dimension of H.
C                 (Output)
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
      SUBROUTINE B7ONF (N, IACT, H, LDH, NACT)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, LDH, NACT, IACT(*)
      REAL       H(LDH,*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, J
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   SCOPY
C
      J = 0
      NACT = 0
      DO 10  I=1, N
         IF (IACT(I) .NE. 0) THEN
            NACT = NACT + 1
            IF (J .EQ. 0) J = I
         ELSE
            IF (J .NE. 0) THEN
               CALL SCOPY (N, H(I,1), LDH, H(J,1), LDH)
               CALL SCOPY (N, H(1,I), 1, H(1,J), 1)
               J = J + 1
            END IF
         END IF
   10 CONTINUE
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U4INF/DU4INF  (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Provide default parameters for the minimization problem.
C
C  Usage:      CALL U4INF (IPARAM, RPARAM)
C
C  Arguments:
C     IPARAM - Integer parameters vector of length 6.  (Output)
C              IPARAM(1) = initialization flag
C              IPARAM(2) = number of good digits in the function
C              IPARAM(3) = maximum number of iterations.
C              IPARAM(4) = maximum number of function evaluations
C              IPARAM(5) = maximum number of gradient evaluations
C              IPARAM(6) = initialization of Hessian parameter
C              IPARAM(7) = maximum number of Hessian evaluations
C     RPARAM - Integer parameters vector of length 7.  (Output)
C              RPARAM(1) = scaled gradient tolerance
C              RPARAM(2) = scaled step tolerance
C              RPARAM(3) = relative function tolerance
C              RPARAM(4) = absolute function tolerance
C              RPARAM(5) = false convergence tolerance
C              RPARAM(6) = maximum allowable step size
C              RPARAM(7) = size of initial trust region radius
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
      SUBROUTINE U4INF (IPARAM, RPARAM)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    IPARAM(*)
      REAL       RPARAM(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      REAL       ARFTOL, EPS, OV3, TV3
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  AMAX1
      INTRINSIC  AMAX1
      REAL       AMAX1
C     INTRINSIC  SQRT
      INTRINSIC  SQRT
      REAL       SQRT
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   AMACH, IMACH
      INTEGER    IMACH
      REAL       AMACH
C
      EPS = AMACH(4)
      OV3 = 1.0E0/3.0E0
      TV3 = 2.0E0/3.0E0
C                                  Set up integer parameters
      IPARAM(1) = 1
      IPARAM(2) = IMACH(7)
      RPARAM(1) = SQRT(EPS)
      ARFTOL = 1.0E-10
      IPARAM(3) = 100
      IPARAM(4) = 400
      IPARAM(5) = 400
      IPARAM(6) = 0
      IPARAM(7) = 100
C                                  Set up real parameters
      RPARAM(2) = EPS**TV3
      RPARAM(3) = AMAX1(ARFTOL,EPS**TV3)
      RPARAM(4) = AMAX1(ARFTOL,EPS*EPS)
      RPARAM(5) = 1.0E2*EPS
      RPARAM(6) = -9999.0E0
      RPARAM(7) = -9999.0E0
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  B6ONF/DB6ONF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    February 1, 1988
C
C  Purpose:    Check XLB(I) .le. XC(I) .ge. XUB(I).
C
C  Usage:      CALL B6ONF (N, XC, XLB, XUB, GC, IACT, EPS, CHANGE)
C
C  Arguments:
C     N      - The size fo the problem.  (Input)
C     XC     - Real vector of length N containing the current point.
C                 (Input)
C     XLB    - Real vector of length N containing the lower bounds.
C                 (Input)
C     XUB    - Real vector of length N containing the upper bounds.
C                 (Input)
C     GC     - Real vector of length N containing the current gradient.
C                 (Input)
C     IACT   - Integer vector of length N indicating if XC(I) had to be
C              moved to an upper or lower bound.  (Input)
C     EPS    - An estimated bound on the relative noise in the function
C              value.  (Input)
C     CHANGE - Logical variable indicating if any change occurred.
C                 (Input)
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
      SUBROUTINE B6ONF (N, XC, XLB, XUB, GC, IACT, EPS, CHANGE)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, IACT(*)
      REAL       EPS, XC(*), XLB(*), XUB(*), GC(*)
      LOGICAL    CHANGE
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I
      REAL       AEPS
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS
      INTRINSIC  ABS
      REAL       ABS
C                                  SET UP IACT
      AEPS = EPS
      DO 10  I=1, N
         IF (ABS(XC(I)-XLB(I)) .LE. AEPS) THEN
            IACT(I) = 1
            CHANGE = .TRUE.
         ELSE
            IF (ABS(XC(I)-XUB(I)) .LE. AEPS) THEN
               IACT(I) = 2
               CHANGE = .TRUE.
            ELSE
               IACT(I) = 0
            END IF
         END IF
   10 CONTINUE
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  I1KGT
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    January 17, 1984
C
C  Purpose:    Allocate numerical workspace.
C
C  Usage:      I1KGT(NELMTS,ITYPE)
C
C  Arguments:
C     NELMTS - Number of elements of data type ITYPE to be
C              allocated.  (Input)
C     ITYPE  - Data type of array to be allocated.  (Input)
C                 1 - logical
C                 2 - integer
C                 3 - real
C                 4 - double precision
C                 5 - complex
C                 6 - double complex
C     I1KGT  - Integer function.  (Output)  Returns the index of the
C              first element in the current allocation.
C
C  Remarks:
C  1. On return, the array will occupy
C     WKSP(I1KGT), WKSP(I1KGT+1), ..., WKSP(I1KGT+NELMTS-1) where
C     WKSP is an array of data type ITYPE equivalenced to RWKSP.
C
C  2. If I1KGT is negative, the absolute value of I1KGT is the
C     additional workspace needed for the current allocation.
C
C  3. The allocator reserves the first sixteen integer locations of
C     the stack for its own internal bookkeeping.  These are initialized
C     by the function IWKIN upon the first call to the allocation
C     package.
C
C  4. The use of the first ten integer locations is as follows:
C      WKSP( 1) - LOUT    The number of current allocations
C      WKSP( 2) - LNOW    The current active length of the stack
C      WKSP( 3) - LUSED   The maximum value of WKSP(2) achieved
C                         thus far
C      WKSP( 4) - LBND    The lower bound of permanent storage which
C                         is one numeric storage unit more than the
C                         maximum allowed length of the stack.
C      WKSP( 5) - LMAX    The maximum length of the storage array
C      WKSP( 6) - LALC    The total number of allocations handled by
C                         I1KGT
C      WKSP( 7) - LNEED   The number of numeric storage units by which
C                         the array size must be increased for all past
C                         allocations to succeed
C      WKSP( 8) - LBOOK   The number of numeric storage units used for
C                         bookkeeping
C      WKSP( 9) - LCHAR   The pointer to the portion of the permanent
C                         stack which contains the bookkeeping and
C                         pointers for the character workspace
C                         allocation.
C      WKSP(10) - LLCHAR  The length of the array beginning at LCHAR
C                         set aside for character workspace bookkeeping
C                         and pointers.
C                 NOTE -  If character workspace is not being used,
C                         LCHAR and LLCHAR can be ignored.
C  5. The next six integer locations contain values describing the
C     amount of storage allocated by the allocation system to the
C     various data types.
C      WKSP(11) - Numeric storage units allocated to LOGICAL
C      WKSP(12) - Numeric storage units allocated to INTEGER
C      WKSP(13) - Numeric storage units allocated to REAL
C      WKSP(14) - Numeric storage units allocated to DOUBLE PRECISION
C      WKSP(15) - Numeric storage units allocated to COMPLEX
C      WKSP(16) - Numeric storage units allocated to DOUBLE COMPLEX
C
C  Copyright:  1984 by IMSL, Inc. All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION I1KGT (NELMTS, ITYPE)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    NELMTS, ITYPE
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IDUMAL, IGAP, ILEFT, IPA, IPA7, ISA, ISA7,
     &           ISIZE(6), JTYPE, LALC, LBND, LBOOK, LMAX, LNEED,
     &           LNEED1, LNOW, LOUT, LUSED
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      LOGICAL    FIRST
      SAVE       FIRST
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM8/
      INTEGER    PROLVL, XXLINE(10), XXPLEN(10), ICALOC(10), INALOC(10)
      COMMON     /ERCOM8/ PROLVL, XXLINE, XXPLEN, ICALOC, INALOC
      SAVE       /ERCOM8/
C                              SPECIFICATIONS FOR COMMON /ERCOM9/
      CHARACTER  XXPROC(10)*31
      COMMON     /ERCOM9/ XXPROC
      SAVE       /ERCOM9/
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
C                                  SPECIFICATIONS FOR EQUIVALENCE
      EQUIVALENCE (LOUT, IWKSP(1))
      EQUIVALENCE (LNOW, IWKSP(2))
      EQUIVALENCE (LUSED, IWKSP(3))
      EQUIVALENCE (LBND, IWKSP(4))
      EQUIVALENCE (LMAX, IWKSP(5))
      EQUIVALENCE (LALC, IWKSP(6))
      EQUIVALENCE (LNEED, IWKSP(7))
      EQUIVALENCE (LBOOK, IWKSP(8))
      EQUIVALENCE (ISIZE(1), IWKSP(11))
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  IABS,MAX0,MOD
      INTRINSIC  IABS, MAX0, MOD
      INTEGER    IABS, MAX0, MOD
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1POS, E1PSH, E1STI, IWKIN
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   I1KQU
      INTEGER    I1KQU
C
      DATA FIRST/.TRUE./
C
      CALL E1PSH ('I1KGT ')
C
      IF (FIRST) THEN
C                                  INITIALIZE WORKSPACE IF NEEDED
         FIRST = .FALSE.
         CALL IWKIN (0)
      END IF
C                                  NUMBER OF ELEMENTS LESS THAN 0
      IF (NELMTS .LT. 0) THEN
         CALL E1STI (1, NELMTS)
         CALL E1MES (5, 2, 'Number of elements is not positive.%/'//
     &               'NELMTS = %(I1).')
         CALL E1POP ('I1KGT ')
         GO TO 9000
      END IF
C                                  ILLEGAL DATA TYPE REQUESTED
      IF (ITYPE.EQ.0 .OR. IABS(ITYPE).GE.7) THEN
         CALL E1MES (5, 3, 'Illegal data type requested.')
         CALL E1POP ('I1KGT ')
         GO TO 9000
      END IF
C                                  BOOKKEEPING OVERWRITTEN
      IF (LNOW.LT.LBOOK .OR. LNOW.GT.LUSED .OR. LUSED.GT.LMAX .OR.
     &    LNOW.GE.LBND .OR. LOUT.GT.LALC) THEN
         CALL E1MES (5, 4, 'One or more of the first eight '//
     &               'bookkeeping locations in IWKSP have been '//
     &               'overwritten.')
         CALL E1POP ('I1KGT ')
         GO TO 9000
      END IF
C
      CALL E1POP ('I1KGT ')
C                                  DETERMINE NUMBER OF LOCATIONS STILL
C                                  AVAILABLE FOR DATA TYPE ITYPE
C                                  NOTE: I1KQU ALLOWS FOR 2 INTEGER
C                                        POINTERS WHICH MUST BE HANDLED
C                                        ARTIFICIALLY IF ILEFT = 0.
      ILEFT = I1KQU(IABS(ITYPE))
C
      IF (ITYPE .GT. 0) THEN
C                                  RELEASABLE STORAGE
         IF (ILEFT .GE. NELMTS) THEN
            I1KGT = (LNOW*ISIZE(2)-1)/ISIZE(ITYPE) + 2
            I = ((I1KGT-1+NELMTS)*ISIZE(ITYPE)-1)/ISIZE(2) + 3
C                                  IWKSP(I-1) CONTAINS THE DATA TYPE FOR
C                                  THIS ALLOCATION. IWKSP(I) CONTAINS
C                                  LNOW FOR THE PREVIOUS ALLOCATION.
            IWKSP(I-1) = ITYPE
            IWKSP(I) = LNOW
            LOUT = LOUT + 1
            LALC = LALC + 1
            LNOW = I
            LUSED = MAX0(LUSED,LNOW)
            LNEED = 0
         ELSE
C                                  RELEASABLE STORAGE WAS REQUESTED
C                                  BUT THE STACK WOULD OVERFLOW.
C                                  THEREFORE, ALLOCATE RELEASABLE
C                                  SPACE THROUGH THE END OF THE STACK
            IF (LNEED .EQ. 0) THEN
               IDUMAL = (LNOW*ISIZE(2)-1)/ISIZE(ITYPE) + 2
               I = ((IDUMAL-1+ILEFT)*ISIZE(ITYPE)-1)/ISIZE(2) + 3
C                                  ADVANCE COUNTERS AND STORE POINTERS
C                                  IF THERE IS ROOM TO DO SO
               IF (I .LT. LBND) THEN
C                                  IWKSP(I-1) CONTAINS THE DATA TYPE FOR
C                                  THIS ALLOCATION. IWKSP(I) CONTAINS
C                                  LNOW FOR THE PREVIOUS ALLOCATION.
                  IWKSP(I-1) = ITYPE
                  IWKSP(I) = LNOW
                  LOUT = LOUT + 1
                  LALC = LALC + 1
                  LNOW = I
                  LUSED = MAX0(LUSED,LNOW)
               END IF
            END IF
C                                  CALCULATE AMOUNT NEEDED TO ACCOMODATE
C                                  THIS ALLOCATION REQUEST
            LNEED1 = (NELMTS-ILEFT)*ISIZE(ITYPE)
            IF (ILEFT .EQ. 0) THEN
               IGAP = ISIZE(ITYPE) - MOD(LNOW+LNEED,ISIZE(ITYPE))
               IF (IGAP .EQ. ISIZE(ITYPE)) IGAP = 0
               LNEED1 = LNEED1 + 2*ISIZE(2) + IGAP
            END IF
C                                  MODIFY LNEED ACCORDING TO THE SIZE
C                                  OF THE BASE BEING USED (D.P. HERE)
            LNEED = LNEED + ((LNEED1+ISIZE(3)-1)/ISIZE(3))
C                                  SINCE CURRENT ALLOCATION IS ILLEGAL,
C                                  RETURN THE NEGATIVE OF THE ADDITIONAL
C                                  AMOUNT NEEDED TO MAKE IT LEGAL
            I1KGT = -LNEED
         END IF
      ELSE
C                                  PERMANENT STORAGE
         IF (ILEFT .GE. NELMTS) THEN
            JTYPE = -ITYPE
            I1KGT = (LBND*ISIZE(2)-1)/ISIZE(JTYPE) + 1 - NELMTS
            I = ((I1KGT-1)*ISIZE(JTYPE))/ISIZE(2) - 1
C                                  IWKSP(I) CONTAINS LBND FOR PREVIOUS
C                                  PERMANENT STORAGE ALLOCATION.
C                                  IWKSP(I+1) CONTAINS THE DATA TYPE FOR
C                                  THIS ALLOCATION.
            IWKSP(I) = LBND
            IWKSP(I+1) = JTYPE
            LALC = LALC + 1
            LBND = I
            LNEED = 0
         ELSE
C                                  PERMANENT STORAGE WAS REQUESTED
C                                  BUT THE STACK WOULD OVERFLOW,
C                                  THEREFORE, ALLOCATE RELEASABLE
C                                  SPACE THROUGH THE END OF THE STACK
            IF (LNEED .EQ. 0) THEN
               JTYPE = -ITYPE
               IDUMAL = (LNOW*ISIZE(2)-1)/ISIZE(JTYPE) + 2
               I = ((IDUMAL-1+ILEFT)*ISIZE(JTYPE)-1)/ISIZE(2) + 3
C                                  ADVANCE COUNTERS AND STORE POINTERS
C                                  IF THERE IS ROOM TO DO SO
               IF (I .LT. LBND) THEN
C                                  IWKSP(I-1) CONTAINS THE DATA TYPE FOR
C                                  THIS ALLOCATION. IWKSP(I) CONTAINS
C                                  LNOW FOR THE PREVIOUS ALLOCATION.
                  IWKSP(I-1) = JTYPE
                  IWKSP(I) = LNOW
                  LOUT = LOUT + 1
                  LALC = LALC + 1
                  LNOW = I
                  LUSED = MAX0(LUSED,LNOW)
               END IF
            END IF
C                                  CALCULATE AMOUNT NEEDED TO ACCOMODATE
C                                  THIS ALLOCATION REQUEST
            LNEED1 = (NELMTS-ILEFT)*ISIZE(-ITYPE)
            IF (ILEFT .EQ. 0) THEN
               IGAP = ISIZE(-ITYPE) - MOD(LNOW+LNEED,ISIZE(-ITYPE))
               IF (IGAP .EQ. ISIZE(-ITYPE)) IGAP = 0
               LNEED1 = LNEED1 + 2*ISIZE(2) + IGAP
            END IF
C                                  MODIFY LNEED ACCORDING TO THE SIZE
C                                  OF THE BASE BEING USED (D.P. HERE)
            LNEED = LNEED + ((LNEED1+ISIZE(3)-1)/ISIZE(3))
C                                  SINCE CURRENT ALLOCATION IS ILLEGAL,
C                                  RETURN THE NEGATIVE OF THE ADDITIONAL
C                                  AMOUNT NEEDED TO MAKE IT LEGAL
            I1KGT = -LNEED
         END IF
      END IF
C                                  STACK OVERFLOW - UNRECOVERABLE ERROR
 9000 IF (LNEED .GT. 0) THEN
         CALL E1POS (-5, IPA, ISA)
         CALL E1POS (5, 0, 0)
         CALL E1POS (-7, IPA7, ISA7)
         CALL E1POS (7, 0, 0)
         CALL E1PSH ('I1KGT ')
         CALL E1STI (1, LNEED+(LMAX/ISIZE(3)))
         IF (XXLINE(PROLVL).GE.1 .AND. XXLINE(PROLVL).LE.999) THEN
            CALL E1MES (7, 1, 'Insufficient workspace for current '//
     &                  'allocation(s).  Correct by inserting the '//
     &                  'following PROTRAN line: $OPTIONS;WORKSPACE=%'//
     &                  '(I1)')
         ELSE
            CALL E1MES (5, 5, 'Insufficient workspace for current '//
     &                  'allocation(s). Correct by calling IWKIN '//
     &                  'from main program with the three following '//
     &                  'statements:  (REGARDLESS OF PRECISION)%/'//
     &                  '      COMMON /WORKSP/  RWKSP%/      REAL '//
     &                  'RWKSP(%(I1))%/      CALL IWKIN(%(I1))')
         END IF
         CALL E1POP ('I1KGT ')
         CALL E1POS (5, IPA, ISA)
         CALL E1POS (7, IPA7, ISA7)
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  IWKIN (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    January 17, 1984
C
C  Purpose:    Initialize bookkeeping locations describing the
C              workspace stack.
C
C  Usage:      CALL IWKIN (NSU)
C
C  Argument:
C     NSU    - Number of numeric storage units to which the workspace
C              stack is to be initialized
C
C  GAMS:       N4
C
C  Chapters:   MATH/LIBRARY Reference Material
C              STAT/LIBRARY Reference Material
C
C  Copyright:  1984 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE IWKIN (NSU)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    NSU
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    ISIZE(6), LALC, LBND, LBOOK, LMAX, LNEED, LNOW, LOUT,
     &           LUSED, MELMTS, MTYPE
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      LOGICAL    FIRST
      SAVE       FIRST
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
C                                  SPECIFICATIONS FOR EQUIVALENCE
      EQUIVALENCE (LOUT, IWKSP(1))
      EQUIVALENCE (LNOW, IWKSP(2))
      EQUIVALENCE (LUSED, IWKSP(3))
      EQUIVALENCE (LBND, IWKSP(4))
      EQUIVALENCE (LMAX, IWKSP(5))
      EQUIVALENCE (LALC, IWKSP(6))
      EQUIVALENCE (LNEED, IWKSP(7))
      EQUIVALENCE (LBOOK, IWKSP(8))
      EQUIVALENCE (ISIZE(1), IWKSP(11))
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  MAX0
      INTRINSIC  MAX0
      INTEGER    MAX0
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1STI
C
      DATA FIRST/.TRUE./
C
      IF (.NOT.FIRST) THEN
         IF (NSU .NE. 0) THEN
            CALL E1STI (1, LMAX)
            CALL E1MES (5, 100, 'Error from subroutine IWKIN:  '//
     &                  'Workspace stack has previously been '//
     &                  'initialized to %(I1). Correct by making the '//
     &                  'call to IWKIN the first executable '//
     &                  'statement in the main program.  ')
C
            STOP
C
         ELSE
            RETURN
         END IF
      END IF
C
      IF (NSU .EQ. 0) THEN
C                                  IF NSU=0 USE DEFAULT SIZE 5000
         MELMTS = 5000
      ELSE
         MELMTS = NSU
      END IF
C                                  NUMBER OF ITEMS .LT. 0
      IF (MELMTS .LE. 0) THEN
         CALL E1STI (1, MELMTS)
         CALL E1MES (5, 1, 'Error from subroutine IWKIN:  Number '//
     &               'of numeric storage units is not positive. NSU '//
     &               '= %(I1) ')
      ELSE
C
         FIRST = .FALSE.
C                                  HERE TO INITIALIZE
C
C                                  SET DATA SIZES APPROPRIATE FOR A
C                                  STANDARD CONFORMING FORTRAN SYSTEM
C                                  USING THE FORTRAN
C                                  *NUMERIC STORAGE UNIT* AS THE
C                                  MEASURE OF SIZE.
C
C                                  TYPE IS REAL
         MTYPE = 3
C                                  LOGICAL
         ISIZE(1) = 1
C                                  INTEGER
         ISIZE(2) = 1
C                                  REAL
         ISIZE(3) = 1
C                                  DOUBLE PRECISION
         ISIZE(4) = 2
C                                  COMPLEX
         ISIZE(5) = 2
C                                  DOUBLE COMPLEX
         ISIZE(6) = 4
C                                  NUMBER OF WORDS USED FOR BOOKKEEPING
         LBOOK = 16
C                                  CURRENT ACTIVE LENGTH OF THE STACK
         LNOW = LBOOK
C                                  MAXIMUM VALUE OF LNOW ACHIEVED THUS
C                                  FAR
         LUSED = LBOOK
C                                  MAXIMUM LENGTH OF THE STORAGE ARRAY
         LMAX = MAX0(MELMTS,((LBOOK+2)*ISIZE(2)+ISIZE(3)-1)/ISIZE(3))
C                                  LOWER BOUND OF THE PERMANENT STORAGE
C                                  WHICH IS ONE WORD MORE THAN THE
C                                  MAXIMUM ALLOWED LENGTH OF THE STACK
         LBND = LMAX + 1
C                                  NUMBER OF CURRENT ALLOCATIONS
         LOUT = 0
C                                  TOTAL NUMBER OF ALLOCATIONS MADE
         LALC = 0
C                                  NUMBER OF WORDS BY WHICH THE ARRAY
C                                  SIZE MUST BE INCREASED FOR ALL PAST
C                                  ALLOCATIONS TO SUCCEED
         LNEED = 0
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  B5ONF/DB5ONF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Compute the BFGS update, unfactored form.
C
C  Usage:      CALL B5ONF (N, SC, GC, GP, EPSFCN, USRDER, A, LDA, HDIAG,
C                          Y, T)
C
C  Arguments:
C     N      - The size fo the problem.  (Input)
C     SC     - Real vector of length N containing the current step.
C                 (Input)
C     GC     - Real vector of length N containing the gradient at XC.
C                 (Input)
C     GP     - Real vector of length N containing the gradient at XP.
C                 (Input)
C     EPSFCN - An estimated bound on the relative noise in the function
C              value.  (Input)
C     USRDER - Logical variable.  (Input)
C              USRDER = .TRUE.  means that analytic gradient is used,
C              USRDER = .FALSE. means that finite differences gradient
C                               is used.
C     A      - Real N by N symmetric matrix.  (Input/Output)
C              On input the current approximate Hessian is contained in
C                 the strict upper triangular part and HDIAG.
C              On output the updated approximated Hessian is contained
C                 in the lower triangular part and diagonal of A.
C     LDA    - Leading dimension of A exactly as specified in the
C              dimension statement of the calling program.  (Input)
C     HDIAG  - Real vector of length N containing the diagonal of the
C              current approximate Hessian.  (Input)
C     Y      - Real work vector of length N.
C     T      - Real work vector of length N.
C
C  Remark:
C     This is based on Algorithm A9.4.1, page 355 Dennis-Schnabel book
C     with a slightly modified storage scheme.
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
      SUBROUTINE B5ONF (N, SC, GC, GP, EPSFCN, USRDER, A, LDA, HDIAG,
     &                  Y, T)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, LDA
      REAL       EPSFCN, SC(*), GC(*), GP(*), A(LDA,*), HDIAG(*),
     &           Y(*), T(*)
      LOGICAL    USRDER
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, J
      REAL       EPS, GMAX, SNORM, TEMP1, TEMP2, TOL, YNORM
      LOGICAL    SKPUPD
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS,AMAX1,SQRT
      INTRINSIC  ABS, AMAX1, SQRT
      REAL       ABS, AMAX1, SQRT
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   SCOPY, CSFRG
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   AMACH, SDOT, SNRM2
      REAL       AMACH, SDOT, SNRM2
C
      EPS = AMACH(4)
C                                  COPY HESSIAN IN UPPER TRIANGULAR
C                                  PART AND HDIAG TO LOWER TRIANGULAR
C                                  PART AND DIAGONAL.
      CALL SCOPY (N, HDIAG, 1, A(1,1), LDA+1)
      CALL CSFRG (N, A, LDA)
C                                  COMPUTE SECANT CONDITION
      DO 10  I=1, N
         Y(I) = GP(I) - GC(I)
   10 CONTINUE
      TEMP1 = SDOT(N,Y,1,SC,1)
      SNORM = SNRM2(N,SC,1)
      YNORM = SNRM2(N,Y,1)
C                                  DETERMINE TO SKIP UPDATE OR NOT
      IF (TEMP1 .GE. (SQRT(EPS)*SNORM*YNORM)) THEN
         IF (USRDER) THEN
            TOL = EPSFCN
         ELSE
            TOL = SQRT(EPSFCN)
         END IF
         SKPUPD = .TRUE.
C                                  COMPUTE T(i) = H * SC(i) AND CHECK
C                                  UPDATE CONDITION ON ROW I
         DO 20  I=1, N
            T(I) = SDOT(N,A(I,1),LDA,SC,1)
            GMAX = AMAX1(ABS(GC(I)),ABS(GP(I)))
            IF (ABS(Y(I)-T(I)) .GE. (TOL*GMAX)) SKPUPD = .FALSE.
   20    CONTINUE
C
         IF (.NOT.SKPUPD) THEN
            TEMP2 = SDOT(N,SC,1,T,1)
            DO 40  J=1, N
               DO 30  I=J, N
                  A(I,J) = A(I,J) + (Y(I)*Y(J))/TEMP1 -
     &                     (T(I)*T(J))/TEMP2
   30          CONTINUE
   40       CONTINUE
         END IF
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  B4ONF/DB4ONF  (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Find a next iterate by a line search for the minimization
C              problem.
C
C  Usage:      CALL B4ONF (FCN, N, XC, FC, GC, SN, XSCALE, STEPMX,
C                          STEPTL, ICODE, XP, FP, SC, MXTAKE, NFCN)
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
C     XC     - Real vector of length N containing the current iterate.
C                 (Input)
C     FC     - Real scalar containing the function value at XC.  (Input)
C     GC     - Real vector of length N containing the gradient at XC.
C                 (Input)
C     SN     - Real vector of length N containing a descent direction
C              which is the Newton's direction for most cases.  (Input)
C     XSCALE - Real vector of length N containing the diagonal scaling
C              matrix for the variables.  (Input)
C     STEPMX - Real scalar containing the maximum allowable step size.
C                 (Input)
C     STEPTL - Real scalar containing the relative step size at which
C              successive iterates are considered close enough to stop
C              the algorithm.  (Input)
C     ICODE  - Integer return code.  (Output)
C              ICODE = 0 means a satisfactory new iterate is found.
C              ICODE = 1 means the routine failed to locate a new point
C                        sufficiently different from the current point.
C     XP     - Real vector of length N containing the new iterate.
C                 (Output)
C     FP     - Real scalar containing the function value at XP. (Output)
C     SC     - Real vector of length N containing the step taken.
C                 (Output)
C     MXTAKE - Logical variable.  (Output)
C              MXTAKE = .TRUE. if a step of maximum length was taken.
C              MXTAKE = .FALSE. otherwise.
C     NFCN   - Number of function evaluations.  (Input/Output)
C
C  Remark:
C     This is based on Algorithm A6.3.1, page 325, Dennis-Schnabel book.
C     It performs a bactracking linesearch with an alpha-condition.
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
      SUBROUTINE B4ONF (FCN, N, XC, FC, GC, XLB, XUB, SN, XSCALE,
     &                  STEPMX, STEPTL, IACT, ICODE, XP, FP, SC,
     &                  MXTAKE, RNWTNL, EPSFCN, NFCN)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, ICODE, NFCN, IACT(*)
      REAL       FC, STEPMX, STEPTL, FP, RNWTNL, EPSFCN, XC(*), GC(*),
     &           XLB(*), XUB(*), SN(*), XSCALE(*), XP(*), SC(*)
      LOGICAL    MXTAKE
      EXTERNAL   FCN
C                                  SPECIFICATIONS FOR PARAMETERS
      REAL       ALPHA
      PARAMETER  (ALPHA=1.0E-4)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I
      REAL       A, ALAMDA, AMINLA, B, BIG, DISC, FACTOR, PFP, PLAMDA,
     &           RELLEN, SLOPE, SMALL, T1, T2, T3, T4, T5, TEMP, TEMPL
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS,AMAX1,AMIN1,SQRT
      INTRINSIC  ABS, AMAX1, AMIN1, SQRT
      REAL       ABS, AMAX1, AMIN1, SQRT
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1USR, SAXPY, SCOPY, SSCAL, SVCAL
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   AMACH, SDOT
      REAL       AMACH, SDOT
C
      SMALL = AMACH(1)
      BIG = AMACH(2)
      IF (BIG*SMALL .LT. 1.0E0) SMALL = 1.0E0/BIG
      MXTAKE = .FALSE.
      ICODE = 2
C                                  INITIAL STEP TO TRY IS NEWTON STEP
      IF (RNWTNL .GT. STEPMX) THEN
         FACTOR = STEPMX/RNWTNL
         CALL SVCAL (N, FACTOR, SN, 1, SC, 1)
         RNWTNL = STEPMX
      ELSE
         CALL SCOPY (N, SN, 1, SC, 1)
      END IF
      FACTOR = 1.0
      DO 10  I=1, N
         IF (IACT(I) .EQ. 0) THEN
            T1 = 1.0
            IF (SC(I) .GT. 0.0) THEN
               T1 = (XUB(I)-XC(I))/SC(I)
            ELSE IF (SC(I) .LT. 0.0) THEN
               T1 = (XLB(I)-XC(I))/SC(I)
            END IF
            FACTOR = AMIN1(FACTOR,T1)
         END IF
   10 CONTINUE
      IF (FACTOR .NE. 1.0) THEN
         CALL SSCAL (N, FACTOR, SC, 1)
         RNWTNL = FACTOR*RNWTNL
      END IF
C                                  RELLEN = RELATIVE LENGTH OF SC AS
C                                  CALCULATED IN STOPPING RULE
      SLOPE = SDOT(N,GC,1,SC,1)
      RELLEN = 0.0E0
      DO 20  I=1, N
         IF (IACT(I) .EQ. 0) THEN
            TEMP = ABS(SC(I))/(AMAX1(ABS(XC(I)),1.0E0/XSCALE(I)))
            RELLEN = AMAX1(RELLEN,TEMP)
         END IF
   20 CONTINUE
      AMINLA = STEPTL/RELLEN
      ALAMDA = 1.0E0
C                                  LOOP TO CHECK IF XP IS O.K. AND
C                                  GENERATE NEXT LAMBDA IF REQUIRED
   30 CONTINUE
      CALL SCOPY (N, XC, 1, XP, 1)
      CALL SAXPY (N, ALAMDA, SC, 1, XP, 1)
      CALL E1USR ('ON')
      CALL FCN (N, XP, FP)
      CALL E1USR ('OFF')
      NFCN = NFCN + 1
      IF (FP .LE. (FC+ALPHA*ALAMDA*SLOPE)) THEN
C                                  XP SATISFIES ALPHA CONDITION
         ICODE = 0
         IF (ALAMDA*RNWTNL .GT. 0.99E0*STEPMX) MXTAKE = .TRUE.
      ELSE IF (ALAMDA .LT. AMINLA) THEN
C                                  NO SATISFACTORY XP CAN BE FOUND
C                                  SUFFICIENTLY DISTINCT FROM XC
         ICODE = 1
         CALL SCOPY (N, XC, 1, XP, 1)
      ELSE
C                                  REDUCE THE STEP SIZE LAMBDA
         IF (ALAMDA .EQ. 1.0E0) THEN
            TEMPL = -SLOPE/(2.0E0*(FP-FC-SLOPE))
         ELSE
            T1 = FP - FC - ALAMDA*SLOPE
            T2 = PFP - FC - PLAMDA*SLOPE
            T3 = 1.0E0/(ALAMDA-PLAMDA)
            T4 = ALAMDA*ALAMDA
            T5 = PLAMDA*PLAMDA
            A = T3*(T1/T4-T2/T5)
            B = T3*(-T1*PLAMDA/T4+T2*ALAMDA/T5)
C                                  DISC WILL BE NONNEGATIVE AS LONG AS
C                                  ALPHA IS .LT. 0.25
            DISC = B*B - 3.0E0*A*SLOPE
            IF (ABS(A) .GT. SMALL) THEN
               TEMPL = -SLOPE/(2.0E0*B)
            ELSE
               TEMPL = (-B+SQRT(DISC))/(3.0E0*A)
            END IF
            IF (TEMPL .GT. 0.5E0*ALAMDA) TEMPL = 0.5E0*ALAMDA
         END IF
         PLAMDA = ALAMDA
         PFP = FP
         IF (TEMPL .LE. 0.1E0*ALAMDA) THEN
            ALAMDA = 0.1E0*ALAMDA
         ELSE
            ALAMDA = TEMPL
         END IF
      END IF
      IF (ICODE .GE. 2) GO TO 30
C                                  OUTPUT THE STEP TAKEN
      CALL SSCAL (N, ALAMDA, SC, 1)
C
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
      SUBROUTINE B3ONF (FCN, N, XC, XLB, XUB, XSCALE, FSCALE, IPARAM,
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
  110 CALL U6INF (N, XP, SC, FP, WK1, XSCALE, FSCALE, ICODE, ITER,
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
C  IMSL Name:  FDGRD/DFDGRD  (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Approximate the gradient using forward differences.
C
C  Usage:      CALL FDGRD (FCN, N, XC, XSCALE, FC, EPSFCN, GC)
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
C     XC     - Vector of length N containing the point at which the
C              gradient is to be estimated.  (Input)
C     XSCALE - Vector of length N containing the diagonal scaling matrix
C              for the variables.  (Input)
C              In the absence of other information, set all entries
C              to 1.0.
C     FC     - Scalar containing the value of the function at XC.
C              (Input)
C     EPSFCN - Estimate of the relative noise in the function.  (Input)
C              EPSFCN must be less than or equal to 0.1.  In the absence
C              of other information, set EPSFCN to 0.0.
C     GC     - Vector of length N containing the estimated gradient
C              at XC.  (Output)
C
C  Remark:
C     This is Algorithm A5.6.3, Dennis and Schnabel, 1983, page 322.
C
C  Keywords:   Forward difference; Gradient
C
C  GAMS:       G4f
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
      SUBROUTINE FDGRD (FCN, N, XC, XSCALE, FC, EPSFCN, GC)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N
      REAL       FC, EPSFCN, XC(*), XSCALE(*), GC(*)
      EXTERNAL   FCN
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, J
      REAL       EPS, FNEW, STEPSZ, XTEMPJ
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS,AMAX1,SQRT
      INTRINSIC  ABS, AMAX1, SQRT
      REAL       ABS, AMAX1, SQRT
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, E1STI, E1STR
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   AMACH, N1RCD
      INTEGER    N1RCD
      REAL       AMACH
C
      CALL E1PSH ('FDGRD ')
C
      IF (N .LE. 0) THEN
         CALL E1STI (1, N)
         CALL E1MES (5, 1, 'The number of variables must be '//
     &               'positive while N = %(I1) is given.')
      ELSE IF (EPSFCN.GT.0.1E0 .OR. EPSFCN.LT.0.0E0) THEN
         CALL E1STR (1, EPSFCN)
         CALL E1MES (5, 2, 'The estimate for the relative '//
     &               'noise in the function must be between 0.0 '//
     &               'and 0.1 while EPSFCN = %(R1) is given.')
      ELSE
         DO 10  I=1, N
            IF (XSCALE(I) .LE. 0.0E0) THEN
               CALL E1STI (1, I)
               CALL E1STR (1, XSCALE(I))
               CALL E1MES (5, 3, 'The values for the diagonal '//
     &                     'scaling matrix must be positive while '//
     &                     'XSCALE(%(I1)) = %(R1) is given.')
               GO TO 9000
            END IF
   10    CONTINUE
      END IF
C
      IF (N1RCD(0) .EQ. 0) THEN
         EPS = SQRT(AMAX1(EPSFCN,AMACH(4)))
         DO 20  J=1, N
            STEPSZ = (EPS)*AMAX1(ABS(XC(J)),1.0E0/XSCALE(J))
            IF (XC(J) .LT. 0.0) STEPSZ = -STEPSZ
            XTEMPJ = XC(J)
            XC(J) = XTEMPJ + STEPSZ
            CALL FCN (N, XC, FNEW)
            XC(J) = XTEMPJ
            GC(J) = (FNEW-FC)/STEPSZ
   20    CONTINUE
      END IF
C
 9000 CALL E1POP ('FDGRD ')
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  SNRM2 (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1986
C
C  Purpose:    Compute the Euclidean length or L2 norm of a
C              single-precision vector.
C
C  Usage:      SNRM2(N, SX, INCX)
C
C  Arguments:
C     N      - Length of vector X.  (Input)
C     SX     - Real vector of length N*INCX.  (Input)
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be SX(1+(I-1)*INCX). INCX must be
C              greater than zero.
C     SNRM2  - Square root of the sum from I=1 to N of X(I)**2.
C              (Output)
C              X(I) refers to a specific element of SX. See INCX
C              argument description.
C
C  GAMS:       D1a3b
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      REAL FUNCTION SNRM2 (N, SX, INCX)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX
      REAL       SX(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, J, NEXT, NN
      REAL       HITEST, SUM, XMAX
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      REAL       CUTHI, CUTLO, ONE, ZERO
      SAVE       CUTHI, CUTLO, ONE, ZERO
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS,SQRT
      INTRINSIC  ABS, SQRT
      REAL       ABS, SQRT
C
      DATA ZERO/0.0E0/, ONE/1.0E0/
      DATA CUTLO/4.441E-16/, CUTHI/1.304E19/
C
      IF (N .GT. 0) GO TO 10
      SNRM2 = ZERO
      GO TO 140
C
   10 ASSIGN 30 TO NEXT
      SUM = ZERO
      NN = N*INCX
C                                  BEGIN MAIN LOOP
      I = 1
   20 GO TO NEXT, (30, 40, 70, 80)
   30 IF (ABS(SX(I)) .GT. CUTLO) GO TO 110
      ASSIGN 40 TO NEXT
      XMAX = ZERO
C                                  PHASE 1. SUM IS ZERO
   40 IF (SX(I) .EQ. ZERO) GO TO 130
      IF (ABS(SX(I)) .GT. CUTLO) GO TO 110
C                                  PREPARE FOR PHASE 2.
      ASSIGN 70 TO NEXT
      GO TO 60
C                                  PREPARE FOR PHASE 4.
   50 I = J
      ASSIGN 80 TO NEXT
      SUM = (SUM/SX(I))/SX(I)
   60 XMAX = ABS(SX(I))
      GO TO 90
C                                  PHASE 2. SUM IS SMALL. SCALE TO
C                                  AVOID DESTRUCTIVE UNDERFLOW.
   70 IF (ABS(SX(I)) .GT. CUTLO) GO TO 100
C                                  COMMON CODE FOR PHASES 2 AND 4. IN
C                                  PHASE 4 SUM IS LARGE. SCALE TO
C                                  AVOID OVERFLOW.
   80 IF (ABS(SX(I)) .LE. XMAX) GO TO 90
      SUM = ONE + SUM*(XMAX/SX(I))**2
      XMAX = ABS(SX(I))
      GO TO 130
C
   90 SUM = SUM + (SX(I)/XMAX)**2
      GO TO 130
C                                  PREPARE FOR PHASE 3.
  100 SUM = (SUM*XMAX)*XMAX
C                                  FOR REAL OR D.P. SET HITEST =
C                                  CUTHI/N FOR COMPLEX SET HITEST =
C                                  CUTHI/(2*N)
  110 HITEST = CUTHI/N
C                                  PHASE 3. SUM IS MID-RANGE. NO
C                                  SCALING.
      DO 120  J=I, NN, INCX
         IF (ABS(SX(J)) .GE. HITEST) GO TO 50
  120 SUM = SUM + SX(J)*SX(J)
      SNRM2 = SQRT(SUM)
      GO TO 140
C
  130 CONTINUE
      I = I + INCX
      IF (I .LE. NN) GO TO 20
C                                  END OF MAIN LOOP. COMPUTE SQUARE
C                                  ROOT AND ADJUST FOR SCALING.
      SNRM2 = XMAX*SQRT(SUM)
  140 CONTINUE
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
      SUBROUTINE B2ONF (FCN, N, XGUESS, IBTYPE, XLB, XUB, XSCALE,
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
         IF (N1RCD(0) .EQ. 0) CALL B3ONF (FCN, N, X, XLB, XUB, XSCALE,
     &       FSCALE, IPARAM, RPARAM, IWK, FVALUE, WK(1), WK(N+1),
     &       WK(2*N+1), WK(3*N+1), WK(4*N+1), WK((N+8)*N+1), N,
     &       WK(5*N+1), WK(6*N+1), WK(7*N+1), WK(8*N+1))
      END IF
C
      CALL E1POP ('B2ONF ')
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U18NF/DU18NF  (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Stopping conditions for unconstrained minimization.
C
C  Usage:      CALL U18NF (ICODE)
C
C  Arguments:
C     ICODE  - Integer flag containing an error code.  (Input)
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
      SUBROUTINE U18NF (ICODE)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    ICODE
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES
C
      IF (ICODE .EQ. 3) THEN
         CALL E1MES (4, 3, ' Maximum number of iterations '//
     &               'exceeded.')
      ELSE IF (ICODE .EQ. 4) THEN
         CALL E1MES (4, 4, ' Maximum number of function evaluations '//
     &               'exceeded.')
      ELSE IF (ICODE .EQ. 5) THEN
         CALL E1MES (4, 5, ' Maximum number of gradient evaluations '//
     &               'exceeded.')
      ELSE IF (ICODE .EQ. 7) THEN
         CALL E1MES (4, 7, 'Maximum number of Hessian evaluations '//
     &               'exceeded.')
      ELSE IF (ICODE .EQ. 6) THEN
         CALL E1MES (4, 6, ' Five consecutive steps of length STEPMX '//
     &               'have been taken; either the function is '//
     &               'unbounded below, or has a finite asymptote in '//
     &               'some direction or the maximum allowable step '//
     &               'size STEPMX is too small.')
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  TRNRR/DTRNRR (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 8, 1985
C
C  Purpose:    Transpose a rectangular matrix.
C
C  Usage:      CALL TRNRR (NRA, NCA, A, LDA, NRB, NCB, B, LDB)
C
C  Arguments:
C     NRA    - Number of rows of A.  (Input)
C     NCA    - Number of columns of A.  (Input)
C     A      - Real NRA by NCA matrix in full storage mode.  (Input)
C     LDA    - Leading dimension of A exactly as specified in the
C              dimension statement of the calling program.  (Input)
C     NRB    - Number of rows of B.  (Input)
C              NRB must be equal to NCA.
C     NCB    - Number of columns of B.  (Input)
C              NCB must be equal to NRA.
C     B      - Real NRB by NCB matrix in full storage mode containing
C              the transpose of A.  (Output)
C     LDB    - Leading dimension of B exactly as specified in the
C              dimension statement of the calling program.  (Input)
C
C  Remark:
C     If LDA=LDB and NRA=NCA then A and B may be the same matrix,
C     otherwise A and B must be different matrices.
C
C  Keyword:    Basic matrix operation
C
C  GAMS:       D1b3
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE TRNRR (NRA, NCA, A, LDA, NRB, NCB, B, LDB)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    NRA, NCA, LDA, NRB, NCB, LDB
      REAL       A(LDA,*), B(LDB,*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, E1STI, SCOPY, SSWAP
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   N1RTY
      INTEGER    N1RTY
C
      CALL E1PSH ('TRNRR ')
C
      IF (NRA.LE.0 .OR. NCA.LE.0) THEN
         CALL E1STI (1, NRA)
         CALL E1STI (2, NCA)
         CALL E1MES (5, 1, 'Both the number of rows and the '//
     &               'number of columns of a matrix have '//
     &               'to be positive while NRA = %(I1) and '//
     &               'NCA = %(I2) are given.')
      END IF
C
      IF (NRA .GT. LDA) THEN
         CALL E1STI (1, NRA)
         CALL E1STI (2, LDA)
         CALL E1MES (5, 2, 'The number of rows of the matrix '//
     &               'must be less than or equal to its leading '//
     &               'dimension while NRA = %(I1) and LDA = %(I2) '//
     &               'are given.')
      END IF
C
      IF (NRB.LE.0 .OR. NCB.LE.0) THEN
         CALL E1STI (1, NRB)
         CALL E1STI (2, NCB)
         CALL E1MES (5, 3, 'Both the number of rows and the '//
     &               'number of columns of a matrix have '//
     &               'to be positive while NRB = %(I1) and '//
     &               'NCB = %(I2) are given.')
      END IF
C
      IF (NRB .GT. LDB) THEN
         CALL E1STI (1, NRB)
         CALL E1STI (2, LDB)
         CALL E1MES (5, 4, 'The number of rows of the matrix '//
     &               'must be less than or equal to its leading '//
     &               'dimension while NRB = %(I1) and LDB = %(I2) '//
     &               'are given.')
      END IF
C
      IF (N1RTY(0) .NE. 0) GO TO 9000
C
      IF (NRB.NE.NCA .OR. NCB.NE.NRA) THEN
         CALL E1STI (1, NRA)
         CALL E1STI (2, NCA)
         CALL E1STI (3, NRB)
         CALL E1STI (4, NCB)
         CALL E1MES (5, 5, 'The following must hold NRB = '//
     &               'NCA and NCB = NRA while NRB = %(I3), '//
     &               'NCA = %(I2), NCB = %(I4) and NRA = %(I1).')
      END IF
C
      IF (N1RTY(0) .NE. 0) GO TO 9000
C                                  If LDA=LDB and NRA=NCA then A and B
C                                    may be the same, copy A to B and
C                                    transpose B with SSWAP.
      IF (NRA.EQ.NCA .AND. LDA.EQ.LDB) THEN
         DO 10  I=1, NCA
            CALL SCOPY (NRA, A(1,I), 1, B(1,I), 1)
   10    CONTINUE
         DO 20  I=1, NCA - 1
            CALL SSWAP (NRA-I, B(I+1,I), 1, B(I,I+1), LDB)
   20    CONTINUE
      ELSE
         DO 30  I=1, NCA
            CALL SCOPY (NRA, A(1,I), 1, B(I,1), LDB)
   30    CONTINUE
      END IF
 9000 CALL E1POP ('TRNRR ')
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1PSH
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 2, 1984
C
C  Purpose:    To push a subroutine name onto the error control stack.
C
C  Usage:      CALL E1PSH(NAME)
C
C  Arguments:
C     NAME   - A character string of length six specifing the name of
C              the subroutine.  (Input)
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1PSH (NAME)
C                                  SPECIFICATIONS FOR ARGUMENTS
      CHARACTER  NAME*(*)
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    IFINIT
      SAVE       IFINIT
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1INIT, E1MES, E1STI
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   I1KST
      INTEGER    I1KST
C
      DATA IFINIT/0/
C                                  INITIALIZE ERROR TABLE IF NECESSARY
      IF (IFINIT .EQ. 0) THEN
         CALL E1INIT
         IFINIT = 1
      END IF
      IF (CALLVL .GE. MAXLEV) THEN
         CALL E1STI (1, MAXLEV)
         CALL E1MES (5, 1, 'Error condition in E1PSH.  Push would '//
     &               'cause stack level to exceed %(I1). ')
         STOP
      ELSE
C                                  STORE ALLOCATION LEVEL
         IALLOC(CALLVL) = I1KST(1)
C                                  INCREMENT THE STACK POINTER BY ONE
         CALLVL = CALLVL + 1
C                                  PUT SUBROUTINE NAME INTO STACK
         RNAME(CALLVL) = NAME
C                                  SET ERROR TYPE AND ERROR CODE
         ERTYPE(CALLVL) = 0
         ERCODE(CALLVL) = 0
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  N1RCD
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 6, 1984
C
C  Purpose:    Retrieve an error code.
C
C  Usage:      N1RCD(IOPT)
C
C  Arguments:
C     IOPT   - Integer specifying the level.  (Input)
C              If IOPT=0 the error code for the current level is
C              returned.  If IOPT=1 the error code for the most
C              recently called routine (last pop) is returned.
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION N1RCD (IOPT)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    IOPT
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1PRT, M1VECH
C
      IF (IOPT.NE.0 .AND. IOPT.NE.1) THEN
         ERTYPE(CALLVL) = 5
         ERCODE(CALLVL) = 1
         MSGLEN = 47
         CALL M1VECH ('.  The argument passed to N1RCD must be 0 or '//
     &                '1. ', MSGLEN, MSGSAV, MSGLEN)
         CALL E1PRT
         STOP
      ELSE
         N1RCD = ERCODE(CALLVL+IOPT)
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  SSET (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1986
C
C  Purpose:    Set the components of a vector to a scalar, all
C              single precision.
C
C  Usage:      CALL SSET (N, SA, SX, INCX)
C
C  Arguments:
C     N      - Length of vector X.  (Input)
C     SA     - Real scalar.  (Input)
C     SX     - Real vector of length N*INCX.  (Input/Output)
C              SSET replaces X(I) with SA for I=1,...,N. X(I) refers to
C              a specific element of SX. See INCX argument description.
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be SX(1+(I-1)*INCX). INCX must be
C              greater than zero.
C
C  GAMS:       D1a1
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE SSET (N, SA, SX, INCX)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX
      REAL       SA, SX(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, M, MP1, NINCX
C                                  SPECIFICATIONS FOR SPECIAL CASES
C     INTRINSIC  MOD
      INTRINSIC  MOD
      INTEGER    MOD
C
      IF (N .GT. 0) THEN
         IF (INCX .NE. 1) THEN
C                                  CODE FOR INCREMENT NOT EQUAL TO 1
            NINCX = N*INCX
            DO 10  I=1, NINCX, INCX
               SX(I) = SA
   10       CONTINUE
         ELSE
C                                  CODE FOR INCREMENT EQUAL TO 1
            M = MOD(N,8)
C                                  CLEAN-UP LOOP
            DO 30  I=1, M
               SX(I) = SA
   30       CONTINUE
            MP1 = M + 1
            DO 40  I=MP1, N, 8
               SX(I) = SA
               SX(I+1) = SA
               SX(I+2) = SA
               SX(I+3) = SA
               SX(I+4) = SA
               SX(I+5) = SA
               SX(I+6) = SA
               SX(I+7) = SA
   40       CONTINUE
         END IF
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1POP
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 13, 1984
C
C  Purpose:    To pop a subroutine name from the error control stack.
C
C  Usage:      CALL E1POP(NAME)
C
C  Arguments:
C     NAME   - A character string of length six specifying the name
C              of the subroutine.  (Input)
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1POP (NAME)
C                                  SPECIFICATIONS FOR ARGUMENTS
      CHARACTER  NAME*(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    IERTYP, IR
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1PRT, E1PSH, E1STI, E1STL, I1KRL
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   I1KST
      INTEGER    I1KST
C
      IF (CALLVL .LE. 1) THEN
         CALL E1PSH ('E1POP ')
         CALL E1STL (1, NAME)
         CALL E1MES (5, 1, 'Error condition in E1POP.  Cannot pop '//
     &               'from %(L1) because stack is empty.')
         STOP
      ELSE IF (NAME .NE. RNAME(CALLVL)) THEN
         CALL E1STL (1, NAME)
         CALL E1STL (2, RNAME(CALLVL))
         CALL E1MES (5, 2, 'Error condition in E1POP.  %(L1) does '//
     &               'not match the name %(L2) in the stack.')
         STOP
      ELSE
         IERTYP = ERTYPE(CALLVL)
         IF (IERTYP .NE. 0) THEN
C                                  M1VE ERROR TYPE AND ERROR CODE TO
C                                    PREVIOUS LEVEL FOR ERROR TYPES 2-7
            IF (IERTYP.GE.2 .AND. IERTYP.LE.7) THEN
               ERTYPE(CALLVL-1) = ERTYPE(CALLVL)
               ERCODE(CALLVL-1) = ERCODE(CALLVL)
            END IF
C                                  CHECK PRINT TABLE TO DETERMINE
C                                    WHETHER TO PRINT STORED MESSAGE
            IF (IERTYP .LE. 4) THEN
               IF (ISUSER(CALLVL-1) .AND. PRINTB(IERTYP).EQ.1)
     &             CALL E1PRT
            ELSE
               IF (PRINTB(IERTYP) .EQ. 1) CALL E1PRT
            END IF
C                                  CHECK STOP TABLE AND ERROR TYPE TO
C                                    DETERMINE WHETHER TO STOP
            IF (IERTYP .LE. 4) THEN
               IF (ISUSER(CALLVL-1) .AND. STOPTB(IERTYP).EQ.1) THEN
                  STOP
               END IF
            ELSE IF (IERTYP .EQ. 5) THEN
               IF (STOPTB(IERTYP) .EQ. 1) THEN
                  STOP
               END IF
            ELSE IF (HDRFMT(IERTYP) .EQ. 1) THEN
               IF (ISUSER(CALLVL-1)) THEN
                  IF (N1RGB(0) .NE. 0) THEN
                     STOP
                  END IF
               END IF
            END IF
         END IF
C                                  SET ERROR TYPE AND CODE
         IF (CALLVL .LT. MAXLEV) THEN
            ERTYPE(CALLVL+1) = -1
            ERCODE(CALLVL+1) = -1
         END IF
C                                  SET IR = AMOUNT OF WORKSPACE
C                                  ALLOCATED AT THIS LEVEL
         IR = I1KST(1) - IALLOC(CALLVL-1)
         IF (IR .GT. 0) THEN
C                                  RELEASE WORKSPACE
            CALL I1KRL (IR)
            IALLOC(CALLVL) = 0
         ELSE IF (IR .LT. 0) THEN
            CALL E1STI (1, CALLVL)
            CALL E1STI (2, IALLOC(CALLVL-1))
            CALL E1STI (3, I1KST(1))
            CALL E1MES (5, 3, 'Error condition in E1POP. '//
     &                  ' The number of workspace allocations at '//
     &                  'level %(I1) is %(I2).  However, the total '//
     &                  'number of workspace allocations is %(I3).')
            STOP
         END IF
C                                  DECREASE THE STACK POINTER BY ONE
         CALLVL = CALLVL - 1
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  SCOPY (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1986
C
C  Purpose:    Copy a vector X to a vector Y, both single precision.
C
C  Usage:      CALL SCOPY (N, SX, INCX, SY, INCY)
C
C  Arguments:
C     N      - Length of vectors X and Y.  (Input)
C     SX     - Real vector of length MAX(N*IABS(INCX),1).  (Input)
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be.. SX(1+(I-1)*INCX) if INCX .GE. 0
C              or SX(1+(I-N)*INCX) if INCX .LT. 0.
C     SY     - Real vector of length MAX(N*IABS(INCY),1).  (Output)
C              SCOPY copies X(I) to Y(I) for I=1,...,N. X(I) and Y(I)
C              refer to specific elements of SX and SY, respectively.
C              See INCX and INCY argument descriptions.
C     INCY   - Displacement between elements of SY.  (Input)
C              Y(I) is defined to be.. SY(1+(I-1)*INCY) if INCY .GE. 0
C              or SY(1+(I-N)*INCY) if INCY .LT. 0.
C
C  GAMS:       D1a
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE SCOPY (N, SX, INCX, SY, INCY)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX, INCY
      REAL       SX(*), SY(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IX, IY, M, MP1
C                                  SPECIFICATIONS FOR SPECIAL CASES
C     INTRINSIC  MOD
      INTRINSIC  MOD
      INTEGER    MOD
C
      IF (N .GT. 0) THEN
         IF (INCX.NE.1 .OR. INCY.NE.1) THEN
C                                  CODE FOR UNEQUAL INCREMENTS
            IX = 1
            IY = 1
            IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
            IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
            DO 10  I=1, N
               SY(IY) = SX(IX)
               IX = IX + INCX
               IY = IY + INCY
   10       CONTINUE
         ELSE
C                                  CODE FOR BOTH INCREMENTS EQUAL TO 1
            M = MOD(N,7)
C                                  CLEAN-UP LOOP
            DO 30  I=1, M
               SY(I) = SX(I)
   30       CONTINUE
            MP1 = M + 1
            DO 40  I=MP1, N, 7
               SY(I) = SX(I)
               SY(I+1) = SX(I+1)
               SY(I+2) = SX(I+2)
               SY(I+3) = SX(I+3)
               SY(I+4) = SX(I+4)
               SY(I+5) = SX(I+5)
               SY(I+6) = SX(I+6)
   40       CONTINUE
         END IF
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  CDGRD/DCDGRD (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Approximate the gradient using central differences.
C
C  Usage:      CALL CDGRD (FCN, N, XC, XSCALE, EPSFCN, GC)
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
C     XC     - Vector of length N containing the point at which the
C              gradient is to be estimated.  (Input)
C     XSCALE - Vector of length N containing the diagonal scaling
C              matrix for the variables.  (Input)
C              In the absence of other information, set all entries
C              to 1.0.
C     EPSFCN - Estimate for the relative noise in the function.  (Input)
C              EPSFCN must be less than or equal to 0.1.  In the absence
C              of other information, set EPSFCN to 0.0.
C     GC     - Vector of length N containing the estimated gradient
C              at XC.  (Output)
C
C  Remark:
C     This is Algorithm A5.6.4, Dennis and Schnabel, 1983, page 323.
C
C  Keywords:   Central difference; Gradient
C
C  GAMS:       G4f
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
      SUBROUTINE CDGRD (FCN, N, XC, XSCALE, EPSFCN, GC)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N
      REAL       EPSFCN, XC(*), XSCALE(*), GC(*)
      EXTERNAL   FCN
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, J
      REAL       EPS, FNEW1, FNEW2, STEPSZ, XTEMPJ
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS,AMAX1,SQRT
      INTRINSIC  ABS, AMAX1, SQRT
      REAL       ABS, AMAX1, SQRT
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, E1STI, E1STR
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   AMACH, N1RCD
      INTEGER    N1RCD
      REAL       AMACH
C
      CALL E1PSH ('CDGRD ')
C
      IF (N .LE. 0) THEN
         CALL E1STI (1, N)
         CALL E1MES (5, 1, 'The number of variables must be '//
     &               'positive while N = %(I1) is given.')
      ELSE IF (EPSFCN.GT.0.1E0 .OR. EPSFCN.LT.0.0E0) THEN
         CALL E1STR (1, EPSFCN)
         CALL E1MES (5, 2, 'The estimate for the relative '//
     &               'noise in the function must be between 0.0 '//
     &               'and 0.1 while EPSFCN = %(R1) is given.')
      ELSE
         DO 10  I=1, N
            IF (XSCALE(I) .LE. 0.0E0) THEN
               CALL E1STI (1, I)
               CALL E1STR (1, XSCALE(I))
               CALL E1MES (5, 4, 'The values for the diagonal '//
     &                     'scaling matrix must be positive while '//
     &                     'XSCALE(%(I1)) = %(R1) is given.')
               GO TO 9000
            END IF
   10    CONTINUE
      END IF
C
      IF (N1RCD(0) .EQ. 0) THEN
         EPS = SQRT(AMAX1(EPSFCN,AMACH(4)))
         DO 20  J=1, N
            STEPSZ = (EPS)*AMAX1(ABS(XC(J)),1.0E0/XSCALE(J))
            IF (XC(J) .LT. 0.0) STEPSZ = -STEPSZ
            XTEMPJ = XC(J)
            XC(J) = XTEMPJ + STEPSZ
            CALL FCN (N, XC, FNEW1)
            XC(J) = XTEMPJ - STEPSZ
            CALL FCN (N, XC, FNEW2)
            XC(J) = XTEMPJ
            GC(J) = (FNEW1-FNEW2)/(2.0E0*STEPSZ)
   20    CONTINUE
      END IF
C
 9000 CALL E1POP ('CDGRD ')
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  AMACH (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 15, 1984
C
C  Purpose:    Retrieve single-precision machine constants.
C
C  Usage:      AMACH(N)
C
C  Arguments:
C     N      - Index of desired constant.  (Input)
C     AMACH  - Machine constant.  (Output)
C              AMACH(1) = B**(EMIN-1), the smallest positive magnitude.
C              AMACH(2) = B**EMAX*(1 - B**(-T)), the largest magnitude.
C              AMACH(3) = B**(-T), the smallest relative spacing.
C              AMACH(4) = B**(1-T), the largest relative spacing.
C              AMACH(5) = LOG10(B), the log, base 10, of the radix.
C              AMACH(6) = not-a-number.
C              AMACH(7) = positive machine infinity.
C              AMACH(8) = negative machine infinity.
C
C  GAMS:       R1
C
C  Chapters:   MATH/LIBRARY Reference Material
C              STAT/LIBRARY Reference Material
C              SFUN/LIBRARY Reference Material
C
C  Copyright:  1984 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      REAL FUNCTION AMACH (N)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      REAL       RMACH(8)
      SAVE       RMACH
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, E1STI
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    IRMACH(8)
C
      EQUIVALENCE (RMACH, IRMACH)
C                                  DEFINE CONSTANTS
      DATA RMACH(1)/1.17577E-38/
      DATA RMACH(2)/3.40204E+38/
      DATA RMACH(3)/5.96184E-8/
      DATA RMACH(4)/1.19237E-7/
      DATA RMACH(5)/.301029995663981195E0/
      DATA IRMACH(6)/2143289344/
      DATA IRMACH(7)/2139095040/
      DATA IRMACH(8)/-8388608/
C
      IF (N.LT.1 .OR. N.GT.8) THEN
         CALL E1PSH ('AMACH ')
         AMACH = RMACH(6)
         CALL E1STI (1, N)
         CALL E1MES (5, 5, 'The argument must be between 1 '//
     &               'and 8 inclusive. N = %(I1)')
         CALL E1POP ('AMACH ')
      ELSE
         AMACH = RMACH(N)
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  I1KQU
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    January 17, 1984
C
C  Purpose:    Return number of elements of data type ITYPE that
C              remain to be allocated in one request.
C
C  Usage:      I1KQU(ITYPE)
C
C  Arguments:
C     ITYPE  - Type of storage to be checked (Input)
C                 1 - logical
C                 2 - integer
C                 3 - real
C                 4 - double precision
C                 5 - complex
C                 6 - double complex
C     I1KQU  - Integer function. (Output) Returns number of elements
C              of data type ITYPE remaining in the stack.
C
C  Copyright:  1983 by IMSL, Inc.  All Rights Reserved
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION I1KQU (ITYPE)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    ITYPE
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    ISIZE(6), LALC, LBND, LBOOK, LMAX, LNEED, LNOW, LOUT,
     &           LUSED
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      LOGICAL    FIRST
      SAVE       FIRST
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
C                                  SPECIFICATIONS FOR EQUIVALENCE
      EQUIVALENCE (LOUT, IWKSP(1))
      EQUIVALENCE (LNOW, IWKSP(2))
      EQUIVALENCE (LUSED, IWKSP(3))
      EQUIVALENCE (LBND, IWKSP(4))
      EQUIVALENCE (LMAX, IWKSP(5))
      EQUIVALENCE (LALC, IWKSP(6))
      EQUIVALENCE (LNEED, IWKSP(7))
      EQUIVALENCE (LBOOK, IWKSP(8))
      EQUIVALENCE (ISIZE(1), IWKSP(11))
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  MAX0
      INTRINSIC  MAX0
      INTEGER    MAX0
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, IWKIN
C
      DATA FIRST/.TRUE./
C
      CALL E1PSH ('I1KQU ')
C
      IF (FIRST) THEN
C                                  INITIALIZE WORKSPACE IF NEEDED
         FIRST = .FALSE.
         CALL IWKIN (0)
      END IF
C                                  BOOKKEEPING OVERWRITTEN
      IF (LNOW.LT.LBOOK .OR. LNOW.GT.LUSED .OR. LUSED.GT.LMAX .OR.
     &    LNOW.GE.LBND .OR. LOUT.GT.LALC) THEN
         CALL E1MES (5, 7, 'One or more of the first eight '//
     &               'bookkeeping locations in IWKSP have been '//
     &               'overwritten.')
      ELSE IF (ITYPE.LE.0 .OR. ITYPE.GE.7) THEN
C                                  ILLEGAL DATA TYPE REQUESTED
         CALL E1MES (5, 8, 'Illegal data type requested.')
      ELSE
C                                  THIS CALCULATION ALLOWS FOR THE
C                                  TWO POINTER LOCATIONS IN THE STACK
C                                  WHICH ARE ASSIGNED TO EACH ALLOCATION
         I1KQU = MAX0(((LBND-3)*ISIZE(2))/ISIZE(ITYPE)-(LNOW*ISIZE(2)-
     &           1)/ISIZE(ITYPE)-1,0)
      END IF
C
      CALL E1POP ('I1KQU ')
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U6IAH/DU6IAH (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Calculated the perturbed Cholesky decomposition.
C
C  Usage:      CALL U6IAH (N, A, LDA, OFLMAX, ADDMAX)
C
C  Arguments:
C     N      - Dimension of the problem.  (Input)
C     A      - Real N by N matrix.  (Input/Output)
C              On input, A is the symmetric positive definite matrix
C                 with only the lower triangle and diagonal stored.
C              On output, A contains L of the Cholesky factorization of
C                 the perturbed matrix in the lower triangular part and
C                 diagonal.
C     LDA    - Row dimension of A exactly as specified in the dimension
C              statement of the calling program.  (Input)
C     OFLMAX - Bound on the size of the off-diagonal elements of the
C              Cholesky factors.  (Input)
C     ADDMAX - Maximum amount implicitly added to diagonal of "A" in
C              forming the Cholesky decomposition of A+AMU*I.  (Output)
C
C  Remark:
C     This is based on Algorithm A5.5.2, page 318, Dennis-Schnabel book.
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
      SUBROUTINE U6IAH (N, A, LDA, OFLMAX, ADDMAX)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, LDA
      REAL       OFLMAX, ADDMAX, A(LDA,*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, INDMAX, J
      REAL       AMINJJ, AMINL, AMINL2, EPS, TEMP
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS,AMAX1,SQRT
      INTRINSIC  ABS, AMAX1, SQRT
      REAL       ABS, AMAX1, SQRT
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   SSCAL
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   AMACH, ISAMAX, SDOT, SNRM2
      INTEGER    ISAMAX
      REAL       AMACH, SDOT, SNRM2
C
      EPS = AMACH(4)
      AMINL = EPS**0.25E0*OFLMAX
      AMINL2 = SQRT(EPS)*OFLMAX
C                                  "A" IS KNOWN TO BE POSITIVE DEFINITE
      IF (OFLMAX .EQ. 0.0E0) THEN
         INDMAX = ISAMAX(N,A(1,1),LDA+1)
         OFLMAX = SQRT(ABS(A(INDMAX,INDMAX)))
         AMINL2 = SQRT(EPS)*OFLMAX
      END IF
C
      ADDMAX = 0.0E0
      DO 20  J=1, N
C                                  FORM COLUMN J OF THE CHOL. FACTOR
         A(J,J) = A(J,J) - SNRM2(J-1,A(J,1),LDA)**2
         AMINJJ = 0.0E0
         DO 10  I=J + 1, N
            A(I,J) = A(I,J) - SDOT(J-1,A(I,1),LDA,A(J,1),LDA)
            AMINJJ = AMAX1(ABS(A(I,J)),AMINJJ)
   10    CONTINUE
         AMINJJ = AMAX1(AMINJJ/OFLMAX,AMINL)
         IF (A(J,J) .GT. AMINJJ*AMINJJ) THEN
            A(J,J) = SQRT(A(J,J))
         ELSE
            AMINJJ = AMAX1(AMINJJ,AMINL2)
            ADDMAX = AMAX1(ADDMAX,AMINJJ*AMINJJ-A(J,J))
            A(J,J) = AMINJJ
         END IF
C
         TEMP = 1.0E0/A(J,J)
         IF (J .LT. N) CALL SSCAL (N-J, TEMP, A(J+1,J), 1)
   20 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  I1KRL
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1983
C
C  Purpose:    Deallocate the last N allocations made in the workspace.
C              stack by I1KGT
C
C  Usage:      CALL I1KRL(N)
C
C  Arguments:
C     N      - Number of allocations to be released top down (Input)
C
C  Copyright:  1983 by IMSL, Inc.  All Rights Reserved
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE I1KRL (N)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IN, LALC, LBND, LBOOK, LMAX, LNEED, LNOW, LOUT,
     &           LUSED, NDX, NEXT
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      LOGICAL    FIRST
      SAVE       FIRST
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
C                                  SPECIFICATIONS FOR EQUIVALENCE
      EQUIVALENCE (LOUT, IWKSP(1))
      EQUIVALENCE (LNOW, IWKSP(2))
      EQUIVALENCE (LUSED, IWKSP(3))
      EQUIVALENCE (LBND, IWKSP(4))
      EQUIVALENCE (LMAX, IWKSP(5))
      EQUIVALENCE (LALC, IWKSP(6))
      EQUIVALENCE (LNEED, IWKSP(7))
      EQUIVALENCE (LBOOK, IWKSP(8))
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1STI, IWKIN
C
      DATA FIRST/.TRUE./
C
      IF (FIRST) THEN
C                                  INITIALIZE WORKSPACE IF NEEDED
         FIRST = .FALSE.
         CALL IWKIN (0)
      END IF
C                                  CALLING I1KRL(0) WILL CONFIRM
C                                  INTEGRITY OF SYSTEM AND RETURN
      IF (N .LT. 0) THEN
         CALL E1MES (5, 10, 'Error from subroutine I1KRL:  Attempt'//
     &               ' to release a negative number of workspace'//
     &               ' allocations. ')
         GO TO 9000
      END IF
C                                  BOOKKEEPING OVERWRITTEN
      IF (LNOW.LT.LBOOK .OR. LNOW.GT.LUSED .OR. LUSED.GT.LMAX .OR.
     &    LNOW.GE.LBND .OR. LOUT.GT.LALC) THEN
         CALL E1MES (5, 11, 'Error from subroutine I1KRL:  One or '//
     &               'more of the first eight bookkeeping locations '//
     &               'in IWKSP have been overwritten.  ')
         GO TO 9000
      END IF
C                                  CHECK ALL THE POINTERS IN THE
C                                  PERMANENT STORAGE AREA.  THEY MUST
C                                  BE MONOTONE INCREASING AND LESS THAN
C                                  OR EQUAL TO LMAX, AND THE INDEX OF
C                                  THE LAST POINTER MUST BE LMAX+1.
      NDX = LBND
      IF (NDX .NE. LMAX+1) THEN
         DO 10  I=1, LALC
            NEXT = IWKSP(NDX)
            IF (NEXT .EQ. LMAX+1) GO TO 20
C
            IF (NEXT.LE.NDX .OR. NEXT.GT.LMAX) THEN
               CALL E1MES (5, 12, 'Error from subroutine I1KRL:  '//
     &                     'A pointer in permanent storage has been '//
     &                     ' overwritten. ')
               GO TO 9000
            END IF
            NDX = NEXT
   10    CONTINUE
         CALL E1MES (5, 13, 'Error from subroutine I1KRL:  A '//
     &               'pointer in permanent storage has been '//
     &               'overwritten. ')
         GO TO 9000
      END IF
   20 IF (N .GT. 0) THEN
         DO 30  IN=1, N
            IF (LNOW .LE. LBOOK) THEN
               CALL E1MES (5, 14, 'Error from subroutine I1KRL:  '//
     &                     'Attempt to release a nonexistant '//
     &                     'workspace  allocation. ')
               GO TO 9000
            ELSE IF (IWKSP(LNOW).LT.LBOOK .OR. IWKSP(LNOW).GE.LNOW-1)
     &              THEN
C                                  CHECK TO MAKE SURE THE BACK POINTERS
C                                  ARE MONOTONE.
               CALL E1STI (1, LNOW)
               CALL E1MES (5, 15, 'Error from subroutine I1KRL:  '//
     &                     'The pointer at IWKSP(%(I1)) has been '//
     &                     'overwritten.  ')
               GO TO 9000
            ELSE
               LOUT = LOUT - 1
               LNOW = IWKSP(LNOW)
            END IF
   30    CONTINUE
      END IF
C
 9000 RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1UCS
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 8, 1984
C
C  Purpose:    To update the checksum number for error messages.
C
C  Usage:      CALL E1UCS
C
C  Arguments:  None
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1UCS
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IBEG, IBEG2, IEND, ILOC, IPOS, JLOC, NCODE, NLEN
      DOUBLE PRECISION DNUM
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      DOUBLE PRECISION DMAX
      CHARACTER  BLANK(1), COMMA(1), EQUAL(1), LPAR(1)
      SAVE       BLANK, COMMA, DMAX, EQUAL, LPAR
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  DMOD
      INTRINSIC  DMOD
      DOUBLE PRECISION DMOD
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   S1ANUM
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   ICASE, I1X
      INTEGER    ICASE, I1X
C
      DATA BLANK(1)/' '/, COMMA(1)/','/, LPAR(1)/'('/
      DATA EQUAL(1)/'='/, DMAX/1.0D+9/
C
      IF (MSGLEN .GT. 1) THEN
         IPOS = 0
         IBEG2 = 1
   10    IBEG = IBEG2
         IEND = MSGLEN
C                                  LOOK FOR BLANK, COMMA, LEFT PAREN.,
C                                  OR EQUAL SIGN
         ILOC = I1X(MSGSAV(IBEG),IEND-IBEG+1,BLANK,1)
         JLOC = I1X(MSGSAV(IBEG),IEND-IBEG+1,COMMA,1)
         IF (ILOC.EQ.0 .OR. (JLOC.GT.0.AND.JLOC.LT.ILOC)) ILOC = JLOC
         JLOC = I1X(MSGSAV(IBEG),IEND-IBEG+1,LPAR,1)
         IF (ILOC.EQ.0 .OR. (JLOC.GT.0.AND.JLOC.LT.ILOC)) ILOC = JLOC
         JLOC = I1X(MSGSAV(IBEG),IEND-IBEG+1,EQUAL,1)
         IF (ILOC.EQ.0 .OR. (JLOC.GT.0.AND.JLOC.LT.ILOC)) ILOC = JLOC
         IF (ILOC .GE. 1) THEN
            CALL S1ANUM (MSGSAV(IBEG+ILOC), IEND-IBEG-ILOC+1, NCODE,
     &                   NLEN)
            IF (NCODE.EQ.2 .OR. NCODE.EQ.3) THEN
C                                  FLOATING POINT NUMBER FOUND.
C                                  SET POINTERS TO SKIP OVER IT
               IBEG2 = IBEG + ILOC + NLEN
               IF (IBEG2 .LE. MSGLEN) THEN
                  CALL S1ANUM (MSGSAV(IBEG2), IEND-IBEG2+1, NCODE,
     &                         NLEN)
                  IF ((MSGSAV(IBEG2).EQ.'+'.OR.MSGSAV(IBEG2).EQ.
     &                '-') .AND. (NCODE.EQ.1.OR.NCODE.EQ.2)) THEN
C                                  INTEGER IMMEDIATELY FOLLOWS A REAL AS
C                                  WITH SOME CDC NOS. LIKE 1.2345678+123
C                                  SET POINTERS TO SKIP OVER IT
                     IF (NCODE.EQ.2 .AND. MSGSAV(IBEG2+NLEN-1).EQ.
     &                   '.') THEN
C                                  DO NOT SKIP AN END-OF-SENTENCE PERIOD
                        IBEG2 = IBEG2 + NLEN - 1
                     ELSE
                        IBEG2 = IBEG2 + NLEN
                     END IF
                  END IF
               END IF
            ELSE
               IBEG2 = IBEG + ILOC
            END IF
            IEND = IBEG + ILOC - 1
         END IF
C                                  UPDATE CKSUM USING PART OF MESSAGE
         DO 20  I=IBEG, IEND
            IPOS = IPOS + 1
            DNUM = ICASE(MSGSAV(I))
            ERCKSM = DMOD(ERCKSM+DNUM*IPOS,DMAX)
   20    CONTINUE
C                                  GO BACK FOR MORE IF NEEDED
         IF (IEND.LT.MSGLEN .AND. IBEG2.LT.MSGLEN) GO TO 10
C                                  UPDATE CKSUM USING ERROR TYPE
         DNUM = ERTYPE(CALLVL)
         ERCKSM = DMOD(ERCKSM+DNUM*(IPOS+1),DMAX)
C                                  UPDATE CKSUM USING ERROR CODE
         DNUM = ERCODE(CALLVL)
         ERCKSM = DMOD(ERCKSM+DNUM*(IPOS+2),DMAX)
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  I1DX (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 9, 1985
C
C  Purpose:    Determine the array subscript indicating the starting
C              element at which a key character sequence begins.
C              (Case-insensitive version)
C
C  Usage:      I1DX(CHRSTR, I1LEN, KEY, KLEN)
C
C  Arguments:
C     CHRSTR - Character array to be searched.  (Input)
C     I1LEN  - Length of CHRSTR.  (Input)
C     KEY    - Character array that contains the key sequence.  (Input)
C     KLEN   - Length of KEY.  (Input)
C     I1DX   - Integer function.  (Output)
C
C  Remarks:
C  1. Returns zero when there is no match.
C
C  2. Returns zero if KLEN is longer than ISLEN.
C
C  3. Returns zero when any of the character arrays has a negative or
C     zero length.
C
C  GAMS:       N5c
C
C  Chapter:    MATH/LIBRARY Utilities
C
C  Copyright:  1985 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION I1DX (CHRSTR, I1LEN, KEY, KLEN)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    I1LEN, KLEN
      CHARACTER  CHRSTR(*), KEY(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, II, J
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   ICASE, I1CSTR
      INTEGER    ICASE, I1CSTR
C
      I1DX = 0
      IF (KLEN.LE.0 .OR. I1LEN.LE.0) GO TO 9000
      IF (KLEN .GT. I1LEN) GO TO 9000
C
      I = 1
      II = I1LEN - KLEN + 1
   10 IF (I .LE. II) THEN
         IF (ICASE(CHRSTR(I)) .EQ. ICASE(KEY(1))) THEN
            IF (KLEN .NE. 1) THEN
               J = KLEN - 1
               IF (I1CSTR(CHRSTR(I+1),J,KEY(2),J) .EQ. 0) THEN
                  I1DX = I
                  GO TO 9000
               END IF
            ELSE
               I1DX = I
               GO TO 9000
            END IF
         END IF
         I = I + 1
         GO TO 10
      END IF
C
 9000 RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  CSFRG/DCSFRG (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 23, 1985
C
C  Purpose:    Extend a real symmetric matrix defined in its upper
C              triangle to its lower triangle.
C
C  Usage:      CALL CSFRG (N, A, LDA)
C
C  Arguments:
C     N      - Order of the matrix A.  (Input)
C     A      - N x N symmetric matrix to be filled out.  (Input/Output)
C     LDA    - Leading dimension of A exactly as specified in the
C              dimension statement of the calling program.  (Input)
C
C  GAMS:       D1b9
C
C  Chapter:    MATH/LIBRARY Basic Matrix/Vector Operations
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE CSFRG (N, A, LDA)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, LDA
      REAL       A(LDA,*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, E1POP, E1PSH, E1STI, SCOPY
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   N1RCD
      INTEGER    N1RCD
C
      CALL E1PSH ('CSFRG ')
C
      IF (N .LE. 0) THEN
         CALL E1STI (1, N)
         CALL E1MES (5, 1, 'N = %(I1).  The order of A, N, '//
     &               'must be greater than 0.')
      END IF
C
      IF (N .GT. LDA) THEN
         CALL E1STI (1, N)
         CALL E1STI (2, LDA)
         CALL E1MES (5, 2, 'N = %(I1) and LDA = %(I2).  The order of '//
     &               'A, N, must be less than or equal to the '//
     &               'leading dimension of A, LDA.')
      END IF
      IF (N1RCD(0) .NE. 0) GO TO 9000
C                                  Copy upper triangular values to lower
C                                  triangular values
      DO 10  I=1, N - 1
         CALL SCOPY (N-I, A(I,I+1), LDA, A(I+1,I), 1)
   10 CONTINUE
C                                  Exit section
 9000 CALL E1POP ('CSFRG ')
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  SVCAL (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    May 7, 1987
C
C  Purpose:    Multiply a vector by a scalar and store the result in
C              another vector, y = ax, all single precision.
C
C  Usage:      CALL SVCAL (N, SA, SX, INCX, SY, INCY)
C
C  Arguments:
C     N      - Length of vectors X.  (Input)
C     SA     - Real scalar.  (Input)
C     SX     - Real vector of length MAX(N*IABS(INCX),1).  (Input)
C              SVCAL computes SA*X(I) for I = 1,...,N. X(I) refers
C              to a specific element of SX.
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be SX(1+(I-1)*INCX). INCX must be
C              greater than 0.
C     SY     - Real vector of length MAX(N*IABS(INCY),1).  (Output)
C              SVCAL sets Y(I) equal to SA*X(I) for I = 1,...,N.
C              Y(I) refers to a specific element of SY.
C     INCY   - Displacement between elements of SY.  (Input)
C              Y(I) is defined to be SY(1+(I-1)*INCY). INCY must be
C              greater than 0.
C
C  GAMS:       D1a6
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE SVCAL (N, SA, SX, INCX, SY, INCY)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX, INCY
      REAL       SA, SX(*), SY(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IX, IY, M, MP1
C                                  SPECIFICATIONS FOR SPECIAL CASES
C     INTRINSIC  MOD
      INTRINSIC  MOD
      INTEGER    MOD
C
      IF (N .GT. 0) THEN
         IF (INCX.NE.1 .OR. INCY.NE.1) THEN
C                                  CODE FOR UNEQUAL INCREMENTS OR EQUAL
C                                  INCREMENTS NOT EQUAL TO 1
            IX = 1
            IY = 1
            DO 10  I=1, N
               SY(IY) = SA*SX(IX)
               IX = IX + INCX
               IY = IY + INCY
   10       CONTINUE
         ELSE
C                                  CODE FOR BOTH INCREMENTS EQUAL TO 1
            M = MOD(N,4)
C                                  CLEAN-UP LOOP
            DO 30  I=1, M
               SY(I) = SA*SX(I)
   30       CONTINUE
            MP1 = M + 1
            DO 40  I=MP1, N, 4
               SY(I) = SA*SX(I)
               SY(I+1) = SA*SX(I+1)
               SY(I+2) = SA*SX(I+2)
               SY(I+3) = SA*SX(I+3)
   40       CONTINUE
         END IF
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1STL
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    November 8, 1985
C
C  Purpose:    To store a string for subsequent use within an error
C              message.
C
C  Usage:      CALL E1STL(IL,STRING)
C
C  Arguments:
C     IL     - Integer specifying the substitution index.  IL must be
C              between 1 and 9.  (Input)
C     STRING - A character string.  (Input)
C
C  Copyright:  1985 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1STL (IL, STRING)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    IL
      CHARACTER  STRING*(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, LEN2
      CHARACTER  STRGUP(255)
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    IFINIT
      SAVE       IFINIT
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  IABS,LEN,MIN0
      INTRINSIC  IABS, LEN, MIN0
      INTEGER    IABS, LEN, MIN0
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1INIT, E1INPL
C
      DATA IFINIT/0/
C                                  INITIALIZE IF NECESSARY
      IF (IFINIT .EQ. 0) THEN
         CALL E1INIT
         IFINIT = 1
      END IF
      LEN2 = LEN(STRING)
      LEN2 = MIN0(LEN2,255)
      DO 10  I=1, LEN2
         STRGUP(I) = STRING(I:I)
   10 CONTINUE
      IF (IABS(IL).GE.1 .AND. IABS(IL).LE.9) THEN
         CALL E1INPL ('L', IL, LEN2, STRGUP)
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  SADD (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1986
C
C  Purpose:    Add a scalar to each component of a vector, x = x + a,
C              all single precision.
C
C  Usage:      CALL SADD (N, SA, SX, INCX)
C
C  Arguments:
C     N      - Length of vector X.  (Input)
C     SA     - Real scalar added to each element of X.  (Input)
C     SX     - Real vector of length MAX(N*IABS(INCX),1).
C              (Input/Output)
C              SADD replaces X(I) with X(I) + SA for I = 1,...N.
C              X(I) refers to a specific element of SX.
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be
C                 SX(1+(I-1)*INCX) if INCX.GE.0  or
C                 SX(1+(I-N)*INCX) if INCX.LT.0.
C
C  GAMS:       D1a
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE SADD (N, SA, SX, INCX)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX
      REAL       SA, SX(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, M, MP1, NINCX
C                                  SPECIFICATIONS FOR SPECIAL CASES
C     INTRINSIC  MOD
      INTRINSIC  MOD
      INTEGER    MOD
C
      IF (N .GT. 0) THEN
         IF (INCX .NE. 1) THEN
C                                  CODE FOR INCREMENT NOT EQUAL TO 1
            NINCX = N*INCX
            DO 10  I=1, NINCX, INCX
               SX(I) = SA + SX(I)
   10       CONTINUE
         ELSE
C                                  CODE FOR INCREMENT EQUAL TO 1
            M = MOD(N,5)
C                                  CLEAN-UP LOOP
            DO 30  I=1, M
               SX(I) = SA + SX(I)
   30       CONTINUE
            MP1 = M + 1
            DO 40  I=MP1, N, 5
               SX(I) = SA + SX(I)
               SX(I+1) = SA + SX(I+1)
               SX(I+2) = SA + SX(I+2)
               SX(I+3) = SA + SX(I+3)
               SX(I+4) = SA + SX(I+4)
   40       CONTINUE
         END IF
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  M1VE
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 5, 1984
C
C  Purpose:    Move a subset of one character array to another.
C
C  Usage:      CALL M1VE(INSTR, INBEG, INEND, INLEN, OUTSTR, OUTBEG,
C                         OUTEND, OUTLEN, IER)
C
C  Arguments:
C     INSTR  - Source character array.  (Input)
C     INBEG  - First element of INSTR to be moved.  (Input)
C     INEND  - Last element of INSTR to be moved.  (Input)
C              The source subset is INSTR(INBEG),...,INSTR(INEND).
C     INLEN  - Length of INSTR.  (Input)
C     OUTSTR - Destination character array.  (Output)
C     IUTBEG - First element of OUTSTR destination.  (Input)
C     IUTEND - Last element of OUTSTR  destination.  (Input)
C              The destination subset is OUTSRT(IUTBEG),...,
C              OUTSTR(IUTEND).
C     IUTLEN - Length of OUTSTR.  (Input)
C     IER    - Completion code.  (Output)
C              IER = -2  indicates that the input parameters, INBEG,
C                        INEND, INLEN, IUTBEG, IUTEND are not
C                        consistent.  One of the conditions
C                        INBEG.GT.0, INEND.GE.INBEG, INLEN.GE.INEND,
C                        IUTBEG.GT.0, or IUTEND.GE.IUTBEG is not
C                        satisfied.
C              IER = -1  indicates that the length of OUTSTR is
C                        insufficient to hold the subset of INSTR.
C                        That is, IUTLEN is less than IUTEND.
C              IER =  0  indicates normal completion
C              IER >  0  indicates that the specified subset of OUTSTR,
C                        OUTSTR(IUTBEG),...,OUTSTR(IUTEND) is not long
C                        enough to hold the subset INSTR(INBEG),...,
C                        INSTR(INEND) of INSTR.  IER is set to the
C                        number of characters that were not moved.
C
C  Remarks:
C  1. If the subset of OUTSTR is longer than the subset of INSTR,
C     trailing blanks are moved to OUTSTR.
C  2. If the subset of INSTR is longer than the subset of OUTSTR,
C     the shorter subset is moved to OUTSTR and IER is set to the number
C     of characters that were not moved to OUTSTR.
C  3. If the length of OUTSTR is insufficient to hold the subset,
C     IER is set to -2 and nothing is moved.
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE M1VE (INSTR, INBEG, INEND, INLEN, OUTSTR, IUTBEG,
     &                 IUTEND, IUTLEN, IER)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    INBEG, INEND, INLEN, IUTBEG, IUTEND, IUTLEN, IER
      CHARACTER  INSTR(*), OUTSTR(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    IUTLAS, KI, KO
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      CHARACTER  BLANK
      SAVE       BLANK
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  MIN0
      INTRINSIC  MIN0
      INTEGER    MIN0
C
      DATA BLANK/' '/
C                                  CHECK INBEG, INEND, INLEN, IUTBEG,
C                                  AND IUTEND
C
      IF (INBEG.LE.0 .OR. INEND.LT.INBEG .OR. INLEN.LT.INEND .OR.
     &    IUTBEG.LE.0 .OR. IUTEND.LT.IUTBEG) THEN
         IER = -2
         RETURN
      ELSE IF (IUTLEN .LT. IUTEND) THEN
         IER = -1
         RETURN
      END IF
C                                  DETERMINE LAST CHARACTER TO M1VE
      IUTLAS = IUTBEG + MIN0(INEND-INBEG,IUTEND-IUTBEG)
C                                  M1VE CHARACTERS
      KI = INBEG
      DO 10  KO=IUTBEG, IUTLAS
         OUTSTR(KO) = INSTR(KI)
         KI = KI + 1
   10 CONTINUE
C                                   SET IER TO NUMBER OF CHARACTERS THAT
C                                   WHERE NOT MOVED
      IER = KI - INEND - 1
C                                   APPEND BLANKS IF NECESSARY
      DO 20  KO=IUTLAS + 1, IUTEND
         OUTSTR(KO) = BLANK
   20 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  SASUM (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1986
C
C  Purpose:    Sum the absolute values of the components of a
C              single precision vector.
C
C  Usage:      SASUM(N, SX, INCX)
C
C  Arguments:
C     N      - Length of vectors X.  (Input)
C     SX     - Real vector of length N*INCX.  (Input)
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be SX(1+(I-1)*INCX). INCX must be
C              greater than 0.
C     SASUM  - Single precision sum from I=1 to N of ABS(X(I)).
C              (Output)
C              X(I) refers to a specific element of SX.
C
C  GAMS:       D1a
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      REAL FUNCTION SASUM (N, SX, INCX)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX
      REAL       SX(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, M, MP1, NINCX
C                                  SPECIFICATIONS FOR SPECIAL CASES
C     INTRINSIC  MOD
      INTRINSIC  MOD
      INTEGER    MOD
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS
      INTRINSIC  ABS
      REAL       ABS
C
      SASUM = 0.0E0
      IF (N .GT. 0) THEN
         IF (INCX .NE. 1) THEN
C                                  CODE FOR INCREMENT NOT EQUAL TO 1
            NINCX = N*INCX
            DO 10  I=1, NINCX, INCX
               SASUM = SASUM + ABS(SX(I))
   10       CONTINUE
         ELSE
C                                  CODE FOR INCREMENT EQUAL TO 1
            M = MOD(N,6)
C                                  CLEAN-UP LOOP
            DO 30  I=1, M
               SASUM = SASUM + ABS(SX(I))
   30       CONTINUE
            MP1 = M + 1
            DO 40  I=MP1, N, 6
               SASUM = SASUM + ABS(SX(I)) + ABS(SX(I+1)) +
     &                 ABS(SX(I+2)) + ABS(SX(I+3)) + ABS(SX(I+4)) +
     &                 ABS(SX(I+5))
   40       CONTINUE
         END IF
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  I1ERIF
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 13, 1984
C
C  Purpose:    Return the position of the first element of a given
C              character array which is not an element of another
C              character array.
C
C  Usage:      I1ERIF(STR1, LEN1, STR2, LEN2)
C
C  Arguments:
C     STR1   - Character array to be searched.  (Input)
C     LEN1   - Length of STR1.  (Input)
C     STR2   - Character array to be searched for.  (Input)
C     LEN2   - Length of STR2.  (Input)
C     I1ERIF - Integer function.  (Output)
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION I1ERIF (STR1, LEN1, STR2, LEN2)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    LEN1, LEN2
      CHARACTER  STR1(*), STR2(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   I1X
      INTEGER    I1X
C                              FIRST EXECUTABLE STATEMENT
      IF (LEN1.LE.0 .OR. LEN2.LE.0) THEN
         I1ERIF = 1
      ELSE
         DO 10  I=1, LEN1
            IF (I1X(STR2,LEN2,STR1(I),1) .EQ. 0) THEN
               I1ERIF = I
               RETURN
            END IF
   10    CONTINUE
         I1ERIF = 0
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  M1VECH
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    December 31, 1984
C
C  Purpose:    Character substring assignment.
C
C  Usage:      CALL M1VECH (STR1, LEN1, STR2, LEN2)
C
C  Arguments:
C     STR1   - Source substring.  (Input)
C              The source substring is STR1(1:LEN1).
C     LEN1   - Length of STR1.  (Input)
C     STR2   - Destination substring.  (Output)
C              The destination substring is STR2(1:LEN2).
C     LEN2   - Length of STR2.  (Input)
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE M1VECH (STR1, LEN1, STR2, LEN2)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    LEN1, LEN2
      CHARACTER  STR1*(*), STR2*(*)
C
      STR2(1:LEN2) = STR1(1:LEN1)
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  SSCAL (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1986
C
C  Purpose:    Multiply a vector by a scalar, y = ay, both single
C              precision.
C
C  Usage:      CALL SSCAL (N, SA, SX, INCX)
C
C  Arguments:
C     N      - Length of vector X.  (Input)
C     SA     - Real scalar.  (Input)
C     SX     - Real vector of length N*INCX.  (Input/Output)
C              SSCAL replaces X(I) with SA*X(I) for I=1,...,N. X(I)
C              refers to a specific element of SX. See INCX argument
C              description.
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be SX(1+(I-1)*INCX). INCX must be
C              greater than zero.
C
C  GAMS:       D1a6
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE SSCAL (N, SA, SX, INCX)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX
      REAL       SA, SX(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, M, MP1, NS
C
      IF (N .GT. 0) THEN
         IF (INCX .NE. 1) THEN
C                                  CODE FOR INCREMENTS NOT EQUAL TO 1.
            NS = N*INCX
            DO 10  I=1, NS, INCX
               SX(I) = SA*SX(I)
   10       CONTINUE
         ELSE
C                                  CODE FOR INCREMENTS EQUAL TO 1.
C                                  CLEAN-UP LOOP SO REMAINING VECTOR
C                                  LENGTH IS A MULTIPLE OF 5.
            M = N - (N/5)*5
            DO 30  I=1, M
               SX(I) = SA*SX(I)
   30       CONTINUE
            MP1 = M + 1
            DO 40  I=MP1, N, 5
               SX(I) = SA*SX(I)
               SX(I+1) = SA*SX(I+1)
               SX(I+2) = SA*SX(I+2)
               SX(I+3) = SA*SX(I+3)
               SX(I+4) = SA*SX(I+4)
   40       CONTINUE
         END IF
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U13NF/DU13NF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Solve L*s = g for s.
C
C  Usage:      CALL U13NF (N, H, LDH, GC, SNWTN)
C
C  Arguments:
C     N      - Length of the vectors GC, SNWTN.  (Input)
C     H      - N by N matrix containing the Cholesky factor of the
C              Hessian in the lower triangle and diagonal.  (Input)
C     LDH    - Leading dimension of H exactly as specified in the
C              dimension statement of the calling program.  (Input)
C     GC     - Vector of length N containing the current gradient.
C              (Input)
C     SNWTN  - Vector of length N containing the solution.  (Output)
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
      SUBROUTINE U13NF (N, H, LDH, GC, SNWTN)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, LDH
      REAL       H(LDH,*), GC(*), SNWTN(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I
      REAL       SUM
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   SDOT
      REAL       SDOT
C
      SNWTN(1) = GC(1)/H(1,1)
      DO 10  I=2, N
         SUM = SDOT(I-1,H(I,1),LDH,SNWTN,1)
         SNWTN(I) = (GC(I)-SUM)/H(I,I)
   10 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U14NF/DU14NF (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Solve TRANS(L)*s = y for s.
C
C  Usage:      CALL U14NF (N, H, LDH, Y, SNWTN)
C
C  Arguments:
C     N      - Length of the vector SNWTN.  (Input)
C     H      - N by N matrix containing the Cholesky factor of the
C              Hessian in the lower triangle and diagonal.  (Input)
C     LDH    - Leading dimension of H exactly as specified in the
C              dimension statement of the calling program.  (Input)
C     Y      - Vector of length N containing the right-hand-side.
C              (Input)
C     SNWTN  - Vector of length N containing the solution.  (Output)
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
      SUBROUTINE U14NF (N, H, LDH, Y, SNWTN)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, LDH
      REAL       H(LDH,*), Y(*), SNWTN(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I
      REAL       SUM
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   SDOT
      REAL       SDOT
C
      SNWTN(N) = Y(N)/H(N,N)
      DO 10  I=N - 1, 1, -1
         SUM = SDOT(N-I,H(I+1,I),1,SNWTN(I+1),1)
         SNWTN(I) = (Y(I)-SUM)/H(I,I)
   10 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  SDOT (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1986
C
C  Purpose:    Compute the single-precision dot product x*y.
C
C  Usage:      SDOT(N, SX, INCX, SY, INCY)
C
C  Arguments:
C     N      - Length of vectors X and Y.  (Input)
C     SX     - Real vector of length MAX(N*IABS(INCX),1).  (Input)
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be.. SX(1+(I-1)*INCX) if INCX .GE. 0
C              or SX(1+(I-N)*INCX) if INCX .LT. 0.
C     SY     - Real vector of length MAX(N*IABS(INCY),1).  (Input)
C     INCY   - Displacement between elements of SY.  (Input)
C              Y(I) is defined to be.. SY(1+(I-1)*INCY) if INCY .GE. 0
C              or SY(1+(I-N)*INCY) if INCY .LT. 0.
C     SDOT   - Sum from I=1 to N of X(I)*Y(I).  (Output)
C              X(I) and Y(I) refer to specific elements of SX and SY,
C              respectively.  See INCX and INCY argument descriptions.
C
C  GAMS:       D1a4
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      REAL FUNCTION SDOT (N, SX, INCX, SY, INCY)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX, INCY
      REAL       SX(*), SY(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IX, IY, M, MP1
C                                  SPECIFICATIONS FOR SPECIAL CASES
C     INTRINSIC  MOD
      INTRINSIC  MOD
      INTEGER    MOD
C
      SDOT = 0.0E0
      IF (N .GT. 0) THEN
         IF (INCX.NE.1 .OR. INCY.NE.1) THEN
C                                  CODE FOR UNEQUAL INCREMENTS
            IX = 1
            IY = 1
            IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
            IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
            DO 10  I=1, N
               SDOT = SDOT + SX(IX)*SY(IY)
               IX = IX + INCX
               IY = IY + INCY
   10       CONTINUE
         ELSE
C                                  CODE FOR BOTH INCREMENTS EQUAL TO 1
            M = MOD(N,5)
C                                  CLEAN-UP LOOP SO REMAINING VECTOR
            DO 30  I=1, M
               SDOT = SDOT + SX(I)*SY(I)
   30       CONTINUE
            MP1 = M + 1
            DO 40  I=MP1, N, 5
               SDOT = SDOT + SX(I)*SY(I) + SX(I+1)*SY(I+1) +
     &                SX(I+2)*SY(I+2) + SX(I+3)*SY(I+3) +
     &                SX(I+4)*SY(I+4)
   40       CONTINUE
         END IF
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  C1TCI
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 13, 1984
C
C  Purpose:    Convert character string into corresponding integer
C              form.
C
C  Usage:      CALL C1TCI (CHRSTR, SLEN, NUM, IER)
C
C  Arguments:
C   CHRSTR  - Character array that contains the number description.
C             (Input)
C   SLEN    - Length of the character array.  (Input)
C   NUM     - The answer.  (Output)
C   IER     - Completion code.  (Output)  Where
C                IER =-2  indicates that the number is too large to
C                         be converted;
C                IER =-1  indicates that SLEN <= 0;
C                IER = 0  indicates normal completion;
C                IER > 0  indicates that the input string contains a
C                         nonnumeric character.  IER is the index of
C                         the first nonnumeric character in CHRSTR.
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE C1TCI (CHRSTR, SLEN, NUM, IER)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    SLEN, NUM, IER
      CHARACTER  CHRSTR(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    COUNT, I, IMACH5, J, N, S, SIGN
      CHARACTER  ZERO
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      CHARACTER  BLANK, DIGIT*10, MINUS, PLUS
      SAVE       BLANK, DIGIT, MINUS, PLUS
C                                  SPECIFICATIONS FOR EQUIVALENCE
      EQUIVALENCE (DIGIT, ZERO)
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  INDEX
      INTRINSIC  INDEX
      INTEGER    INDEX
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   IMACH
      INTEGER    IMACH
C
      DATA DIGIT/'0123456789'/
      DATA BLANK/' '/, MINUS/'-'/, PLUS/'+'/
C
C                                  CHECK SLEN
      NUM = 0
      IF (SLEN .LE. 0) THEN
         IER = -1
         GO TO 50
      END IF
C                                  HANDLE LEADING BLANKS
      SIGN = 1
      I = 1
   10 IF (I .LE. SLEN) THEN
         IF (CHRSTR(I) .EQ. BLANK) THEN
            I = I + 1
            GO TO 10
         END IF
      ELSE
         IER = 1
         GO TO 50
      END IF
C                                  CHECK FOR SIGN, IF ANY
      S = I
      IF (CHRSTR(I) .EQ. MINUS) THEN
         SIGN = -1
         I = I + 1
      ELSE IF (CHRSTR(I) .EQ. PLUS) THEN
         I = I + 1
      END IF
   20 IF (I .LE. SLEN) THEN
         IF (CHRSTR(I) .EQ. BLANK) THEN
            I = I + 1
            GO TO 20
         END IF
      ELSE
         IER = S
         GO TO 50
      END IF
C                                  SKIP LEADING ZERO
      J = I
   30 IF (I .LE. SLEN) THEN
         IF (CHRSTR(I) .EQ. ZERO) THEN
            I = I + 1
            GO TO 30
         END IF
      ELSE
         IER = 0
         GO TO 50
      END IF
C                                  CHECK FIRST NONBLANK CHARACTER
      COUNT = 0
C                                  CHECK NUMERIC CHARACTERS
      IMACH5 = IMACH(5)
   40 N = INDEX(DIGIT,CHRSTR(I))
      IF (N .NE. 0) THEN
         COUNT = COUNT + 1
         IF (NUM .GT. ((IMACH5-N)+1)/10) THEN
            IER = -2
            GO TO 50
         ELSE
            NUM = NUM*10 - 1 + N
            I = I + 1
            IF (I .LE. SLEN) GO TO 40
         END IF
      END IF
C
      IF (COUNT .EQ. 0) THEN
         IF (I .GT. J) THEN
            IER = I
         ELSE
            IER = S
         END IF
      ELSE IF (I .GT. SLEN) THEN
         NUM = SIGN*NUM
         IER = 0
      ELSE
         NUM = SIGN*NUM
         IER = I
      END IF
C
   50 CONTINUE
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  SAXPY (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1986
C
C  Purpose:    Compute the scalar times a vector plus a vector,
C              y = ax + y, all single precision.
C
C  Usage:      CALL SAXPY (N, SA, SX, INCX, SY, INCY)
C
C  Arguments:
C     N      - Length of vectors X and Y.  (Input)
C     SA     - Real scalar.  (Input)
C     SX     - Real vector of length MAX(N*IABS(INCX),1).  (Input)
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be
C                 SX(1+(I-1)*INCX) if INCX.GE.0  or
C                 SX(1+(I-N)*INCX) if INCX.LT.0.
C     SY     - Real vector of length MAX(N*IABS(INCY),1).
C              (Input/Output)
C              SAXPY replaces Y(I) with SA*X(I) + Y(I) for I=1,...,N.
C              X(I) and Y(I) refer to specific elements of SX and SY.
C     INCY   - Displacement between elements of SY.  (Input)
C              Y(I) is defined to be
C                 SY(1+(I-1)*INCY) if INCY.GE.0  or
C                 SY(1+(I-N)*INCY) if INCY.LT.0.
C
C  GAMS:       D1a7
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE SAXPY (N, SA, SX, INCX, SY, INCY)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX, INCY
      REAL       SA, SX(*), SY(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IX, IY, M, MP1
C                                  SPECIFICATIONS FOR SPECIAL CASES
C     INTRINSIC  MOD
      INTRINSIC  MOD
      INTEGER    MOD
C
      IF (N .GT. 0) THEN
         IF (SA .NE. 0.0) THEN
            IF (INCX.NE.1 .OR. INCY.NE.1) THEN
C                                  CODE FOR UNEQUAL INCREMENTS OR EQUAL
C                                  INCREMENTS NOT EQUAL TO 1
               IX = 1
               IY = 1
               IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
               IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
               DO 10  I=1, N
                  SY(IY) = SY(IY) + SA*SX(IX)
                  IX = IX + INCX
                  IY = IY + INCY
   10          CONTINUE
            ELSE
C                                  CODE FOR BOTH INCREMENTS EQUAL TO 1
               M = MOD(N,4)
C                                  CLEAN-UP LOOP
               DO 30  I=1, M
                  SY(I) = SY(I) + SA*SX(I)
   30          CONTINUE
               MP1 = M + 1
               DO 40  I=MP1, N, 4
                  SY(I) = SY(I) + SA*SX(I)
                  SY(I+1) = SY(I+1) + SA*SX(I+1)
                  SY(I+2) = SY(I+2) + SA*SX(I+2)
                  SY(I+3) = SY(I+3) + SA*SX(I+3)
   40          CONTINUE
            END IF
         END IF
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  I1KST
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1983
C
C  Purpose:    Return control information about the workspace stack.
C
C  Usage:      I1KST(NFACT)
C
C  Arguments:
C     NFACT  - Integer value between 1 and 6 inclusive returns the
C                 following information: (Input)
C                   NFACT = 1 - LOUT: number of current allocations
C                               excluding permanent storage. At the
C                               end of a run, there should be no
C                               active allocations.
C                   NFACT = 2 - LNOW: current active length
C                   NFACT = 3 - LTOTAL: total storage used thus far
C                   NFACT = 4 - LMAX: maximum storage allowed
C                   NFACT = 5 - LALC: total number of allocations made
C                               by I1KGT thus far
C                   NFACT = 6 - LNEED: number of numeric storage units
C                               by which the stack size must be
C                               increased for all past allocations
C                               to succeed
C     I1KST  - Integer function. (Output) Returns a workspace stack
C              statistic according to value of NFACT.
C
C  Copyright:  1983 by IMSL, Inc.  All Rights Reserved
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION I1KST (NFACT)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    NFACT
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    ISTATS(7)
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      LOGICAL    FIRST
      SAVE       FIRST
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
C                                  SPECIFICATIONS FOR EQUIVALENCE
      EQUIVALENCE (ISTATS(1), IWKSP(1))
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES, IWKIN
C
      DATA FIRST/.TRUE./
C
      IF (FIRST) THEN
C                                  INITIALIZE WORKSPACE IF NEEDED
         FIRST = .FALSE.
         CALL IWKIN (0)
      END IF
C
      IF (NFACT.LE.0 .OR. NFACT.GE.7) THEN
         CALL E1MES (5, 9, 'Error from subroutine I1KST:  Argument'//
     &               ' for I1KST must be between 1 and 6 inclusive.')
      ELSE IF (NFACT .EQ. 1) THEN
C                                  LOUT
         I1KST = ISTATS(1)
      ELSE IF (NFACT .EQ. 2) THEN
C                                  LNOW + PERMANENT
         I1KST = ISTATS(2) + (ISTATS(5)-ISTATS(4)+1)
      ELSE IF (NFACT .EQ. 3) THEN
C                                  LUSED + PERMANENT
         I1KST = ISTATS(3) + (ISTATS(5)-ISTATS(4)+1)
      ELSE IF (NFACT .EQ. 4) THEN
C                                  LMAX
         I1KST = ISTATS(5)
      ELSE IF (NFACT .EQ. 5) THEN
C                                  LALC
         I1KST = ISTATS(6)
      ELSE IF (NFACT .EQ. 6) THEN
C                                  LNEED
         I1KST = ISTATS(7)
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1INPL
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 2, 1984
C
C  Purpose:    To store a character string in the parameter list PLIST
C              for use by the error message handler.
C
C  Usage:      CALL E1INPL(FORM,NUM,SLEN,STRUP)
C
C  Arguments:
C     FORM   - A character string of length one to be inserted into
C              PLIST which specifies the form of the string.  (Input)
C              For example, 'L' for string, 'A' for character array,
C              'I' for integer, 'K' for keyword (PROTRAN only).  An
C              asterisk is inserted into PLIST preceding FORM.
C     NUM    - Integer to be inserted as a character into PLIST
C              immediately following FORM.  (Input)  NUM must be between
C              1 and 9.
C     SLEN   - The number of characters in STRUP.  (Input)  LEN must be
C              less than or equal to 255.  The character representation
C              of SLEN is inserted into PLIST after NUM and an asterisk.
C     STRUP  - A character string of length LEN which is to be inserted
C              into PLIST.  (Input)  Trailing blanks are ignored.
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1INPL (FORM, NUM, SLEN, STRUP)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    NUM, SLEN
      CHARACTER  FORM, STRUP(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    IER, L, LEN2, LENCK, LOC, NLEN, NNUM
      CHARACTER  STRNCH(3)
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      CHARACTER  BLANK, PRCNT(1), TEMP(4)
      SAVE       BLANK, PRCNT, TEMP
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  IABS
      INTRINSIC  IABS
      INTEGER    IABS
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   C1TIC, M1VE
C
      DATA TEMP/'*', ' ', ' ', '*'/, PRCNT/'%'/, BLANK/' '/
C
      NNUM = IABS(NUM)
      LENCK = PLEN + SLEN + 8
      IF (NNUM.GE.1 .AND. NNUM.LE.9 .AND. LENCK.LE.300) THEN
         TEMP(2) = FORM
         CALL C1TIC (NNUM, TEMP(3), 1, IER)
         LOC = PLEN + 1
         IF (LOC .EQ. 2) LOC = 1
         CALL M1VE (TEMP, 1, 4, 4, PLIST(LOC), 1, 4, 262, IER)
         LOC = LOC + 4
         IF (NUM .LT. 0) THEN
            LEN2 = SLEN
         ELSE
            DO 10  L=1, SLEN
               LEN2 = SLEN - L + 1
               IF (STRUP(LEN2) .NE. BLANK) GO TO 20
   10       CONTINUE
            LEN2 = 1
   20       CONTINUE
         END IF
         NLEN = 1
         IF (LEN2 .GE. 10) NLEN = 2
         IF (LEN2 .GE. 100) NLEN = 3
         CALL C1TIC (LEN2, STRNCH, NLEN, IER)
         CALL M1VE (STRNCH, 1, NLEN, 3, PLIST(LOC), 1, NLEN, 262, IER)
         LOC = LOC + NLEN
         CALL M1VE (PRCNT, 1, 1, 1, PLIST(LOC), 1, 1, 262, IER)
         LOC = LOC + 1
         CALL M1VE (STRUP, 1, LEN2, LEN2, PLIST(LOC), 1, LEN2, 262,
     &              IER)
         PLEN = LOC + LEN2 - 1
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  SSWAP (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1986
C
C  Purpose:    Interchange vectors X and Y, both single precision.
C
C  Usage:      CALL SSWAP (N, SX, INCX, SY, INCY)
C
C  Arguments:
C     N      - Length of vectors X and Y.  (Input)
C     SX     - Real vector of length MAX(N*IABS(INCX),1).
C              (Input/Output)
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be
C                 SX(1+(I-1)*INCX) if INCX.GE.0  or
C                 SX(1+(I-N)*INCX) if INCX.LT.0.
C     SY     - Real vector of length MAX(N*IABS(INCY),1).
C              (Input/Output)
C     INCY   - Displacement between elements of SY.  (Input)
C              Y(I) is defined to be
C                 SY(1+(I-1)*INCY) if INCY.GE.0  or
C                 SY(1+(I-N)*INCY) if INCY.LT.0.
C
C  GAMS:       D1a5
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE SSWAP (N, SX, INCX, SY, INCY)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX, INCY
      REAL       SX(*), SY(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IX, IY, M, MP1
      REAL       STEMP
C                                  SPECIFICATIONS FOR SPECIAL CASES
C     INTRINSIC  MOD
      INTRINSIC  MOD
      INTEGER    MOD
C
      IF (N .GT. 0) THEN
         IF (INCX.NE.1 .OR. INCY.NE.1) THEN
C                                  CODE FOR UNEQUAL INCREMENTS OR EQUAL
C                                    INCREMENTS NOT EQUAL TO 1
            IX = 1
            IY = 1
            IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
            IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
            DO 10  I=1, N
               STEMP = SX(IX)
               SX(IX) = SY(IY)
               SY(IY) = STEMP
               IX = IX + INCX
               IY = IY + INCY
   10       CONTINUE
         ELSE
C                                  CODE FOR BOTH INCREMENTS EQUAL TO 1
            M = MOD(N,3)
C                                  CLEAN-UP LOOP
            DO 30  I=1, M
               STEMP = SX(I)
               SX(I) = SY(I)
               SY(I) = STEMP
   30       CONTINUE
            MP1 = M + 1
            DO 40  I=MP1, N, 3
               STEMP = SX(I)
               SX(I) = SY(I)
               SY(I) = STEMP
               STEMP = SX(I+1)
               SX(I+1) = SY(I+1)
               SY(I+1) = STEMP
               STEMP = SX(I+2)
               SX(I+2) = SY(I+2)
               SY(I+2) = STEMP
   40       CONTINUE
         END IF
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1PRT
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 14, 1984
C
C  Purpose:    To print an error message.
C
C  Usage:      CALL E1PRT
C
C  Arguments:  None
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1PRT
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    ALL, I, IBEG, IBLOC, IBLOC2, IEND, IER, IHDR, J,
     &           LERTYP, LOC, LOCM1, LOCX, MAXLOC, MAXTMP, MLOC, MOD,
     &           NCBEG, NLOC, NOUT
      CHARACTER  MSGTMP(70), STRING(10)
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      CHARACTER  ATLINE(9), BLANK(1), DBB(3), FROM(6), MSGTYP(8,7),
     &           PERSLA(2), QMARK, UNKNOW(8)
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                              SPECIFICATIONS FOR COMMON /ERCOM8/
      INTEGER    PROLVL, XXLINE(10), XXPLEN(10), ICALOC(10), INALOC(10)
      COMMON     /ERCOM8/ PROLVL, XXLINE, XXPLEN, ICALOC, INALOC
      SAVE       /ERCOM8/
C                              SPECIFICATIONS FOR COMMON /ERCOM9/
      CHARACTER  XXPROC(10)*31
      COMMON     /ERCOM9/ XXPROC
      SAVE       /ERCOM9/
      SAVE       ATLINE, BLANK, DBB, FROM, MSGTYP, PERSLA, QMARK,
     &           UNKNOW
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  MIN0
      INTRINSIC  MIN0
      INTEGER    MIN0
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   C1TIC, M1VE, UMACH
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   I1DX, I1ERIF
      INTEGER    I1DX, I1ERIF
C
      DATA MSGTYP/'N', 'O', 'T', 'E', ' ', ' ', ' ', ' ', 'A',
     &     'L', 'E', 'R', 'T', ' ', ' ', ' ', 'W', 'A', 'R',
     &     'N', 'I', 'N', 'G', ' ', 'F', 'A', 'T', 'A', 'L',
     &     ' ', ' ', ' ', 'T', 'E', 'R', 'M', 'I', 'N', 'A',
     &     'L', 'W', 'A', 'R', 'N', 'I', 'N', 'G', ' ', 'F',
     &     'A', 'T', 'A', 'L', ' ', ' ', ' '/
      DATA UNKNOW/'U', 'N', 'K', 'N', 'O', 'W', 'N', ' '/
      DATA ATLINE/' ', 'a', 't', ' ', 'l', 'i', 'n', 'e', ' '/
      DATA BLANK/' '/, FROM/' ', 'f', 'r', 'o', 'm', ' '/
      DATA DBB/'.', ' ', ' '/, PERSLA/'%', '/'/
      DATA QMARK/'?'/
C
      IF (MSGLEN .LE. 0) RETURN
      CALL UMACH (2, NOUT)
      MAXTMP = 70
      MOD = 0
      LERTYP = ERTYPE(CALLVL)
      IHDR = HDRFMT(LERTYP)
      IF (IHDR .EQ. 3) THEN
         IF (XXPROC(PROLVL)(1:1).EQ.QMARK .AND. XXLINE(PROLVL).EQ.0)
     &       THEN
            IHDR = 1
         END IF
      END IF
      IEND = 0
      IF (IHDR.EQ.1 .AND. ERTYPE(CALLVL).LE.4) THEN
         MSGTMP(1) = BLANK(1)
         IEND = 1
C                                  CONVERT ERROR CODE INTO CHAR STRING
         CALL C1TIC (ERCODE(CALLVL), STRING, 10, IER)
C                                  LOCATE START OF NON-BLANK CHARACTERS
         IBEG = I1ERIF(STRING,10,BLANK,1)
C                                  M1VE IT TO MSGTMP
         CALL M1VE (STRING, IBEG, 10, 10, MSGTMP, IEND+1,
     &              IEND+11-IBEG, MAXTMP, IER)
         IEND = IEND + 11 - IBEG
      END IF
      IF (IHDR .NE. 2) THEN
         CALL M1VE (FROM, 1, 6, 6, MSGTMP, IEND+1, IEND+6, MAXTMP, IER)
         IEND = IEND + 6
      END IF
      IF (IHDR .EQ. 3) THEN
C                                  THIS IS A PROTRAN RUN TIME ERROR MSG.
C                                  RETRIEVE THE PROCEDURE NAME
         CALL M1VE (XXPROC(PROLVL), 1, XXPLEN(PROLVL), 31, MSGTMP,
     &              IEND+1, IEND+XXPLEN(PROLVL), MAXTMP, IER)
         MLOC = IEND + XXPLEN(PROLVL) + 1
         MSGTMP(MLOC) = BLANK(1)
         IEND = IEND + I1DX(MSGTMP(IEND+1),XXPLEN(PROLVL)+1,BLANK,1) -
     &          1
         IF (XXLINE(PROLVL) .GT. 0) THEN
C                                  INSERT ATLINE
            CALL M1VE (ATLINE, 1, 9, 9, MSGTMP, IEND+1, IEND+9,
     &                 MAXTMP, IER)
            IEND = IEND + 9
C                                  CONVERT PROTRAN GLOBAL LINE NUMBER
            CALL C1TIC (XXLINE(PROLVL), STRING, 10, IER)
C                                  LOCATE START OF NON-BLANK CHARACTERS
            IBEG = I1ERIF(STRING,10,BLANK,1)
C                                  M1VE GLOBAL LINE NUMBER TO MSGTMP
            CALL M1VE (STRING, IBEG, 10, 10, MSGTMP, IEND+1,
     &                 IEND+11-IBEG, MAXTMP, IER)
            IEND = IEND + 11 - IBEG
         END IF
      ELSE
C                                  THIS IS EITHER A LIBRARY ERROR MSG
C                                  OR A PROTRAN PREPROCESSOR ERROR MSG
         IF (IHDR .EQ. 1) THEN
C                                  THIS IS A LIBRARY ERROR MESSAGE.
C                                  RETRIEVE ROUTINE NAME
            CALL M1VE (RNAME(CALLVL), 1, 6, 6, MSGTMP, IEND+1, IEND+6,
     &                 MAXTMP, IER)
            MSGTMP(IEND+7) = BLANK(1)
            IEND = IEND + I1DX(MSGTMP(IEND+1),7,BLANK,1) - 1
         END IF
C                                  ADD DOT, BLANK, BLANK IF NEEDED
         IF (I1DX(MSGSAV,3,DBB,3) .NE. 1) THEN
            CALL M1VE (DBB, 1, 3, 3, MSGTMP, IEND+1, IEND+3, MAXTMP,
     &                 IER)
            IEND = IEND + 3
            MOD = 3
         END IF
      END IF
C                                  MSGTMP AND MSGSAV NOW CONTAIN THE
C                                   ERROR MESSAGE IN FINAL FORM.
      NCBEG = 59 - IEND - MOD
      ALL = 0
      IBLOC = I1DX(MSGSAV,MSGLEN,PERSLA,2)
      IF (IBLOC.NE.0 .AND. IBLOC.LT.NCBEG) THEN
         LOCM1 = IBLOC - 1
         LOC = IBLOC + 1
      ELSE IF (MSGLEN .LE. NCBEG) THEN
         LOCM1 = MSGLEN
         ALL = 1
      ELSE
         LOC = NCBEG
C                                  CHECK FOR APPROPRIATE PLACE TO SPLIT
   10    CONTINUE
         IF (MSGSAV(LOC) .NE. BLANK(1)) THEN
            LOC = LOC - 1
            IF (LOC .GT. 1) GO TO 10
            LOC = NCBEG + 1
         END IF
         LOCM1 = LOC - 1
      END IF
C                                  NO BLANKS FOUND IN FIRST NCBEG CHARS
      IF (LERTYP.GE.1 .AND. LERTYP.LE.7) THEN
         WRITE (NOUT,99995) (MSGTYP(I,LERTYP),I=1,8),
     &                     (MSGTMP(I),I=1,IEND), (MSGSAV(I),I=1,LOCM1)
      ELSE
         WRITE (NOUT,99995) (UNKNOW(I),I=1,8), (MSGTMP(I),I=1,IEND),
     &                     (MSGSAV(I),I=1,LOCM1)
      END IF
      IF (ALL .EQ. 0) THEN
C                                  PREPARE TO WRITE CONTINUATION OF
C                                    MESSAGE
C
C                                  FIND WHERE TO BREAK MESSAGE
C                                    LOC = NUMBER OF CHARACTERS OF
C                                          MESSAGE WRITTEN SO FAR
   20    LOCX = LOC + 64
         NLOC = LOC + 1
         IBLOC2 = IBLOC
         MAXLOC = MIN0(MSGLEN-LOC,64)
         IBLOC = I1DX(MSGSAV(NLOC),MAXLOC,PERSLA,2)
         IF (MSGSAV(NLOC).EQ.BLANK(1) .AND. IBLOC2.EQ.0) NLOC = NLOC +
     &       1
         IF (IBLOC .GT. 0) THEN
C                                  PAGE BREAK FOUND AT IBLOC
            LOCX = NLOC + IBLOC - 2
            WRITE (NOUT,99996) (MSGSAV(I),I=NLOC,LOCX)
            LOC = NLOC + IBLOC
            GO TO 20
C                                  DON'T BOTHER LOOKING FOR BLANK TO
C                                    BREAK AT IF LOCX .GE. MSGLEN
         ELSE IF (LOCX .LT. MSGLEN) THEN
C                                  CHECK FOR BLANK TO BREAK THE LINE
   30       CONTINUE
            IF (MSGSAV(LOCX) .EQ. BLANK(1)) THEN
C                                  BLANK FOUND AT LOCX
               WRITE (NOUT,99996) (MSGSAV(I),I=NLOC,LOCX)
               LOC = LOCX
               GO TO 20
            END IF
            LOCX = LOCX - 1
            IF (LOCX .GT. NLOC) GO TO 30
            LOCX = LOC + 64
C                                  NO BLANKS FOUND IN NEXT 64 CHARS
            WRITE (NOUT,99996) (MSGSAV(I),I=NLOC,LOCX)
            LOC = LOCX
            GO TO 20
         ELSE
C                                  ALL THE REST WILL FIT ON 1 LINE
            LOCX = MSGLEN
            WRITE (NOUT,99996) (MSGSAV(I),I=NLOC,LOCX)
         END IF
      END IF
C                                  SET LENGTH OF MSGSAV AND PLEN
C                                    TO SHOW THAT MESSAGE HAS
C                                    ALREADY BEEN PRINTED
 9000 MSGLEN = 0
      PLEN = 1
      IF (TRACON(LERTYP).EQ.1 .AND. CALLVL.GT.2) THEN
C                                  INITIATE TRACEBACK
         WRITE (NOUT,99997)
         DO 9005  J=CALLVL, 1, -1
            IF (J .GT. 1) THEN
               IF (ISUSER(J-1)) THEN
                  WRITE (NOUT,99998) RNAME(J), ERTYPE(J), ERCODE(J)
               ELSE
                  WRITE (NOUT,99999) RNAME(J), ERTYPE(J), ERCODE(J)
               END IF
            ELSE
               WRITE (NOUT,99998) RNAME(J), ERTYPE(J), ERCODE(J)
            END IF
 9005    CONTINUE
      END IF
C
      RETURN
99995 FORMAT (/, ' *** ', 8A1, ' ERROR', 59A1)
99996 FORMAT (' *** ', 9X, 64A1)
99997 FORMAT (14X, 'Here is a traceback of subprogram calls',
     &       ' in reverse order:', /, 14X, '      Routine    Error ',
     &       'type    Error code', /, 14X, '      -------    ',
     &       '----------    ----------')
99998 FORMAT (20X, A6, 5X, I6, 8X, I6)
99999 FORMAT (20X, A6, 5X, I6, 8X, I6, 4X, '(Called internally)')
      END
C-----------------------------------------------------------------------
C  IMSL Name:  N1RGB
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 2, 1984
C
C  Purpose:    Return a positive number as a flag to indicated that a
C              stop should occur due to one or more global errors.
C
C  Usage:      N1RGB(IDUMMY)
C
C  Arguments:
C     IDUMMY - Integer scalar dummy argument.
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION N1RGB (IDUMMY)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    IDUMMY
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  INITIALIZE FUNCTION
      N1RGB = 0
C                                  CHECK FOR GLOBAL ERROR TYPE 6
      IF (IFERR6 .GT. 0) THEN
         N1RGB = STOPTB(6)
         IFERR6 = 0
      END IF
C                                  CHECK FOR GLOBAL ERROR TYPE 7
      IF (IFERR7 .GT. 0) THEN
         N1RGB = STOPTB(7)
         IFERR7 = 0
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  U19NF/DU19NF  (Single/Double precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 16, 1985
C
C  Purpose:    Check validity of input to unconstrained minimization.
C
C  Usage:      CALL U19NF (ICODE)
C
C  Arguments:
C     ICODE  - Integer flag containing an error code.  (Input)
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
      SUBROUTINE U19NF (ICODE)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    ICODE
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1MES
C
      IF (ICODE .EQ. 0) THEN
         CALL E1MES (5, 1, 'The size of the problem must be '//
     &               'positive while N = %(I1) is given.')
      ELSE IF (ICODE .EQ. 1) THEN
         CALL E1MES (6, 1, 'This routine may be inefficient '//
     &               'for a problem of size N = 1.')
      ELSE IF (ICODE .EQ. 2) THEN
         CALL E1MES (6, 2, 'The diagonal scaling matrix for '//
     &               'the variables must be positive while some '//
     &               'of the entries are less than or equal to zero. '//
     &               ' The algorithm will use the identity scaling '//
     &               'matrix for XSCALE.')
      ELSE IF (ICODE .EQ. 4) THEN
         CALL E1MES (6, 4, 'The estimate of the number of '//
     &               'good digits in the function must be positive '//
     &               'while FDIGIT = %(I1) is given.  The algorithm '//
     &               'will assume that the function is accurate to '//
     &               'the precision of the arithmetic.')
      ELSE IF (ICODE .EQ. 5) THEN
         CALL E1MES (6, 5, 'The maximum number of iterations '//
     &               'must be positive while MXITER = %(I1) is '//
     &               'given.  The algorithm will use MXITER = 100.')
      ELSE IF (ICODE .EQ. 6) THEN
         CALL E1MES (6, 6, 'The maximum number of function '//
     &               'evaluations must be positive while MAXFCN = '//
     &               '%(I1) is given.  The algorithm will use '//
     &               'MAXFCN = 400.')
      ELSE IF (ICODE .EQ. 7) THEN
         CALL E1MES (6, 7, 'The maximum number of gradient '//
     &               'evaluations must be positive while MAXGRD = '//
     &               '%(I1) is given.  The algorithm will use '//
     &               'MAXGRD = 400.')
      ELSE IF (ICODE .EQ. 8) THEN
         CALL E1MES (6, 8, 'The maximum number of Hessian '//
     &               'evaluations must be positive while MAXHES = '//
     &               '%(I1) is given.  The algorithm will use '//
     &               'MAXHES = 100.')
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  IMACH (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 26, 1984
C
C  Purpose:    Retrieve integer machine constants.
C
C  Usage:      IMACH(N)
C
C  Arguments:
C     N      - Index of desired constant.  (Input)
C     IMACH  - Machine constant.  (Output)
C
C  Remark:
C     Following is a description of the assorted integer machine
C     constants.
C
C     Words
C
C        IMACH( 1) = Number of bits per integer storage unit.
C        IMACH( 2) = Number of characters per integer storage unit.
C
C     Integers
C
C        Assume integers are represented in the S-DIGIT, BASE-A form
C        SIGN ( X(S-1)*A**(S-1) + ... + X(1)*A + X(0) )
C        where 0 .LE. X(I) .LT. A for I=0,...,S-1.  Then
C
C        IMACH( 3) = A, the base.
C        IMACH( 4) = S, number of BASE-A digits.
C        IMACH( 5) = A**S - 1, largest magnitude.
C
C     Floating-point numbers
C
C        Assume floating-point numbers are represented in the T-DIGIT,
C        BASE-B form SIGN (B**E)*( (X(1)/B) + ... + (X(T)/B**T) )
C        where 0 .LE. X(I) .LT. B for I=1,...,T,
C        0 .LT. X(1), and EMIN .LE. E .LE. EMAX.  Then
C
C        IMACH( 6) = B, the base.
C
C        Single precision
C
C           IMACH( 7) = T, number of BASE-B digits.
C           IMACH( 8) = EMIN, smallest exponent E.
C           IMACH( 9) = EMAX, largest exponent E.
C
C        Double precision
C
C           IMACH(10) = T, number of BASE-B digits.
C           IMACH(11) = EMIN, smallest exponent E.
C           IMACH(12) = EMAX, largest exponent E.
C
C  GAMS:       R1
C
C  Chapters:   MATH/LIBRARY Reference Material
C              STAT/LIBRARY Reference Material
C              SFUN/LIBRARY Reference Material
C
C  Copyright:  1984 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION IMACH (N)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    NOUT
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    IMACHV(12)
      SAVE       IMACHV
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   UMACH
C                                  DEFINE CONSTANTS
      DATA IMACHV(1)/32/
      DATA IMACHV(2)/4/
      DATA IMACHV(3)/2/
      DATA IMACHV(4)/31/
      DATA IMACHV(5)/2147483647/
      DATA IMACHV(6)/2/
      DATA IMACHV(7)/24/
      DATA IMACHV(8)/-125/
      DATA IMACHV(9)/128/
      DATA IMACHV(10)/53/
      DATA IMACHV(11)/-1021/
      DATA IMACHV(12)/1024/
C
      IF (N.LT.1 .OR. N.GT.12) THEN
C                                  ERROR.  INVALID RANGE FOR N.
         CALL UMACH (2, NOUT)
         WRITE (NOUT,99999) N
99999    FORMAT (/, ' *** TERMINAL ERROR 5 from IMACH.  The argument',
     &          /, ' ***          must be between 1 and 12 inclusive.'
     &          , /, ' ***          N = ', I6, '.', /)
         IMACH = 0
         STOP
C
      ELSE
         IMACH = IMACHV(N)
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1INIT
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 13, 1984
C
C  Purpose:    Initialization.
C
C  Usage:      CALL E1INIT
C
C  Arguments:  None
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1INIT
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    L
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    ISINIT
      SAVE       ISINIT
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                              SPECIFICATIONS FOR COMMON /ERCOM8/
      INTEGER    PROLVL, XXLINE(10), XXPLEN(10), ICALOC(10), INALOC(10)
      COMMON     /ERCOM8/ PROLVL, XXLINE, XXPLEN, ICALOC, INALOC
      SAVE       /ERCOM8/
C                              SPECIFICATIONS FOR COMMON /ERCOM9/
      CHARACTER  XXPROC(10)*31
      COMMON     /ERCOM9/ XXPROC
      SAVE       /ERCOM9/
C
      DATA ISINIT/0/
C
      IF (ISINIT .EQ. 0) THEN
C                                  INITIALIZE
         CALLVL = 1
         ERCODE(1) = 0
         ERTYPE(1) = 0
         IALLOC(1) = 0
         ISUSER(1) = .TRUE.
         IFERR6 = 0
         IFERR7 = 0
         PLEN = 1
         MAXLEV = 50
         DO 10  L=2, 51
            ERTYPE(L) = -1
            ERCODE(L) = -1
            IALLOC(L) = 0
            ISUSER(L) = .FALSE.
   10    CONTINUE
         DO 20  L=1, 7
            HDRFMT(L) = 1
            TRACON(L) = 1
   20    CONTINUE
         PROLVL = 1
         DO 30  L=1, 10
   30    ICALOC(L) = 0
         XXLINE(1) = 0
         XXPLEN(1) = 1
         XXPROC(1) = '?'
         RNAME(1) = 'USER'
         PRINTB(1) = 0
         PRINTB(2) = 0
         DO 40  L=3, 7
   40    PRINTB(L) = 1
         STOPTB(1) = 0
         STOPTB(2) = 0
         STOPTB(3) = 0
         STOPTB(4) = 1
         STOPTB(5) = 1
         STOPTB(6) = 0
         STOPTB(7) = 1
         ERCKSM = 0.0D0
C                                  SET FLAG TO INDICATE THAT
C                                    INITIALIZATION HAS OCCURRED
         ISINIT = 1
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  E1POS
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 2, 1984
C
C  Purpose:    Set or retrieve print and stop attributes.
C
C  Usage:      CALL E1POS(IERTYP,IPATT,ISATT)
C
C  Arguments:
C     IERTYP - Integer specifying the error type for which print and
C              stop attributes are to be set or retrieved.  (Input)  If
C              IERTYP is 0 then the settings apply to all error types.
C              If IERTYP is between 1 and 7, then the settings only
C              apply to that specified error type.  If IERTYP is
C              negative then the current print and stop attributes will
C              be returned in IPATT and ISATT.
C     IPATT  - If IERTYP is positive, IPATT is an integer specifying the
C              desired print attribute as follows: -1 means no change,
C              0 means NO, 1 means YES, and 2 means assign the default
C              setting.  (Input)  If IERTYP is negative, IPATT is
C              returned as 1 if print is YES or 0 if print is NO for
C              error type IABS(IERTYP).  (Output)
C     ISATT  - If IERTYP is positive, ISATT is an integer specifying the
C              desired stop attribute as follows: -1 means no change,
C              0 means NO, 1 means YES, and 2 means assign the default
C              setting.  (Input)  If IERTYP is negative, ISATT is
C              returned as 1 if print is YES or 0 if print is NO for
C              error type IABS(IERTYP).  (Output)
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE E1POS (IERTYP, IPATT, ISATT)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    IERTYP, IPATT, ISATT
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IER
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    DEFLTP(7), DEFLTS(7), IFINIT
      SAVE       DEFLTP, DEFLTS, IFINIT
C                                  SPECIFICATIONS FOR SPECIAL CASES
C                              SPECIFICATIONS FOR COMMON /ERCOM1/
      INTEGER    CALLVL, MAXLEV, MSGLEN, ERTYPE(51), ERCODE(51),
     &           PRINTB(7), STOPTB(7), PLEN, IFERR6, IFERR7,
     &           IALLOC(51), HDRFMT(7), TRACON(7)
      COMMON     /ERCOM1/ CALLVL, MAXLEV, MSGLEN, ERTYPE, ERCODE,
     &           PRINTB, STOPTB, PLEN, IFERR6, IFERR7, IALLOC, HDRFMT,
     &           TRACON
      SAVE       /ERCOM1/
C                              SPECIFICATIONS FOR COMMON /ERCOM2/
      CHARACTER  MSGSAV(255), PLIST(300), RNAME(51)*6
      COMMON     /ERCOM2/ MSGSAV, PLIST, RNAME
      SAVE       /ERCOM2/
C                              SPECIFICATIONS FOR COMMON /ERCOM3/
      DOUBLE PRECISION ERCKSM
      COMMON     /ERCOM3/ ERCKSM
      SAVE       /ERCOM3/
C                              SPECIFICATIONS FOR COMMON /ERCOM4/
      LOGICAL    ISUSER(51)
      COMMON     /ERCOM4/ ISUSER
      SAVE       /ERCOM4/
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  IABS
      INTRINSIC  IABS
      INTEGER    IABS
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   E1INIT, E1MES, E1STI
C
      DATA IFINIT/0/
      DATA DEFLTP/0, 0, 1, 1, 1, 1, 1/, DEFLTS/0, 0, 0, 1, 1, 0, 1/
C                                  INITIALIZE ERROR TABLE IF NECESSARY
      IF (IFINIT .EQ. 0) THEN
         CALL E1INIT
         IFINIT = 1
      END IF
      IER = 0
      IF (IERTYP .GE. 0) THEN
         IF (IPATT.LT.-1 .OR. IPATT.GT.2) THEN
            CALL E1STI (1, IPATT)
            CALL E1MES (5, 1, 'Invalid value specified for print '//
     &                  'table attribute.  IPATT must be -1, 0, 1, '//
     &                  'or 2.  IPATT = %(I1)')
            IER = 1
         END IF
         IF (ISATT.LT.-1 .OR. ISATT.GT.2) THEN
            CALL E1STI (1, ISATT)
            CALL E1MES (5, 1, 'Invalid value specified for stop '//
     &                  'table attribute.  ISATT must be -1, 0, 1, '//
     &                  'or 2.  ISATT = %(I1)')
            IER = 1
         END IF
      END IF
      IF (IER .EQ. 0) THEN
         IF (IERTYP .EQ. 0) THEN
            IF (IPATT.EQ.0 .OR. IPATT.EQ.1) THEN
               DO 10  I=1, 7
   10          PRINTB(I) = IPATT
            ELSE IF (IPATT .EQ. 2) THEN
C                                  ASSIGN DEFAULT SETTINGS
               DO 20  I=1, 7
   20          PRINTB(I) = DEFLTP(I)
            END IF
            IF (ISATT.EQ.0 .OR. ISATT.EQ.1) THEN
               DO 30  I=1, 7
   30          STOPTB(I) = ISATT
            ELSE IF (ISATT .EQ. 2) THEN
C                                  ASSIGN DEFAULT SETTINGS
               DO 40  I=1, 7
   40          STOPTB(I) = DEFLTS(I)
            END IF
         ELSE IF (IERTYP.GE.1 .AND. IERTYP.LE.7) THEN
            IF (IPATT.EQ.0 .OR. IPATT.EQ.1) THEN
               PRINTB(IERTYP) = IPATT
            ELSE IF (IPATT .EQ. 2) THEN
C                                  ASSIGN DEFAULT SETTING
               PRINTB(IERTYP) = DEFLTP(IERTYP)
            END IF
            IF (ISATT.EQ.0 .OR. ISATT.EQ.1) THEN
               STOPTB(IERTYP) = ISATT
            ELSE IF (ISATT .EQ. 2) THEN
C                                  ASSIGN DEFAULT SETTING
               STOPTB(IERTYP) = DEFLTS(IERTYP)
            END IF
         ELSE IF (IERTYP.LE.-1 .AND. IERTYP.GE.-7) THEN
            I = IABS(IERTYP)
            IPATT = PRINTB(I)
            ISATT = STOPTB(I)
         END IF
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  C1TIC
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 9, 1984
C
C  Purpose:    Convert an integer to its corresponding character form.
C              (Right justified)
C
C  Usage:      CALL C1TIC(NUM, CHRSTR, SLEN, IER)
C
C  Arguments:
C     NUM    - Integer number.  (Input)
C     CHRSTR - Character array that receives the result.  (Output)
C     SLEN   - Length of the character array.  (Input)
C     IER    - Completion code.  (Output) Where
C                 IER < 0  indicates that SLEN <= 0,
C                 IER = 0  indicates normal completion,
C                 IER > 0  indicates that the character array is too
C                       small to hold the complete number.  IER
C                       indicates how many significant digits are
C                       being truncated.
C
C  Remarks:
C  1. The character array is filled in a right justified manner.
C  2. Leading zeros are replaced by blanks.
C  3. Sign is inserted only for negative number.
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE C1TIC (NUM, CHRSTR, SLEN, IER)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    NUM, SLEN, IER
      CHARACTER  CHRSTR(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, J, K, L
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      CHARACTER  BLANK(1), DIGIT(10), MINUS(1)
      SAVE       BLANK, DIGIT, MINUS
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  IABS
      INTRINSIC  IABS
      INTEGER    IABS
C                                  SPECIFICATIONS FOR SUBROUTINES
      EXTERNAL   M1VE
C
      DATA DIGIT/'0', '1', '2', '3', '4', '5', '6', '7', '8',
     &     '9'/
      DATA BLANK/' '/, MINUS/'-'/
C                                  CHECK SLEN
      IF (SLEN .LE. 0) THEN
         IER = -1
         RETURN
      END IF
C                                  THE NUMBER IS ZERO
      IF (NUM .EQ. 0) THEN
         CALL M1VE (BLANK, 1, 1, 1, CHRSTR, 1, SLEN-1, SLEN, I)
         CHRSTR(SLEN) = DIGIT(1)
         IER = 0
         RETURN
      END IF
C                                  CONVERT NUMBER DIGIT BY DIGIT TO
C                                  CHARACTER FORM
      J = SLEN
      K = IABS(NUM)
   10 IF (K.GT.0 .AND. J.GE.1) THEN
         L = K
         K = K/10
         L = L - K*10
         CHRSTR(J) = DIGIT(L+1)
         J = J - 1
         GO TO 10
      END IF
C
   20 IF (K .EQ. 0) THEN
         IF (NUM .LT. 0) THEN
            CALL M1VE (MINUS, 1, 1, 1, CHRSTR, J, J, SLEN, I)
            IF (I .NE. 0) THEN
               IER = 1
               RETURN
            END IF
            J = J - 1
         END IF
         IER = 0
         CALL M1VE (BLANK, 1, 1, 1, CHRSTR, 1, J, SLEN, I)
         RETURN
      END IF
C                                  DETERMINE THE NUMBER OF SIGNIFICANT
C                                  DIGITS BEING TRUNCATED
      I = 0
   30 IF (K .GT. 0) THEN
         K = K/10
         I = I + 1
         GO TO 30
      END IF
C
      IF (NUM .LT. 0) I = I + 1
      IER = I
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  ISAMAX (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 9, 1986
C
C  Purpose:    Find the smallest index of the component of a
C              single-precision vector having maximum absolute value.
C
C  Usage:      ISAMAX(N, SX, INCX)
C
C  Arguments:
C     N      - Length of vector X.  (Input)
C     SX     - Real vector of length N*INCX.  (Input)
C     INCX   - Displacement between elements of SX.  (Input)
C              X(I) is defined to be SX(1+(I-1)*INCX). INCX must be
C              greater than zero.
C     ISAMAX - The smallest index I such that ABS(X(I)) is the maximum
C              of ABS(X(J)) for J=1 to N.  (Output)
C              X(I) refers to a specific element of SX. see INCX
C              argument description.
C
C  GAMS:       D1a2
C
C  Chapters:   MATH/LIBRARY Basic Matrix/Vector Operations
C              STAT/LIBRARY Mathematical Support
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION ISAMAX (N, SX, INCX)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, INCX
      REAL       SX(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, II, NS
      REAL       SMAX, XMAG
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ABS
      INTRINSIC  ABS
      REAL       ABS
C
      ISAMAX = 0
      IF (N .GE. 1) THEN
         ISAMAX = 1
         IF (N .GT. 1) THEN
            IF (INCX .NE. 1) THEN
C                                  CODE FOR INCREMENTS NOT EQUAL TO 1.
               SMAX = ABS(SX(1))
               NS = N*INCX
               II = 1
               DO 10  I=1, NS, INCX
                  XMAG = ABS(SX(I))
                  IF (XMAG .GT. SMAX) THEN
                     ISAMAX = II
                     SMAX = XMAG
                  END IF
                  II = II + 1
   10          CONTINUE
            ELSE
C                                  CODE FOR INCREMENTS EQUAL TO 1.
               SMAX = ABS(SX(1))
               DO 20  I=2, N
                  XMAG = ABS(SX(I))
                  IF (XMAG .GT. SMAX) THEN
                     ISAMAX = I
                     SMAX = XMAG
                  END IF
   20          CONTINUE
            END IF
         END IF
      END IF
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  S1ANUM
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 28, 1984
C
C  Purpose:    Scan a token and identify it as follows: integer, real
C              number (single/double), FORTRAN relational operator,
C              FORTRAN logical operator, or FORTRAN logical constant.
C
C  Usage:      CALL S1ANUM(INSTR, SLEN, CODE, OLEN)
C
C  Arguments:
C     INSTR  - Character string to be scanned.  (Input)
C     SLEN   - Length of INSTR.  (Input)
C     CODE   - Token code.  (Output)  Where
C                 CODE =  0  indicates an unknown token,
C                 CODE =  1  indicates an integer number,
C                 CODE =  2  indicates a (single precision) real number,
C                 CODE =  3  indicates a (double precision) real number,
C                 CODE =  4  indicates a logical constant (.TRUE. or
C                               .FALSE.),
C                 CODE =  5  indicates the relational operator .EQ.,
C                 CODE =  6  indicates the relational operator .NE.,
C                 CODE =  7  indicates the relational operator .LT.,
C                 CODE =  8  indicates the relational operator .LE.,
C                 CODE =  9  indicates the relational operator .GT.,
C                 CODE = 10  indicates the relational operator .GE.,
C                 CODE = 11  indicates the logical operator .AND.,
C                 CODE = 12  indicates the logical operator .OR.,
C                 CODE = 13  indicates the logical operator .EQV.,
C                 CODE = 14  indicates the logical operator .NEQV.,
C                 CODE = 15  indicates the logical operator .NOT..
C     OLEN   - Length of the token as counted from the first character
C              in INSTR.  (Output)  OLEN returns a zero for an unknown
C              token (CODE = 0).
C
C  Remarks:
C  1. Blanks are considered significant.
C  2. Lower and upper case letters are not significant.
C
C  Copyright:  1984 by IMSL, Inc.  All rights reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE S1ANUM (INSTR, SLEN, CODE, OLEN)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    SLEN, CODE, OLEN
      CHARACTER  INSTR(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, IBEG, IIBEG, J
      LOGICAL    FLAG
      CHARACTER  CHRSTR(6)
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    TABPTR(16), TDCNST, TICNST, TOKEN(13), TRCNST, TZERR
      CHARACTER  DIGIT(10), LETTER(52), MINUS, PERIOD, PLUS, TABLE(38)
      SAVE       DIGIT, LETTER, MINUS, PERIOD, PLUS, TABLE, TABPTR,
     &           TDCNST, TICNST, TOKEN, TRCNST, TZERR
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   I1X, I1CSTR
      INTEGER    I1X, I1CSTR
C
      DATA TOKEN/5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 4, 4/
      DATA TABLE/'D', 'E', 'E', 'Q', 'N', 'E', 'L', 'T', 'L',
     &     'E', 'G', 'T', 'G', 'E', 'A', 'N', 'D', 'O', 'R',
     &     'E', 'Q', 'V', 'N', 'E', 'Q', 'V', 'N', 'O', 'T',
     &     'T', 'R', 'U', 'E', 'F', 'A', 'L', 'S', 'E'/
      DATA TABPTR/1, 2, 3, 5, 7, 9, 11, 13, 15, 18, 20, 23, 27, 30,
     &     34, 39/
      DATA DIGIT/'0', '1', '2', '3', '4', '5', '6', '7', '8',
     &     '9'/
      DATA LETTER/'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
     &     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
     &     'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
     &     'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
     &     'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
     &     'x', 'y', 'z'/
      DATA PERIOD/'.'/, PLUS/'+'/, MINUS/'-'/
      DATA TZERR/0/, TICNST/1/
      DATA TRCNST/2/, TDCNST/3/
C
      IF (SLEN .LE. 0) THEN
         CODE = 0
         OLEN = 0
         RETURN
      END IF
C                                  STATE 0 - ASSUME ERROR TOKEN
      IBEG = 1
      CODE = TZERR
C                                  CHECK SIGN
      IF (INSTR(IBEG).EQ.MINUS .OR. INSTR(IBEG).EQ.PLUS) THEN
         FLAG = .TRUE.
         IIBEG = IBEG
         IBEG = IBEG + 1
      ELSE
         FLAG = .FALSE.
      END IF
C                                  STATE 1 - ASSUME INTEGER CONSTANT
      IF (I1X(DIGIT,10,INSTR(IBEG),1) .NE. 0) THEN
         CODE = TICNST
         IIBEG = IBEG
         IBEG = IBEG + 1
C
   10    IF (IBEG .LE. SLEN) THEN
C
            IF (I1X(DIGIT,10,INSTR(IBEG),1) .NE. 0) THEN
               IIBEG = IBEG
               IBEG = IBEG + 1
               GO TO 10
C
            END IF
C
         ELSE
            GO TO 80
C
         END IF
C
         IF (INSTR(IBEG) .NE. PERIOD) GO TO 80
      END IF
C                                  STATE 2 - ASSUME REAL CONSTANT
      IF (CODE .EQ. TICNST) THEN
         CODE = TRCNST
         IIBEG = IBEG
         IBEG = IBEG + 1
         IF (IBEG .GT. SLEN) GO TO 80
      ELSE IF (INSTR(IBEG).EQ.PERIOD .AND. SLEN.GE.2) THEN
         IF (I1X(DIGIT,10,INSTR(IBEG+1),1) .NE. 0) THEN
            CODE = TRCNST
            IIBEG = IBEG + 1
            IBEG = IBEG + 2
            IF (IBEG .GT. SLEN) GO TO 80
         END IF
      END IF
C
      IF (I1X(DIGIT,10,INSTR(IBEG),1) .NE. 0) THEN
         CODE = TRCNST
         IIBEG = IBEG
         IBEG = IBEG + 1
C
   20    IF (IBEG .LE. SLEN) THEN
C
            IF (I1X(DIGIT,10,INSTR(IBEG),1) .NE. 0) THEN
               IIBEG = IBEG
               IBEG = IBEG + 1
               GO TO 20
C
            END IF
C
         ELSE
            GO TO 80
C
         END IF
C
      END IF
C
      IF (CODE .EQ. TZERR) THEN
         IF (INSTR(IBEG) .NE. PERIOD) GO TO 80
         IBEG = IBEG + 1
         IF (IBEG .GT. SLEN) GO TO 80
      END IF
C
      IF (I1X(LETTER,52,INSTR(IBEG),1) .EQ. 0) GO TO 80
      CHRSTR(1) = INSTR(IBEG)
C
      DO 30  I=2, 6
         IBEG = IBEG + 1
         IF (IBEG .GT. SLEN) GO TO 80
         IF (I1X(LETTER,52,INSTR(IBEG),1) .EQ. 0) GO TO 40
         CHRSTR(I) = INSTR(IBEG)
   30 CONTINUE
C
      GO TO 80
C
   40 CONTINUE
C
      DO 50  J=1, 15
         IF (I1CSTR(CHRSTR,I-1,TABLE(TABPTR(J)),TABPTR(J+1)-TABPTR(J))
     &        .EQ. 0) GO TO 60
   50 CONTINUE
C
      GO TO 80
C                                  STATE 4 - LOGICAL OPERATOR
   60 IF (J .GT. 2) THEN
C
         IF (CODE .EQ. TRCNST) THEN
C
            IF (INSTR(IBEG) .EQ. PERIOD) THEN
               CODE = TICNST
               IIBEG = IIBEG - 1
            END IF
C
            GO TO 80
C
         ELSE IF (INSTR(IBEG) .NE. PERIOD) THEN
            GO TO 80
C
         ELSE IF (FLAG) THEN
            GO TO 80
C
         ELSE
            CODE = TOKEN(J-2)
            IIBEG = IBEG
            GO TO 80
C
         END IF
C
      END IF
C                                  STATE 5 - DOUBLE PRECISION CONSTANT
      IF (CODE .NE. TRCNST) GO TO 80
      IF (INSTR(IBEG).EQ.MINUS .OR. INSTR(IBEG).EQ.PLUS) IBEG = IBEG +
     &    1
      IF (IBEG .GT. SLEN) GO TO 80
C
      IF (I1X(DIGIT,10,INSTR(IBEG),1) .EQ. 0) THEN
         GO TO 80
C
      ELSE
         IIBEG = IBEG
         IBEG = IBEG + 1
C
   70    IF (IBEG .LE. SLEN) THEN
C
            IF (I1X(DIGIT,10,INSTR(IBEG),1) .NE. 0) THEN
               IIBEG = IBEG
               IBEG = IBEG + 1
               GO TO 70
C
            END IF
C
         END IF
C
      END IF
C
      IF (J .EQ. 1) CODE = TDCNST
C
   80 CONTINUE
C
      IF (CODE .EQ. TZERR) THEN
         OLEN = 0
C
      ELSE
         OLEN = IIBEG
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  ICASE (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 9, 1985
C
C  Purpose:    Convert from character to the integer ASCII value without
C              regard to case.
C
C  Usage:      ICASE(CH)
C
C  Arguments:
C     CH     - Character to be converted.  (Input)
C     ICASE  - Integer ASCII value for CH without regard to the case
C              of CH.  (Output)
C              ICASE returns the same value as IMSL routine IACHAR for
C              all but lowercase letters.  For these, it returns the
C              IACHAR value for the corresponding uppercase letter.
C
C  GAMS:       N3
C
C  Chapter:    MATH/LIBRARY Utilities
C              STAT/LIBRARY Utilities
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION ICASE (CH)
C                                  SPECIFICATIONS FOR ARGUMENTS
      CHARACTER  CH
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   IACHAR
      INTEGER    IACHAR
C
      ICASE = IACHAR(CH)
      IF (ICASE.GE.97 .AND. ICASE.LE.122) ICASE = ICASE - 32
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  I1CSTR (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 10, 1985
C
C  Purpose:    Case insensitive comparison of two character arrays.
C
C  Usage:      I1CSTR(STR1, LEN1, STR2, LEN2)
C
C  Arguments:
C     STR1   - First character array.  (Input)
C     LEN1   - Length of STR1.  (Input)
C     STR2   - Second character array.  (Input)
C     LEN2   - Length of STR2.  (Input)
C     I1CSTR - Integer function.  (Output) Where
C              I1CSTR = -1  if STR1 .LT. STR2,
C              I1CSTR =  0  if STR1 .EQ. STR2,
C              I1CSTR =  1  if STR1 .GT. STR2.
C
C  Remarks:
C  1. If the two arrays, STR1 and STR2,  are of unequal length, the
C     shorter array is considered as if it were extended with blanks
C     to the length of the longer array.
C
C  2. If one or both lengths are zero or negative the I1CSTR output is
C     based on comparison of the lengths.
C
C  GAMS:       N5c
C
C  Chapter:    MATH/LIBRARY Utilities
C
C  Copyright:  1985 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION I1CSTR (STR1, LEN1, STR2, LEN2)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    LEN1, LEN2
      CHARACTER  STR1(LEN1), STR2(LEN2)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    IC1, IC2, ICB, IS, L, LENM
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  ISIGN,MIN0
      INTRINSIC  ISIGN, MIN0
      INTEGER    ISIGN, MIN0
C                                  SPECIFICATIONS FOR FUNCTIONS
      EXTERNAL   ICASE
      INTEGER    ICASE
C
      IF (LEN1.GT.0 .AND. LEN2.GT.0) THEN
C                                  COMPARE FIRST LENM CHARACTERS
         LENM = MIN0(LEN1,LEN2)
         DO 10  L=1, LENM
            IC1 = ICASE(STR1(L))
            IC2 = ICASE(STR2(L))
            IF (IC1 .NE. IC2) THEN
               I1CSTR = ISIGN(1,IC1-IC2)
               RETURN
            END IF
   10    CONTINUE
      END IF
C                                  COMPARISON BASED ON LENGTH OR
C                                  TRAILING BLANKS
      IS = LEN1 - LEN2
      IF (IS .EQ. 0) THEN
         I1CSTR = 0
      ELSE
         IF (LEN1.LE.0 .OR. LEN2.LE.0) THEN
C                                  COMPARISON BASED ON LENGTH
            I1CSTR = ISIGN(1,IS)
         ELSE
C                                  COMPARISON BASED ON TRAILING BLANKS
C                                  TO EXTEND SHORTER ARRAY
            LENM = LENM + 1
            ICB = ICASE(' ')
            IF (IS .GT. 0) THEN
C                                  EXTEND STR2 WITH BLANKS
               DO 20  L=LENM, LEN1
                  IC1 = ICASE(STR1(L))
                  IF (IC1 .NE. ICB) THEN
                     I1CSTR = ISIGN(1,IC1-ICB)
                     RETURN
                  END IF
   20          CONTINUE
            ELSE
C                                  EXTEND STR1 WITH BLANKS
               DO 30  L=LENM, LEN2
                  IC2 = ICASE(STR2(L))
                  IF (ICB .NE. IC2) THEN
                     I1CSTR = ISIGN(1,ICB-IC2)
                     RETURN
                  END IF
   30          CONTINUE
            END IF
C
            I1CSTR = 0
         END IF
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  UMACH (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    March 21, 1984
C
C  Purpose:    Set or retrieve input or output device unit numbers.
C
C  Usage:      CALL UMACH (N, NUNIT)
C
C  Arguments:
C     N      - Index of desired unit.  (Input)
C              The values of N are defined as follows:
C              N = 1, corresponds to the standard input unit.
C              N = 2, corresponds to the standard output unit.
C     NUNIT  - I/O unit.  (Input or Output)
C              If the value of N is negative, the unit corresponding
C              to the index is reset to the value given in NUNIT.
C              Otherwise, the value corresponding to the index is
C              returned in NUNIT.
C
C  GAMS:       R1
C
C  Chapters:   MATH/LIBRARY Reference Material
C              STAT/LIBRARY Reference Material
C              SFUN/LIBRARY Reference Material
C
C  Copyright:  1984 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      SUBROUTINE UMACH (N, NUNIT)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    N, NUNIT
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    NN, NOUT
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      INTEGER    UNIT(2)
      SAVE       UNIT
C                                  SPECIFICATIONS FOR INTRINSICS
C     INTRINSIC  IABS
      INTRINSIC  IABS
      INTEGER    IABS
C
      DATA UNIT(1)/5/
      DATA UNIT(2)/6/
C
      NN = IABS(N)
      IF (NN.NE.1 .AND. NN.NE.2) THEN
C                                  ERROR.  INVALID RANGE FOR N.
         NOUT = UNIT(2)
         WRITE (NOUT,99999) NN
99999    FORMAT (/, ' *** TERMINAL ERROR 5 from UMACH.  The absolute',
     &          /, ' ***          value of the index variable must be'
     &          , /, ' ***          1 or 2.  IABS(N) = ', I6,
     &          '.', /)
         STOP
C                                  CHECK FOR RESET OR RETRIEVAL
      ELSE IF (N .LT. 0) THEN
C                                  RESET
         UNIT(NN) = NUNIT
      ELSE
C                                  RETRIEVE
         NUNIT = UNIT(N)
      END IF
C
      RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  I1X (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    August 30, 1985
C
C  Purpose:    Determine the array subscript indicating the starting
C              element at which a key character sequence begins.
C              (Case-sensitive version)
C
C  Usage:      I1X(CHRSTR, I1LEN, KEY, KLEN)
C
C  Arguments:
C     CHRSTR - Character array to be searched.  (Input)
C     I1LEN  - Length of CHRSTR.  (Input)
C     KEY    - Character array that contains the key sequence.  (Input)
C     KLEN   - Length of KEY.  (Input)
C     I1X    - Integer function.  (Output)
C
C  Remarks:
C  1. Returns zero when there is no match.
C
C  2. Returns zero if KLEN is longer than ISLEN.
C
C  3. Returns zero when any of the character arrays has a negative or
C     zero length.
C
C  GAMS:       N5c
C
C  Chapter:    MATH/LIBRARY Utilities
C
C  Copyright:  1985 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION I1X (CHRSTR, I1LEN, KEY, KLEN)
C                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER    I1LEN, KLEN
      CHARACTER  CHRSTR(*), KEY(*)
C                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER    I, II, J
C
      I1X = 0
      IF (KLEN.LE.0 .OR. I1LEN.LE.0) GO TO 9000
      IF (KLEN .GT. I1LEN) GO TO 9000
C
      I = 1
      II = I1LEN - KLEN + 1
   10 IF (I .LE. II) THEN
         IF (CHRSTR(I) .EQ. KEY(1)) THEN
            DO 20  J=2, KLEN
               IF (CHRSTR(I+J-1) .NE. KEY(J)) GO TO 30
   20       CONTINUE
            I1X = I
            GO TO 9000
   30       CONTINUE
         END IF
         I = I + 1
         GO TO 10
      END IF
C
 9000 RETURN
      END
C-----------------------------------------------------------------------
C  IMSL Name:  IACHAR (Single precision version)
C
C  Computer:   SUN4/SINGLE
C
C  Revised:    September 9, 1985
C
C  Purpose:    Return the integer ASCII value of a character argument.
C
C  Usage:      IACHAR(CH)
C
C  Arguments:
C     CH     - Character argument for which the integer ASCII value
C              is desired.  (Input)
C     IACHAR - Integer ASCII value for CH.  (Output)
C              The character CH is in the IACHAR-th position of the
C              ASCII collating sequence.
C
C  Keywords:   Utilities; Character string manipulation;
C              Character conversion
C
C  GAMS:       N3
C
C  Chapter:    MATH/LIBRARY Utilities
C              STAT/LIBRARY Utilities
C
C  Copyright:  1986 by IMSL, Inc.  All Rights Reserved.
C
C  Warranty:   IMSL warrants only that IMSL testing has been applied
C              to this code.  No other warranty, expressed or implied,
C              is applicable.
C
C-----------------------------------------------------------------------
C
      INTEGER FUNCTION IACHAR (CH)
C                                  SPECIFICATIONS FOR ARGUMENTS
      CHARACTER  CH
C                                  SPECIFICATIONS FOR SAVE VARIABLES
      IACHAR = ICHAR(CH)
      RETURN
      END
