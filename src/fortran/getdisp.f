
c - this program gets the maximum likelihood estimate of the
c   dispersion, including the uncertainties

	parameter(nmax=10000)
	real v(nmax),ev(nmax)
	character file1*50

 101	write(*,"('Data : '$)")
	read *,file1
	OPEN(UNIT=1,file=file1,STATUS='OLD',ERR=101)

C	-- READ IN THE DATA
	N=0
1	CONTINUE
	N=N+1
2	READ (1,*,END=10) v(n),ev(n)
	GOTO 1

10	CONTINUE
	CLOSE (UNIT=1)
	N=N-1
	call dispmaxl(n,v,ev,vm0,vm0unc,sig0,sig0unc,covar0,ier)
	write (6,108) n,vm0,vm0unc,sig0,sig0unc,covar0,ier
108	format(1x,i4,' stars: <v> = ',f7.2,'+-',f5.2,'  sig0 = ',f6.2,
&	'+-',f5.2/13x,'covar = ',f7.5,' ier = ',i2)

	end
