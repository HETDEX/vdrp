c========================================================================
c	Adaptive kernal contour plotting.
c
	subroutine adker1(xdata,n,itype,igin,hin,ain,gin,ip,ek,
     $     xmin,xmax,x,xl)
c------------------------------------------------------------------------
c	usual parameter values: itype=1  (adaptative kernel)
c	(selected if =0)        ik=2     (kernel type)
c                               ain=.5   (exponent)
c
c	x,y: x and y positions for input points
c	n: number of points
c	itype: 0 - non-adaptive kernal; 1 - adaptive kernal
c	ik: kernal type (if zero, program uses 2)
c	hin: kernal width (in units of bins); 0 chooses default
c	ain: smoothing parameter, 0 - program chooses 0.5
c	g: the "average" density used to calculate lambdas (0 uses default)
c
c------------------------------------------------------------------------
	parameter(nmax=30000, nsort=500,ngmax=100)
	parameter (zero=0.e0, pi=3.141592e0, eps=1e-20)

	real x(nmax),ek(igin),clam(nmax),hnew(nmax)
	real xdata(n),xin(nsort)
	real xl(ngmax),ynew(nmax)

	data ig /50/
	data big /1.e10/

	igg=igin
	if(igin.eq.0) igg=ig
	g=gin

c	-- find the min and max

        if(xmin.eq.xmax) then
           xmax=-big
           xmin=big
           do i=1,n
              xmax=max(xmax,xdata(i))
              xmin=min(xmin,xdata(i))
           enddo
        endif

c	-- Decide on what size grid to use

        igx=igg
	if(ip.eq.1) write (6,*) 'Using a grid of ',igx
	jxl=1
	jxh=igx

c	-- Scale data to fit on igx X igy grid
c	   For consistency with what comes below, assume that the bin
c	   centers are at x,y = 1, 2, 3, ..., igx,y.  Then the grid
c	   extends from x,y = 0.5 to igx,y + 0.5.

c	tim=dtime(time)
	scalx=float(igx)/(xmax-xmin)
	do i=1,n
	   x(i)=scalx*(xdata(i)-xmin)+0.50
	   if(x(i).le.0.50) x(i)=0.5001
	   if(x(i).ge.(igx+0.50)) x(i)=igx+0.4990
	enddo
c	tim = dtime(time)
	if(ip.eq.1) write (6,*)
	if(ip.eq.1) write (6,*) ' Used particles     : ',n
	if(ip.eq.1) write (6,*) ' xmin,xmax = ',xmin,xmax
c	write (6,*) ' Fitting particles  : ',tim,' sec'	

c	-- Obtain optimal window width approximation     

	if (hin.eq.zero) then
	   if(n.lt.nsort) then
	      do i=1,n
	         xin(i) = x(i)
	      enddo
	      call biwgt(xin,n,xlbiwt,xsbiwt)
	      s = xsbiwt
	   else
	      sumx=zero
	      sumx2=zero
	      do i=1,n
	         sumx=sumx+x(i)
	         sumx2=sumx2+x(i)*x(i)
	      enddo
	      varx=(sumx2-sumx*sumx/float(n))/float(n-1)
	      s=sqrt(varx)
	   endif
	   h = s*0.75*1.77*n**(-1./6.)
	else
	   h = hin
	endif
        h2=h*h
	
c	tim = dtime(time)
c	write (6,*) ' Optimal wind. width: ',tim,' sec'
	if(ip.eq.1) write (6,*) ' Estimated h	  : ',h
	
c	-- Initialize density matrix

        do j = jxl,jxh
           ek(j) = zero
        enddo

c	-- Obtain contribution to density from each data point considered
c	   one at a time

c	tim=dtime(time)
	do i= 1,n	   

c	   -- First find grid limits over which a data point can contribute

	   jlo = nint(x(i)-h)
	   jhi = nint(x(i)+h)
	   jlo = max(jxl,jlo)
	   jhi = min(jxh,jhi)

c	  -- Now loop through only those indices to which data can contribute  

           xp=x(i)/h
           do j = jlo,jhi
              x1=float(j)-0.5
              x2=x1+1.
              if(abs(x(i)-x1).gt.h) x1=x(i)-h
              if(abs(x(i)-x2).gt.h) x2=x(i)+h
              x1=x1/h
              x2=x2/h
              xint=x2-x1+((xp-x2)**3-(xp-x1)**3)/3.
              ek(j)=ek(j)+xint
           enddo
	enddo

c	con = 3./(4.*h)
	con = 3./(4.*float(n))

c	-- Set minimum density to one point per window

	if (itype.eq.0) then

c	-- Now multiply by required constants to arrive at grid of density 
c	   estimates. Also find maximum and minimum density.  

	   bmaxa=-big
	   bmina=big
           do j = jxl,jxh
              ek(j) = con*ek(j)*scalx
              xl(j)=(float(j)-0.5)/scalx+xmin
              bmaxa=max(bmaxa,ek(j))
              bmina=min(bmina,ek(j))
           enddo
	   if(ip.eq.1) write (6,*) ' min and max grid values = ',
     $          bmina,bmaxa

	else

c	   -- Now for the adaptive kernel calculation
c	   -- Find the geometric mean of the run of densities in the grid in the
c	      regions  nearest each data point

	   if (g.eq.0.) then
	      sum=0
	      do i=1,n
		 ix = nint(x(i))
		 sum = sum + log10(ek(ix)+eps)
	      enddo
	      g = 10**(sum/float(n))
	      if(ip.eq.1) write (6,*) ' g = ',g*con
	   else
	      if(ip.eq.1) write (6,*) ' g = ',g
	      g=g/con
	   end if

c	   -- Find the local bandwidth factors     

	   if(ain.eq.zero) then
	      alpha = 0.5
	   else
	      alpha = ain
	   endif

	   ixmax=0
	   ixmin=0
	   clammax=-big
	   clammin=big
	   do i=1,n
	      ix = nint(x(i))
	      clam(i) = (ek(ix)/g)**(-alpha)
              hnew(i)=h*clam(i)
	      if (clam(i).gt.clammax) then
		 ixmax=ix
		 clammax=clam(i)
	      end if
	      if (clam(i).lt.clammin) then
		 ixmin=ix
		 clammin=clam(i)
	      end if
	   enddo
	   if(ip.eq.1) write (6,*) ' minimum lambda = ',
     $          clammin,' at ',ixmin
	   if(ip.eq.1) write (6,*) ' maximum lambda = ',
     $          clammax,' at ',ixmax

c	   tim = dtime(time)
c	   print *, ' Estimation of g-lam: ',tim,' sec'

           do j = jxl,jxh
              ek(j) = zero
           enddo

c	   -- Now form the adaptive kernel estimate 

c	   tim=dtime(time)
	   do i=1,n

c	      -- First find grid limits over which a data point can contribute

	      jlo = nint(x(i)-hnew(i))
	      jhi = nint(x(i)+hnew(i))
	      jlo = max(jxl,jlo)
	      jhi = min(jxh,jhi)

c	      -- Now loop through only those indices to which data
c	         can contribute 

              xp=x(i)/hnew(i)
              do j = jlo,jhi
                 x1=float(j)-0.5
                 x2=x1+1.
                 if(abs(x(i)-x1).gt.hnew(i)) x1=x(i)-hnew(i)
                 if(abs(x(i)-x2).gt.hnew(i)) x2=x(i)+hnew(i)
                 x1=x1/hnew(i)
                 x2=x2/hnew(i)
                 xint=x2-x1+((xp-x2)**3-(xp-x1)**3)/3.
                 ek(j)=ek(j)+xint
              enddo
	   enddo

c	   -- Now multiply by required constants to arrive at grid of density 
c	      estimates. Also find maximum density     

	   bmaxa=-big
	   bmina=big

c           open(unit=2,file='adker1.out',status='unknown')

           do j = jxl,jxh
              ek(j) = con*ek(j)*scalx
c              xl(j)=float(j)/scalx+xmin
              xl(j)=(float(j)-0.5)/scalx+xmin
c              write(2,*) xl(j),ek(j)
              bmaxa=max(bmaxa,ek(j))
              bmina=min(bmina,ek(j))
           enddo
c           close(2)
	   if(ip.eq.1) write (6,*) ' min and max grid values = ',
     $          bmina,bmaxa

	endif

        ymax=bmaxa+(bmaxa+bmina)/20.
        if(ip.eq.1) print *,'ymax = ',ymax

c	tim = dtime(time)
c	write (6,*) ' Grid calculation   : ',tim,' sec'

        if(ip.eq.1) then
           call pgscf(2)
           call pgslw(2)
           call pgenv(xmin,xmax,0.,ymax*1.25,0,0)
           call pgline(igg,xl,ek)
c        call pghist(n,xdata,xmin,xmax,nbin,1)
        endif

c	eta=0.5
c	do i=1,igg
c	   ynew(i)=eta*(10**xl(i))**eta/(1.+10**xl(i))**(eta+1.)/.4343
c	enddo
c	call pgsls(2)
c	call pgline(igg,xl,ynew)
c	call pgsls(1)

	return
	end
