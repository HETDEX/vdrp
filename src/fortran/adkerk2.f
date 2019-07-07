
	subroutine adkerk2(x,n,igin,h,ans)

	real x(n)

	data ig /50/
	data big /1.e10/

	igg=igin
	if(igin.eq.0) igg=ig

c	-- Decide on what size grid to use

        igx=igg
	jxl=1
	jxh=igx

        h2=h*h
	
c	-- Initialize density matrix

        val1=0.

c	-- Obtain contribution to density from each data point considered
c	   one at a time

        do k=1,n

	   klo = nint(x(k)-h)
	   khi = int(x(k)+h)
	   klo = max(jxl,klo)
	   khi = min(jxh,khi)

           do i=1,n

c	  -- Now loop through only those indices to which data can contribute  

              do j=klo,khi
                 rr1=(x(i)-j)**2/h2
                 rr2=(x(k)-j)**2/h2
                 if(rr1.lt.1..and.rr2.lt.1.) val1=val1+(1.-rr1)*(1.-rr2)
              enddo
           enddo
	enddo

	con = 9./(16.*h2*float(n*n))

        val1=val1*con

        val2=0.

        do k=1,n
           do i=1,n
              if (i.ne.k) then
                 rr=(x(i)-x(k))**2/h2
                 if(rr.lt.1.) val2=val2+(1.-rr)
              endif
           enddo
        enddo

        con=3./(4.*h*float(n)*float(n-1))

        val2=val2*con-3./(4.*h*float(n-1))

        ans=val1-2.*val2

        return
        end
