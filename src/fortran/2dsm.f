c
c	Uses Wahba's code for doing a two-dimensional spline fit.
c       The input data are x, y, and z.
c
        parameter (nmax=1000,maxtbl=200,mxncts=10,igmax=50,maxcon=20,
     $       maxuni=nmax*2,maxpar=maxuni+mxncts,
     $       lwa=maxuni*(2+mxncts+maxuni)+nmax,
     $       liwa=2*nmax+maxuni,lds=1)
	real x(nmax),y(nmax),z(nmax)
        real con(maxcon),tr(6),grid(igmax,igmax),xt(maxtbl),yt(maxtbl)
        real*8 yres(nmax),des(nmax,2),adiag(nmax)
        real*8 lamlim(2),dout(5),tbl(maxtbl,3),coef(maxpar)
        real*8 auxtbl(3,3),svals(nmax),work(lwa),pdes(igmax*igmax,2)
        real*8 pred(igmax*igmax),s(lds,nmax),pdes2(nmax,2)
        integer dim,iout(4),iwork(liwa)
	character file1*40,type*4,label*11

	parameter(zero=0.e0,one=1.e0,big=1.e10)

 1      call qc1('Input data ','2dchi.def',file1)
	open(unit=1,file=file1,status='old',err=1)
	
        igrid=igmax

c - get the data

        xmin=big
        xmax=-big
        ymin=big
        ymax=-big

	n=0
	do i=1,nmax
	   read(1,*,end=666) x1,x2,x3
           n=n+1
           x(n)=x1
           y(n)=x2
           z(n)=x3
           pdes2(n,1)=dble(x(n))
           pdes2(n,2)=dble(y(n))
           xmin=min(xmin,x(n))
           xmax=max(xmax,x(n))
           ymin=min(ymin,y(n))
           ymax=max(ymax,y(n))
	enddo
666	continue
        close(1)

        xbit=(xmax-xmin)/20.
        xmin=xmin-xbit
        xmax=xmax+xbit
        ybit=(ymax-ymin)/20.
        ymin=ymin-ybit
        ymax=ymax+ybit

        if(n.gt.nmax) then
           write(*,"('Too many points - make nmax bigger')")
           goto 766
        endif

        do i=1,n
           yres(i)=dble(z(i))
           des(i,1)=dble(x(i))
           des(i,2)=dble(y(i))
        enddo

        nobs=n
        dim=2
        ncov1=0
        ncov2=0
        m=2
        ntbl=100
        call qi1('Enter job ','2dchi.def',job)
        if (mod(job/100,10).ne.0) 
     $       call qd2('Lam min and max ','2dchi.def',
     $       lamlim(1),lamlim(2))
        call savdef

c        call dptpss(des,nmax,nobs,dim,m,s,lds,ncov1,ncov2,yres,ntbl,
c     $       adiag,lamlim,dout,iout,coef,svals,tbl,maxtbl,auxtbl,work,
c     $       lwa,iwork,liwa,job,info)

        call dtpss(des,nmax,nobs,dim,m,s,lds,ncov1,yres,ntbl,
     $       adiag,lamlim,dout,iout,coef,svals,tbl,maxtbl,auxtbl,work,
     $       lwa,iwork,liwa,job,info)

C        write(6,*) 'lamhat = ',sngl(dout(1))
C        write(6,*) '    Lambda        V'
C        write(6,*) sngl(auxtbl(1,1)),sngl(auxtbl(1,2))
C        write(6,6002) sngl(auxtbl(2,2))
C        write(6,6003) sngl(auxtbl(3,2))
 6002   format('      0       ',f10.3)
 6003   format('    Infty     ',f10.3)

        if (info.gt.0) then
           write(*,*) 'dtpss info',info
           goto 766
        endif

c - get the estimate on the grid

        ng=0
        do i=1,igrid
           do j=1,igrid
              ng=ng+1
              xp=xmin+(xmax-xmin)*float(i-1)/float(igrid-1)
              yp=ymin+(ymax-ymin)*float(j-1)/float(igrid-1)
              pdes(ng,1)=dble(xp)
              pdes(ng,2)=dble(yp)
           enddo
        enddo

        ldpdes=igrid*igrid
        ncov2=0

        call dpred(pdes,ldpdes,ng,dim,m,des,nmax,iout(4),s,lds,
     $       ncov1,ncov2,coef,iout(2),pred,work,lwa,iwork,info)

        if (info.gt.0) then
           write(*,*) 'dpred info',info
           goto 766
        endif

        ytmin=big
        ytmax=-big
        do i=1,ntbl
           xt(i)=sngl(tbl(i,1))
           yt(i)=sngl(tbl(i,2))
           ytmin=min(ytmin,yt(i))
           ytmax=max(ytmax,yt(i))
        enddo

c - make the grid

        xpmin=big
        xpmax=-big

        np=0
        do i=1,igrid
           do j=1,igrid
              np=np+1
              xp=sngl(pred(np))
              grid(i,j)=xp
              xpmin=min(xpmin,xp)
              xpmax=max(xpmax,xp)
           enddo
        enddo

        open(unit=11,file='2dchi.out',status='unknown')
        write(11,*) igmax,igrid,xpmin,xpmax,xmin,xmax,ymin,ymax
        write(11,*) grid
        close(11)

c - get the estimate at the points

        ldpdes=nmax
        ncov2=0

        call dpred(pdes2,ldpdes,nmax,dim,m,des,nmax,iout(4),s,lds,
     $       ncov1,ncov2,coef,iout(2),pred,work,lwa,iwork,info)

        if (info.gt.0) then
           write(*,*) 'dpred2 info',info
           goto 766
        endif

        open(unit=11,file='diff.out',status='unknown')
        do i=1,n
           write(11,*) sngl(pred(i)),z(i)
        enddo
        close(11)

c - get the contour levels

        call qi1('Contours from file (1) or here (0) ','2dchi.def',
     $       icont)

        if(icont.eq.1) then
           write(6,*) 'Lowest value: ',xpmin,'  highest: ',xpmax
           call qr1('Minimum contour ','2dchi.def',chimin)
           if(chimin.eq.0) chimin=xpmin
           open(unit=12,file='contour.dat',status='old')
c           read(12,*) chimin
           ncon=0
           do i=1,maxcon
              read(12,*,end=668) x1
              ncon=ncon+1
              if(ncon.le.maxcon) con(ncon)=x1+chimin
           enddo
 668       continue
        else
           write(6,*) 'Lowest value: ',xpmin,'  highest: ',xpmax
c           call qr3('Contour low, high, and spacing ','2dchi.def',
c     $          clow,chigh,cspace)
c           clow=-10.
           clow=-xpmin
           chigh=xpmax*0.9
           ncon=8
           cspace=(chigh-clow)/float(ncon)
           do i=1,ncon
              con(i)=clow+cspace*(i-1)
           enddo
        endif

        call qc1('Graphics device/type ','2dchi.def',type)
        call pgbegin(0,type,1,1)
        call pgpap(0.,1.)
        call savdef
        call pgscf(2)
        call pgslw(2)
        call pgsch(1.4)
  
c - draw contours

        if(ncon.gt.maxcon) write(6,*) 'Too many contours!'

c	write (6,*) 'Contour levels: ',(con(j),j=1,ncon)

        tr(1)=xmin-(xmax-xmin)/float(igrid-1)
        tr(2)=(xmax-xmin)/float(igrid-1)
        tr(3)=0.
        tr(4)=ymin-(ymax-ymin)/float(igrid-1)
        tr(5)=0.
        tr(6)=(ymax-ymin)/float(igrid-1)

        call pgenv(xmin,xmax,ymin,ymax,0,0)
        call pgpoint(n,x,y,17)
        call pglabel('X (arcsec)','Y (arcsec)','')

        nname=0
        do i=1,80
           if(file1(i:i).eq.'.') goto 767
           nname=nname+1
        enddo
 767    continue
        call pgsch(2.2)
        call pgsch(1.8)

        call pgslw(3)
        call pgcont(grid,igmax,igmax,1,igrid,1,igrid,con,ncon,tr)
 2001   format(f7.3)
        call pgsch(1.1)
        do i=1,ncon
           write(label,2001) con(i)-chimin
           call pgconl(grid,igmax,igmax,1,igrid,1,igrid,con(i),tr,
     $          label,50,15)
        enddo

	call pgend

 766    continue

	end
