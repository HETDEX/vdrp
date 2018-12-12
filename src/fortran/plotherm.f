
      parameter(nmax=10000)
      real v(nmax),y(nmax)
      parameter(pi=3.1415926539)
      
      nl=1000

      call pgbegin(0,'?',2,2)
      call pgpap(0.,1.)
      call pgsch(1.5)
      call pgscf(2)
      call pgslw(2)

      vmin=-4.
      vmax=4.
      ymin=-.2
      ymax=1.
      amp=2.
      vel=0.
      sig=1.
      do i=1,nl
         v(i)=vmin+(vmax-vmin)*float(i-1)/float(nl-1)
      enddo

      call pgenv(vmin,vmax,ymin,ymax,0,0)
      call pglabel('','','-0.2<h3<0.2, h4=0')
      h4=0.
      do h3=-.2,0.2,.05
         do i=1,nl
            w=(v(i)-vel)/sig
            gaus=exp(-w*w/2.)/sqrt(2.*pi)
            y(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))
         enddo
         call pgsci(1)
         if(h3.le.0) call pgsci(2)
         call pgline(nl,v,y)
      enddo

      call pgenv(vmin,vmax,ymin,ymax,0,0)
      call pglabel('','','-0.2<h3<0.2, h4=-0.1')
      h4=-0.1
      do h3=-.2,0.2,.05
         do i=1,nl
            w=(v(i)-vel)/sig
            gaus=exp(-w*w/2.)/sqrt(2.*pi)
            y(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))
         enddo
         call pgsci(1)
         if(h3.le.0) call pgsci(2)
         call pgline(nl,v,y)
      enddo

      call pgenv(vmin,vmax,ymin,ymax,0,0)
      call pglabel('','','h3=0, -0.2<h4<0.2')
      h3=0.
      do h4=-.2,0.2,.05
         do i=1,nl
            w=(v(i)-vel)/sig
            gaus=exp(-w*w/2.)/sqrt(2.*pi)
            y(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))
         enddo
         call pgsci(1)
         if(h4.le.0) call pgsci(2)
         call pgline(nl,v,y)
      enddo

      call pgenv(vmin,vmax,ymin,ymax,0,0)
      call pglabel('','','h3=-0.1, -0.2<h4<0.2')
      h3=-0.1
      do h4=-.2,0.2,.05
         do i=1,nl
            w=(v(i)-vel)/sig
            gaus=exp(-w*w/2.)/sqrt(2.*pi)
            y(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))
         enddo
         call pgsci(1)
         if(h4.le.0) call pgsci(2)
         call pgline(nl,v,y)
      enddo

      call pgend
      end

      function fh3(x)
      fh3=1./sqrt(6.)*(2.*sqrt(2.)*x*x*x-3.*sqrt(2.)*x)
      return
      end
      function fh4(x)
      fh4=1./sqrt(24.)*(4.*x*x*x*x-12.*x*x+3.)
      return
      end
