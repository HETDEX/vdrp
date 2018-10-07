c	Program to convert a .mag1.lis file to a .als format file.
c
c
c	-- Open the input and output files, write a fake header
        character*30 file1
 1      write(*,"('Input file : '$)")
        read(*,'(a)') file1
	open (unit=1,file=file1,status='old',err=1)
	open (unit=2,file='convcoo.out',status='unknown',err=99)
	write (2,106)
106	format(1x/1x/1x)
c
	nc=0
20	continue
	read (1,*,end=50) ind,x,y
	write (2,100) ind,x,y
100	format(1x,i5,2(1x,f8.3))
c100	format(1x,i5,2(1x,f8.2))
	nc=nc+1
	goto 20
c
c	-- done
50	continue
	close(unit=1)
	close(unit=2)
	write (6,*) nc,' stars converted'
	stop
c
c	-- file open error
99	continue
	write (6,*) 'file open error'
	stop
	end
