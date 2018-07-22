#!/bin/bash
# This shel script check that the results from
# the original astrometry code and the vdrp code are
# identical (or close enough).
d=$1

echo "#######################################"
echo "# daophot find"
echo "#######################################"
echo "### Checking that coo are identical (no output is good) ..."
for g in `ls $d.orig/*.coo`; do f=`basename $g`; diffout=`diff $d.orig/$f $d.vdrp/$f`; if [ x"$diffout" != x"" ]; then echo $f ":"; echo $diffout; fi ; done

echo ""

echo "### Checking that lst files are identical (no output is good)..."
for g in `ls $d.orig/*.lst`; do f=`basename $g`; diffout=`diff $d.orig/$f $d.vdrp/$f`; if [ x"$diffout" != x"" ]; then echo $f ":"; echo $diffout; fi ; done

echo ""

echo "#######################################"
echo "# daophot phot & allstar output"
echo "#######################################"
echo "### Checking that .ap are identical (no output is good)..."
for g in `ls $d.orig/*.ap`; do f=`basename $g`; diffout=`diff $d.orig/$f $d.vdrp/$f`; if [ x"$diffout" != x"" ]; then echo $f ":"; echo $diffout; fi ; done

echo ""

echo "Checking that als files are identical (no output is good)..."
for g in `ls $d.orig/2???????T??????_???.als`; do f=`basename $g`; diffout=`diff $d.orig/$f $d.vdrp/$f`; if [ x"$diffout" != x"" ]; then echo $f ":"; echo $diffout; fi ; done

echo ""

echo "#######################################"
echo "# mktot"
echo "#######################################"
echo "### Checking that *tot.als files numerically identical (they can be sorted differently). Should be all zero ..."
for g in `ls $d.vdrp/*tot.als`; 
	do f=`basename $g`; 
	cat ./$d.orig/$f | awk '{s+=1; if (s>3) { print $0} }' | sort -k2 -n > t1; 
	cat ./$d.vdrp/$f | awk '{s+=1; if (s>3) { print $0} }' | sort -k2 -n > t2; 
	paste t1 t2 | awk '{ d+= ($2-$6)^2. + ($3-$7)^2.}END{print d}'; 
	done

echo ""

echo "#######################################"
echo "# rmaster"
echo "#######################################"
echo "### Checking that all.mch identical (no output is good)..."
f=all.mch
diff $d.orig/$f  $d.vdrp/$f

echo ""

echo "### Checking that all.raw identical (no output is good)..."
f=all.raw
diff $d.orig/$f  $d.vdrp/$f

echo ""

echo "#######################################"
echo "# flux_norm"
echo "#######################################"
echo "### Checking that norm.dat are similar enough ..."
cat $d/norm.dat; echo "";cat $d.orig/norm.dat

echo ""

echo "#######################################"
echo "# get_ra_dec_orig"
echo "#######################################"
echo "### Checking that radec.orig are identical ..."
f=radec.orig
diff $d.vdrp/$f $d.orig/$f

echo ""


echo "#######################################"
echo "# compute_offset"
echo "#######################################"
echo "### Checking that radec.dat is close enough ..."
f=radec.dat
diff $d.vdrp/$f $d.orig/$f

echo ""

echo "### Checking getoff2.out close enough ..."
f=getoff2.out
diff $d.vdrp/$f $d.orig/$f


echo ""

echo "### Checking that getoff.out is close enough, should be all zero ..."
f=getoff.out
for g in `ls $d.vdrp/*tot.als`; do 
	f=`basename $f`; 
	cat ./$d.orig/$f | awk '{s+=1; if (s>3) { print $0} }' | sort -k2 -n > t1; cat ./$d.vdrp/$f | awk '{s+=1; if (s>3) { print $0} }' | sort -k2 -n > t2; paste t1 t2 | awk '{ d+= ($3-$5)^2. + ($4-$6)^2.}END{print d}';
done

