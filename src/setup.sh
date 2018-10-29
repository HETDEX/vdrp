# allow python to find vdrp modules
export PYTHONPATH=$PYTHONPATH:/home/03570/jsnigula/code/vdrp/src/python
# this is required for daophot
module load gcc
# to find daophot
export PATH=/home/00115/gebhardt/lib/daoprogs:$PATH
# to find daomaster
export PATH=/home/00115/gebhardt/lib/daoprogs/moreprogs2:$PATH
# to find getoff2
export PATH=/work/00115/gebhardt/maverick/scripts/astrometry:$PATH
# to find  immosaicv & imrot
export PATH=~gebhardt/bin:$PATH
