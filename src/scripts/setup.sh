# allow python to find vdrp modules
export PYTHONPATH=$PYTHONPATH:/Users/mxhf/work/MPE/hetdex/src/vdrp_rewrite/vdrp/src/python
# this is required for daophot
#module load gcc/5.4.0
# to find daophot
export PATH=/Users/mxhf/work/MPE/hetdex/src/vdrp_rewrite/daoprogs:$PATH
# to find daomaster
export PATH=/Users/mxhf/work/MPE/hetdex/src/vdrp_rewrite/daoprogs/moreprogs2:$PATH
# to find getoff2
export PATH=/Users/mxhf/work/MPE/hetdex/src/vdrp_rewrite/sciprogs:$PATH
# to find  immosaicv & imrot
export PATH=~gebhardt/bin/immosaicv:$PATH
