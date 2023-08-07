#!/usr/bin/bash


while getopts "d:o:" opt; do
  case $opt in
    d) drep_dir=$OPTARG    ;;
    o) outdir=$OPTARG    ;;
    *) echo 'error' >&2
       exit 1
  esac
done
clusters_file=${outdir}/avamb/avamb_manual_drep_disjoint_clusters.tsv
echo 'creating z y v clusters from the final set of bins'
for s in $(ls $drep_dir)
do
s="$drep_dir"/"$s"/
if [ -d "$s" ]
then
cd $s
for bin in $(ls . 2> /dev/null)

do  
if [[ $bin == **".fna" ]]
then

cluster_name=$(echo $bin | sed 's=.fna==g' | sed 's=.fa==g')

for contig in $(grep '>' $bin | sed 's=>==g') 
do
echo -e   "$cluster_name""\t""$contig"  >> $clusters_file
done


fi
done

fi
done 
