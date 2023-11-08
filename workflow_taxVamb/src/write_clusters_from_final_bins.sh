#!/usr/bin/bash


while getopts "d:o:" opt; do
  case $opt in
    d) drep_dir=$OPTARG    ;;
    o) clusters_file=$OPTARG    ;;
    *) echo 'error' >&2
       exit 1
  esac
done
echo 'creating z y v clusters from the final set of bins'
cd $drep_dir
for bin in $(ls . 2> /dev/null)

do  
if [[ $bin == **".fna" ]]
then

cluster_name=$(echo $bin | sed 's=.fna==g' | sed 's=.fa==g')
echo $cluster_name
#for contig in $(grep '>' $bin | sed 's=>==g') 
#do
#echo -e   "$cluster_name""\t""$contig"  >> $clusters_file
#done


fi
done
