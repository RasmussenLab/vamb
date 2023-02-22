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
for s in $(ls $drep_dir)
do
s="$drep_dir"/"$s"/
if [ -d "$s" ]
then
cd $s
#for bin in $(ls dereplicated_genomes 2> /dev/null)
for bin in $(ls . 2> /dev/null)

do  
if [[ $bin == **".fna" ]]
then
#echo $bin

cluster_name=$(echo $bin | sed 's=.fna==g' | sed 's=.fa==g')
#echo $cluster_name
#bin="$s"dereplicated_genomes/"$bin"
#bin="$s""$bin"
#echo $bin
#grep  '>' $bin | sed 's=>==g'
for contig in $(grep '>' $bin | sed 's=>==g') 
do
echo -e   "$cluster_name""\t""$contig"  >> $clusters_file
done


fi
done
#for bin in $(ls ripped_bins_selected 2> /dev/null)
#do  
#if [[ $bin == **".fna" ]]
#then
#bin="$s"ripped_bins_selected/"$bin"
#
#for contig in $(grep '>' $bin | sed 's=>==g') 
#do
#echo -e   "$cluster_name""$\t""$contig"  >> $clusters_file
#done
#fi
#done 
fi
done 
