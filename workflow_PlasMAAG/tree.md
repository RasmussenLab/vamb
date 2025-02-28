TODO first part
<!-- rename first_pipeline_part and second -->
<!-- rename to contrastive_VAE -->


Create result clusters. | copy with new names
        <!-- plasmid_clusters.tsv -->
        <!-- non_plasmid_clusters.tsv -->
        cluster_scores.tsv | Each clusters score w.r.t plasmid-virus-genome # TODO optional

# Other things
<!-- Remove seed from vamb asymmetric -->
<!-- Update README -->
<!--          - Write section on output files | only on the main output results files. | we can update in the future -->
      

# │       └── vae_clusters_unsplit.tsv # TODO why is this generated ?  - Remvoe it 

results/ 
        plasmid_clusters.tsv
        non_plasmid_clusters.tsv
        cluster_scores.tsv | Each clusters score w.r.t plasmid-virus-genome # TODO optional
        plasmid_bins/ *.fna # TODO future
        nonplasmid_bins/ *.fna # TODO future


first_pipeline_part
├── assembly_mapping_output // sample_Plamb_Ptracker
│       ├── contigs.flt.fna.gz
│       ├── contigs.names.sorted
│       ├── mapped
│       │   ├── sample0.bam
│       │   ├── sample1.bam
│       │   ├── sample2.bam
│       │   ├── sample3.bam
│       │   ├── sample4.bam
│       │   ├── sample5.bam
│       │   ├── sample6.bam
│       │   ├── sample7.bam
│       │   └── sample8.bam
│       ├── mapped_sorted
│       │   ├── sample0.bam.sort
│       │   ├── sample0.bam.sort.bai
│       │   ├── sample1.bam.sort
│       │   ├── sample1.bam.sort.bai
│       │   ├── sample2.bam.sort
│       │   ├── sample2.bam.sort.bai
│       │   ├── sample3.bam.sort
│       │   ├── sample3.bam.sort.bai
│       │   ├── sample4.bam.sort
│       │   ├── sample4.bam.sort.bai
│       │   ├── sample5.bam.sort
│       │   ├── sample5.bam.sort.bai
│       │   ├── sample6.bam.sort
│       │   ├── sample6.bam.sort.bai
│       │   ├── sample7.bam.sort
│       │   ├── sample7.bam.sort.bai
│       │   ├── sample8.bam.sort
│       │   └── sample8.bam.sort.bai
│       ├── spades_sample0
│       │   └── contigs.renamed.fasta
│       ├── spades_sample1
│       │   └── contigs.renamed.fasta
│       ├── spades_sample2
│       │   └── contigs.renamed.fasta
│       ├── spades_sample3
│       │   └── contigs.renamed.fasta
│       ├── spades_sample4
│       │   └── contigs.renamed.fasta
│       ├── spades_sample5
│       │   └── contigs.renamed.fasta
│       ├── spades_sample6
│       │   └── contigs.renamed.fasta
│       ├── spades_sample7
│       │   └── contigs.renamed.fasta
│       └── spades_sample8
│           └── contigs.renamed.fasta
├── log # Removed 
├── Plamb_Ptracker # Rename: second_pipeline_part


## What to do here -- these are not logs but checks
│   ├── log # Fine
│   │   ├── alignment_graph_processing
│   │   │   └── weighted_alignment_graph.finished
│   │   ├── assembly_graph_processing
│   │   │   ├── weighted_assembly_graphs_all_samples.finished
│   │   │   ├── weighted_assembly_graphs_sample0.finished
│   │   │   ├── weighted_assembly_graphs_sample1.finished
│   │   │   ├── weighted_assembly_graphs_sample2.finished
│   │   │   ├── weighted_assembly_graphs_sample3.finished
│   │   │   ├── weighted_assembly_graphs_sample4.finished
│   │   │   ├── weighted_assembly_graphs_sample5.finished
│   │   │   ├── weighted_assembly_graphs_sample6.finished
│   │   │   ├── weighted_assembly_graphs_sample7.finished
│   │   │   └── weighted_assembly_graphs_sample8.finished
│   │   ├── blastn
│   │   │   └── align_contigs.finished
│   │   ├── circularisation
│   │   │   └── circularisation.finished
│   │   ├── classify_bins_with_geNomad.finished
│   │   ├── create_assembly_alignment_graph.finished
│   │   ├── merge_circular_with_graph_clusters.finished
│   │   ├── n2v
│   │   │   └── n2v_assembly_alignment_graph.finished
│   │   ├── neighs
│   │   │   └── extract_neighs_from_n2v_embeddings.finished
│   │   ├── run_geNomad.finished
│   │   └── run_vamb_asymmetric.finished

### Move out ?? 
│   ├── tmp  ### Move out ?? 
│   │   ├── alignment_graph
│   │   │   └── alignment_graph.pkl
│   │   ├── assembly_alignment_graph.pkl
│   │   ├── assembly_graphs
│   │   │   ├── sample0.pkl
│   │   │   ├── sample1.pkl
│   │   │   ├── sample2.pkl
│   │   │   ├── sample3.pkl
│   │   │   ├── sample4.pkl
│   │   │   ├── sample5.pkl
│   │   │   ├── sample6.pkl
│   │   │   ├── sample7.pkl
│   │   │   └── sample8.pkl
│   │   ├── blastn
│   │   │   ├── blastn_all_against_all.txt
│   │   │   └── blastn_all_against_all.txt.redundant
│   │   ├── circularisation
│   │   │   └── max_insert_len_50_circular_clusters.tsv.txt
│   │   ├── geNomad
│   │   │   ├── contigs.flt_aggregated_classification
│   │   │   │   ├── contigs.flt_aggregated_classification.json
│   │   │   │   ├── contigs.flt_aggregated_classification.npz
│   │   │   │   ├── contigs.flt_aggregated_classification.tsv
│   │   │   │   ├── contigs.flt_provirus_aggregated_classification.npz
│   │   │   │   └── contigs.flt_provirus_aggregated_classification.tsv
│   │   │   ├── contigs.flt_aggregated_classification.log
│   │   │   ├── contigs.flt_annotate
│   │   │   │   ├── contigs.flt_annotate.json
│   │   │   │   ├── contigs.flt_genes.tsv
│   │   │   │   ├── contigs.flt_mmseqs2.tsv
│   │   │   │   ├── contigs.flt_proteins.faa
│   │   │   │   └── contigs.flt_taxonomy.tsv
│   │   │   ├── contigs.flt_annotate.log
│   │   │   ├── contigs.flt_find_proviruses
│   │   │   │   ├── contigs.flt_find_proviruses.json
│   │   │   │   ├── contigs.flt_provirus_aragorn.tsv
│   │   │   │   ├── contigs.flt_provirus.fna
│   │   │   │   ├── contigs.flt_provirus_genes.tsv
│   │   │   │   ├── contigs.flt_provirus_mmseqs2.tsv
│   │   │   │   ├── contigs.flt_provirus_proteins.faa
│   │   │   │   ├── contigs.flt_provirus_taxonomy.tsv
│   │   │   │   └── contigs.flt_provirus.tsv
│   │   │   ├── contigs.flt_find_proviruses.log
│   │   │   ├── contigs.flt_marker_classification
│   │   │   │   ├── contigs.flt_features.npz
│   │   │   │   ├── contigs.flt_features.tsv
│   │   │   │   ├── contigs.flt_marker_classification.json
│   │   │   │   ├── contigs.flt_marker_classification.npz
│   │   │   │   ├── contigs.flt_marker_classification.tsv
│   │   │   │   ├── contigs.flt_provirus_features.npz
│   │   │   │   ├── contigs.flt_provirus_features.tsv
│   │   │   │   ├── contigs.flt_provirus_marker_classification.npz
│   │   │   │   └── contigs.flt_provirus_marker_classification.tsv
│   │   │   ├── contigs.flt_marker_classification.log
│   │   │   ├── contigs.flt_nn_classification
│   │   │   │   ├── contigs.flt_nn_classification.json
│   │   │   │   ├── contigs.flt_nn_classification.npz
│   │   │   │   ├── contigs.flt_nn_classification.tsv
│   │   │   │   ├── contigs.flt_provirus_nn_classification.npz
│   │   │   │   └── contigs.flt_provirus_nn_classification.tsv
│   │   │   ├── contigs.flt_nn_classification.log
│   │   │   ├── contigs.flt_summary
│   │   │   │   ├── contigs.flt_plasmid.fna
│   │   │   │   ├── contigs.flt_plasmid_genes.tsv
│   │   │   │   ├── contigs.flt_plasmid_proteins.faa
│   │   │   │   ├── contigs.flt_plasmid_summary.tsv
│   │   │   │   ├── contigs.flt_summary.json
│   │   │   │   ├── contigs.flt_virus.fna
│   │   │   │   ├── contigs.flt_virus_genes.tsv
│   │   │   │   ├── contigs.flt_virus_proteins.faa
│   │   │   │   └── contigs.flt_virus_summary.tsv
│   │   │   └── contigs.flt_summary.log
│   │   ├── n2v
│   │   │   └── assembly_alignment_graph_embeddings
│   │   │       ├── contigs_embedded.txt
│   │   │       ├── embeddings.npz
│   │   │       └── word2vec.model
│   │   └── neighs
│   │       ├── embs_d.pkl
│   │       ├── hoods_clusters_r_0.05_complete.tsv
│   │       ├── hoods_clusters_r_0.05.tsv
│   │       ├── hoods_intraonly_rm_clusters_r_0.05.tsv
│   │       ├── log.txt
│   │       ├── neighs_intraonly_rm_object_r_0.05.npz
│   │       ├── neighs_object_r_0.05.npz
│   │       └── neighs_split_object_r_0.05.npz

│   └── vamb_asymmetric# TODO Change folder name to: contrastive_VAE
│       ├── abundance.npz
│       ├── composition.npz
│       ├── contignames
│       ├── latent.npz
│       ├── lengths.npz
│       ├── log.txt
│       ├── model.pt
│       ├── tmp
│       │   ├── vae_clusters_community_based_split.tsv
│       │   ├── vae_clusters_community_based_unsplit.tsv
│       │   ├── vae_clusters_outside_communities_density_metadata.tsv
│       │   ├── vae_clusters_outside_communities_density_split.tsv
│       │   └── vae_clusters_outside_communities_density_unsplit.tsv
│       ├── vae_clusters_community_based_complete_and_circular_unsplit.tsv // START: should be moved to /tmp
│       ├── vae_clusters_community_based_complete_split.tsv
│       ├── vae_clusters_community_based_complete_unsplit.tsv
│       ├── vae_clusters_density_metadata.tsv
│       ├── vae_clusters_density_split.tsv
│       ├── vae_clusters_density_unsplit.tsv   // END
│       ├── vae_clusters_density_unsplit_geNomadplasclustercontigs_extracted_thr_0.75_thrcirc_0.5_gN_scores.tsv   |-  Scores should be in results
│       ├── vae_clusters_density_unsplit_geNomadplasclustercontigs_extracted_thr_0.75_thrcirc_0.5.tsv           |  Clusters | in results
│       ├── vae_clusters_graph_thr_0.75_candidate_plasmids_gN_scores.tsv  | - Scores should be used in results
│       ├── vae_clusters_graph_thr_0.75_candidate_plasmids.tsv           /  - In results
│       └── vae_clusters_unsplit.tsv # TODO why is this generated ?  - Remvoe it 

<!--  TODO:  -->
├── tmp // This should be moved.  to /secondpart/blastn/database
│   └── Plamb_Ptracker
│       └── blastn
│           ├── contigs.db.ndb
│           ├── contigs.db.nhr
│           ├── contigs.db.nin
│           ├── contigs.db.njs
│           ├── contigs.db.not
│           ├── contigs.db.nsq
│           ├── contigs.db.ntf
│           └── contigs.db.nto

