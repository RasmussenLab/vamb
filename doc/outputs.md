# Outputs

### Vamb
- `log.txt` - A text file with information about the Vamb run. Look here (and at stderr) if you experience errors.
- `composition.npz`: A Numpy .npz file that contain all kmer composition information computed by Vamb from the FASTA file. This can be provided to another run of Vamb to skip the composition calculation step.
- `abundance.npz`: Similar to `composition.npz`, but this file contains information calculated from the BAM files. Using this as input instead of BAM files will skip re-parsing the BAM files, which take a significant amount of time.
- `model.pt`: A file containing the trained VAE model. When running Vamb from a Python interpreter, the VAE can be loaded from this file to skip training.
- `latent.npz`: This contains the output of the VAE model.
- `vae_clusters_unsplit.tsv` - A two-column text file with one row per sequence:
  Left column for the cluster (i.e bin) name, right column for the sequence name.
  You can create the FASTA-file bins themselves using the script in `src/create_fasta.py`
- (if binsplitting is enabled:) `vae_clusters_split.tsv`, similar to the unsplit version, but after binsplitting
- `vae_clusters_metadata.tsv`: A file with some metadata about clusters.
    - Name: The name of the cluster
    - Radius: Cosine radius in latent space. Small clusters are usually more likely to be pure.
    - Peak/valley ratio: A small PVR means the cluster's edges is more well defined, and hence the cluster is more likely pure
    - Kind: Currently, Vamb produces three kinds of clusters:
        - Normal: Defined by a local density in latent space. Most good clusters are of this type
        - Loner: A contig far away from everything else in latent space.
        - Fallback: After failing to produce good clusters for some time, these (usually poor) clusters are created
          to not get stuck in an infinite loop when clustering
    - Bp: Sum of length of all sequences in the cluster
    - Ncontigs: Number of sequences in the cluster

### TaxVamb
[TODO]

### Taxometer
[TODO]

### AVAMB
Same as VAMB, but also:
- `aae_y_clusters_{split,unsplit}.tsv`: The clusters obtained from the categorical latent space
- `aae_z_latent.npz`: Like `latent.npz`, but of the adversarial Z latent space
- `aae_z_clusters_{metadata,split,unsplit}.tsv`: Like the corresponding `vae_clusters*` files, but from the adversarial Z latent space

