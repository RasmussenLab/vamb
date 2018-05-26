# Vamb
Created by Jakob Nybo Nissen and Simon Rasmussen, Technical University of Denmark.

Please contact jakni@bioinformatics.dtu.dk for bug fixes and feature requests.

Vamb is a metagenomic binner which feeds sequence composition information from a contig catalogue and co-abundance information from BAM files into a variational autoencoder and clusters the latent 
representation. It performs excellently with many samples, well with 5-10 samples and poorly relying only on the nucleotide composition. Vamb is implemented almost purely in Python and can be used both 
from commandline and from within a Python interpreter.

### Installation
Vamb requires Python 3.5 or newer and the following Python packages to run:
- PyTorch
- pandas
- Numpy
- pysam

### Running from command line
PUT SOMETHING HERE WHEN THE API IS MORE STABLE.

### Running from Python
PUT SOMETHING HERE WHEN THE API IS MORE STABLE.

### Workflow
__1. Calculation of tetranucleotide frequencies__

First the tetranucleotide frequencies (TNF) of a FASTA file containing the contigs must be calculated. TNFs are represented by their canonical kmer, so there are 136 of them. The TNFs are 
zscore-normalized across contigs. As TNF is unstable with short contigs, all contigs with a length below a certain threshold is removed.

__2. Calculation of relative contig abundance__

Contig abundance is calculated as reads per kilobase reference per million mapped reads (RPKM). As BWA mem handles redundant references poorly, each segment is counted as half a hit, unless the other 
segment is unmapped in which case it's counted as one hit. Secondary hits are counted as well.

__3. Compressing with variational autoencoder__

TNF is represented as a (contigs x 136) array, abundance as a (contigs x samples) array. These are zscore-normalized across the X axis and fed into a variational autoencoder. The loss is a sum of binary 
cross-entropy of abundance, mean square error of TNF and Kullback-Leibler divergence between the latent representation and a gaussian prior.

__4. Iterative medoid clustering of latent representation__

xxx

__5. Cluster merging with walktrap algorithm__

xxx


### FAQ
