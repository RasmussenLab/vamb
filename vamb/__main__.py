#!/usr/bin/env python3

# More imports below, but the user's choice of processors must be parsed before
# numpy can be imported.
import vamb
import numpy as np
import sys
import os
import argparse
import torch
import datetime
import time
from math import isfinite
from typing import Optional, IO

_ncpu = os.cpu_count()
DEFAULT_THREADS = 8 if _ncpu is None else min(_ncpu, 8)

# These MUST be set before importing numpy
# I know this is a shitty hack, see https://github.com/numpy/numpy/issues/11826
os.environ["MKL_NUM_THREADS"] = str(DEFAULT_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(DEFAULT_THREADS)
os.environ["OMP_NUM_THREADS"] = str(DEFAULT_THREADS)

# Append vamb to sys.path to allow vamb import even if vamb was not installed
# using pip
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)


################################# DEFINE FUNCTIONS ##########################

def log(string: str, logfile: IO[str], indent: int = 0):
    print(('\t' * indent) + string, file=logfile)
    logfile.flush()


def calc_tnf(
    outdir: str,
    fastapath: Optional[str],
    npzpath: Optional[str],
    mincontiglength: int,
    logfile: IO[str]
) -> vamb.parsecontigs.Composition:
    begintime = time.time()
    log('\nLoading TNF', logfile, 0)
    log(f'Minimum sequence length: {mincontiglength}', logfile, 1)

    if npzpath is not None:
        log(f'Loading composition from npz {npzpath}', logfile, 1)
        composition = vamb.parsecontigs.Composition.load(npzpath)
        composition.filter_min_length(mincontiglength)
    else:
        assert fastapath is not None
        log(f'Loading data from FASTA file {fastapath}', logfile, 1)
        with vamb.vambtools.Reader(fastapath) as file:
            composition = vamb.parsecontigs.Composition.from_file(
                file, minlength=mincontiglength)
        composition.save(os.path.join(outdir, 'composition.npz'))

    elapsed = round(time.time() - begintime, 2)
    print('', file=logfile)
    log(f'Kept {composition.count_bases()} bases in {composition.nseqs} sequences', logfile, 1)
    log(f'Processed TNF in {elapsed} seconds', logfile, 1)

    return composition


def calc_rpkm(
    outdir: str,
    bampaths: Optional[list[str]],
    npzpath: Optional[str],
    comp_metadata: vamb.parsecontigs.CompositionMetaData,
    verify_refhash: bool,
    minid: float,
    nthreads: int,
    logfile: IO[str]
) -> vamb.parsebam.Abundance:

    begintime = time.time()
    log('\nLoading depths', logfile)
    log(f'Reference hash: {comp_metadata.refhash.hex() if verify_refhash else "None"}', logfile, 1)

    # If rpkm is given, we load directly from .npz file
    if npzpath is not None:
        log(f'Loading depths from npz array {npzpath}', logfile, 1)
        abundance = vamb.parsebam.Abundance.load(
            npzpath,
            comp_metadata.refhash if verify_refhash else None
        )
        # I don't want this check in any constructors of abundance, since the constructors
        # should be able to skip this check in case comp and abundance are independent.
        # But when running the main Vamb workflow, we need to assert this.
        if abundance.nseqs != comp_metadata.nseqs:
            assert not verify_refhash
            raise ValueError(
                f"Loaded abundance has {abundance.nseqs} sequences, "
                f"but composition has {comp_metadata.nseqs}."
            )

    else:
        assert bampaths is not None
        log(f'Parsing {len(bampaths)} BAM files with {nthreads} threads', logfile, 1)

        abundance = vamb.parsebam.Abundance.from_files(
            bampaths, comp_metadata, verify_refhash, minid, nthreads)
        abundance.save(os.path.join(outdir, 'abundance.npz'))

    log(f'Min identity: {abundance.minid}\n', logfile, 1)
    log('Order of columns is:', logfile, 1)
    log('\n\t'.join(abundance.samplenames), logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    print('', file=logfile)
    log(f'Processed RPKM in {elapsed} seconds', logfile, 1)

    return abundance


def trainvae(
    outdir: str,
    rpkms: np.ndarray,
    tnfs: np.ndarray,
    lengths: np.ndarray,
    nhiddens: Optional[list[int]],  # set automatically if None
    nlatent: int,
    alpha: Optional[float],  # set automatically if None
    beta: float,
    dropout: Optional[float],  # set automatically if None
    cuda: bool,
    batchsize: int,
    nepochs: int,
    lrate: float,
    batchsteps: list[int],
    logfile: IO[str]
) -> tuple[np.ndarray, np.ndarray]:

    begintime = time.time()
    log('\nCreating and training VAE', logfile)

    assert len(rpkms) == len(tnfs)

    nsamples = rpkms.shape[1]
    vae = vamb.encode.VAE(
        nsamples, nhiddens=nhiddens, nlatent=nlatent,
        alpha=alpha, beta=beta, dropout=dropout, cuda=cuda
    )

    log('Created VAE', logfile, 1)
    dataloader, mask = vamb.encode.make_dataloader(
        rpkms, tnfs, lengths, batchsize, destroy=True, cuda=cuda
    )
    log('Created dataloader and mask', logfile, 1)
    vamb.vambtools.write_npz(os.path.join(outdir, 'mask.npz'), mask)
    n_discarded = len(mask) - mask.sum()
    log(
        f'Number of sequences unsuitable for encoding: {n_discarded}', logfile, 1)
    log(f'Number of sequences remaining: {len(mask) - n_discarded}', logfile, 1)
    print('', file=logfile)

    modelpath = os.path.join(outdir, 'model.pt')
    vae.trainmodel(
        dataloader, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps,
        logfile=logfile, modelfile=modelpath
    )

    print('', file=logfile)
    log('Encoding to latent representation', logfile, 1)
    latent = vae.encode(dataloader)
    vamb.vambtools.write_npz(os.path.join(outdir, 'latent.npz'), latent)
    del vae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    log(f'Trained VAE and encoded in {elapsed} seconds', logfile, 1)

    return mask, latent


def cluster(
    clusterspath: str,
    latent: np.ndarray,
    contignames: np.ndarray, # of dtype object
    windowsize: int,
    minsuccesses: int,
    maxclusters: Optional[int],
    minclustersize: int,
    separator: Optional[str],
    cuda: bool,
    logfile: IO[str]
) -> None:
    begintime = time.time()

    log('\nClustering', logfile)
    log(f'Windowsize: {windowsize}', logfile, 1)
    log(f'Min successful thresholds detected: {minsuccesses}', logfile, 1)
    log(f'Max clusters: {maxclusters}', logfile, 1)
    log(f'Min cluster size: {minclustersize}', logfile, 1)
    log(f'Use CUDA for clustering: {cuda}', logfile, 1)
    log('Separator: {}'.format(None if separator is None else ("\""+separator+"\"")),
        logfile, 1)

    it = vamb.cluster.cluster(
        latent, destroy=True, windowsize=windowsize,
        normalized=False, minsuccesses=minsuccesses, cuda=cuda
    )

    # Iterator of tuple[clusternumber, set[contigs...]]
    renamed = ((str(i+1), {contignames[m] for m in ms})
               for (i, (_n, ms)) in enumerate(it))

    # Binsplit if given a separator
    if separator is not None:
        maybe_split = vamb.vambtools.binsplit(renamed, separator)
    else:
        maybe_split = renamed

    with open(clusterspath, 'w') as clustersfile:
        clusternumber, ncontigs = vamb.vambtools.write_clusters(
            clustersfile, maybe_split, max_clusters=maxclusters,
            min_size=minclustersize, rename=False
        )

    print('', file=logfile)
    log(f'Clustered {ncontigs} contigs in {clusternumber} bins', logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    log(f'Clustered contigs in {elapsed} seconds', logfile, 1)


def write_fasta(
    outdir: str,
    clusterspath: str,
    fastapath: str,
    contignames: np.ndarray, # of object
    contiglengths: np.ndarray,
    minfasta: int,
    logfile: IO[str]
) -> None:
    begintime = time.time()

    log('\nWriting FASTA files', logfile)
    log('Minimum FASTA size: {minfasta}', logfile, 1)
    assert len(contignames) == len(contiglengths)

    lengthof = dict(zip(contignames, contiglengths))
    filtered_clusters: dict[str, set[str]] = dict()

    with open(clusterspath) as file:
        clusters = vamb.vambtools.read_clusters(file)

    for cluster, contigs in clusters.items():
        size = sum(lengthof[contig] for contig in contigs)
        if size >= minfasta:
            filtered_clusters[cluster] = clusters[cluster]

    del lengthof, clusters
    keep: set[str] = set()
    for contigs in filtered_clusters.values():
        keep.update(set(contigs))

    with vamb.vambtools.Reader(fastapath) as file:
        vamb.vambtools.write_bins(
            os.path.join(outdir, "bins"),
            filtered_clusters,
            file,
            maxbins=None
        )

    ncontigs = sum(map(len, filtered_clusters.values()))
    nfiles = len(filtered_clusters)
    print('', file=logfile)
    log(f'Wrote {ncontigs} contigs to {nfiles} FASTA files', logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    log(f'Wrote FASTA in {elapsed} seconds', logfile, 1)


def run(
    outdir: str,
    fastapath: Optional[str],
    compositionpath: Optional[str],
    bampaths: Optional[list[str]],
    rpkmpath: Optional[str],
    mincontiglength: int,
    norefcheck: bool,
    noencode: bool,
    minid: float,
    nthreads: int,
    nhiddens: Optional[list[int]],
    nlatent: int,
    nepochs: int,
    batchsize: int,
    cuda: bool,
    alpha: Optional[float],
    beta: float,
    dropout: Optional[float],
    lrate: float,
    batchsteps: list[int],
    windowsize: int,
    minsuccesses: int,
    minclustersize: int,
    separator: Optional[str],
    maxclusters: Optional[int],
    minfasta: Optional[int],
    logfile: IO[str]
):

    log('Starting Vamb version ' + '.'.join(map(str, vamb.__version__)), logfile)
    log('Date and time is ' + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()

    # Get TNFs, save as npz
    composition = calc_tnf(
        outdir, fastapath, compositionpath, mincontiglength, logfile
    )

    # Parse BAMs, save as npz
    abundance = calc_rpkm(
        outdir, bampaths, rpkmpath, composition.metadata, not norefcheck, minid, nthreads, logfile
    )

    if noencode:
        elapsed = round(time.time() - begintime, 2)
        log(
            f'\nNoencode set, skipping encoding and clustering.\n\nCompleted Vamb in {elapsed} seconds', logfile)
        return None

    # Train, save model
    mask, latent = trainvae(
        outdir, abundance.matrix, composition.matrix, composition.metadata.lengths, nhiddens, nlatent, alpha, beta,
        dropout, cuda, batchsize, nepochs, lrate, batchsteps, logfile
    )

    # Free up memory
    comp_metadata = composition.metadata
    del composition, abundance

    comp_metadata.filter_mask(mask)  # type: ignore
    assert comp_metadata.nseqs == len(latent)

    # Cluster, save tsv file
    clusterspath = os.path.join(outdir, 'clusters.tsv')
    cluster(
        clusterspath, latent, comp_metadata.identifiers, windowsize, minsuccesses,
        maxclusters, minclustersize, separator, cuda, logfile
    )

    del latent

    if minfasta is not None and fastapath is not None:
        # We have already checked fastapath is not None if minfasta is not None.
        write_fasta(
            outdir, clusterspath, fastapath, comp_metadata.identifiers, comp_metadata.lengths, minfasta, logfile
        )

    elapsed = round(time.time() - begintime, 2)
    log(f'\nCompleted Vamb in {elapsed} seconds', logfile)


def main():
    doc = f"""Vamb: Variational autoencoders for metagenomic binning.
    
    Version: {'.'.join([str(i) for i in vamb.__version__])}

    Default use, good for most datasets:
    vamb --outdir out --fasta my_contigs.fna --bamfiles *.bam -o C

    For advanced use and extensions of Vamb, check documentation of the package
    at https://github.com/RasmussenLab/vamb."""
    parser = argparse.ArgumentParser(
        prog="vamb",
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s outdir tnf_input rpkm_input [options]",
        add_help=False)

    # Help
    helpos = parser.add_argument_group(
        title='Help and version', description=None)
    helpos.add_argument(
        '-h', '--help', help='print help and exit', action='help')
    helpos.add_argument('--version', action='version',
                        version=f'Vamb {".".join(map(str, vamb.__version__))}')

    # Positional arguments
    reqos = parser.add_argument_group(
        title='Output (required)', description=None)
    reqos.add_argument('--outdir', metavar='', required=True,
                       help='output directory to create')

    # TNF arguments
    tnfos = parser.add_argument_group(
        title='TNF input (either fasta or all .npz files required)')
    tnfos.add_argument('--fasta', metavar='', help='path to fasta file')
    tnfos.add_argument('--composition', metavar='',
                       help='path to .npz of composition')

    # RPKM arguments
    rpkmos = parser.add_argument_group(
        title='RPKM input (either BAMs or .npz required)')
    rpkmos.add_argument('--bamfiles', metavar='',
                        help='paths to (multiple) BAM files', nargs='+')
    rpkmos.add_argument('--rpkm', metavar='', help='path to .npz of RPKM (abundances)')

    # Optional arguments
    inputos = parser.add_argument_group(title='IO options', description=None)

    inputos.add_argument('-m', dest='minlength', metavar='', type=int, default=250,
                         help='ignore contigs shorter than this [250]')
    inputos.add_argument('-z', dest='minid', metavar='', type=float, default=0.0,
                         help='ignore reads with nucleotide identity below this [0.0]')
    inputos.add_argument('-p', dest='nthreads', metavar='', type=int, default=DEFAULT_THREADS,
                         help=('number of threads to use '
                               '[min(' + str(DEFAULT_THREADS) + ', nbamfiles)]'))
    inputos.add_argument('--norefcheck', help='skip reference name hashing check [False]',
                         action='store_true')
    inputos.add_argument('--minfasta', dest='minfasta', metavar='', type=int, default=None,
                         help='minimum bin size to output as fasta [None = no files]')
    inputos.add_argument('--noencode', help='Output tnfs and abundances only, do not encode or cluster [False]',
                         action='store_true')

    # VAE arguments
    vaeos = parser.add_argument_group(title='VAE options', description=None)

    vaeos.add_argument('-n', dest='nhiddens', metavar='', type=int, nargs='+',
                       default=None, help='hidden neurons [Auto]')
    vaeos.add_argument('-l', dest='nlatent', metavar='', type=int,
                       default=32, help='latent neurons [32]')
    vaeos.add_argument('-a', dest='alpha',  metavar='', type=float,
                       default=None, help='alpha, weight of TNF versus depth loss [Auto]')
    vaeos.add_argument('-b', dest='beta',  metavar='', type=float,
                       default=200.0, help='beta, capacity to learn [200.0]')
    vaeos.add_argument('-d', dest='dropout',  metavar='', type=float,
                       default=None, help='dropout [Auto]')
    vaeos.add_argument(
        '--cuda', help='use GPU to train & cluster [False]', action='store_true')

    trainos = parser.add_argument_group(
        title='Training options', description=None)

    trainos.add_argument('-e', dest='nepochs', metavar='', type=int,
                         default=300, help='epochs [300]')
    trainos.add_argument('-t', dest='batchsize', metavar='', type=int,
                         default=256, help='starting batch size [256]')
    trainos.add_argument('-q', dest='batchsteps', metavar='', type=int, nargs='*',
                         default=[25, 75, 150, 225], help='double batch size at epochs [25 75 150 225]')
    trainos.add_argument('-r', dest='lrate',  metavar='', type=float,
                         default=1e-3, help='learning rate [0.001]')

    # Clustering arguments
    clusto = parser.add_argument_group(
        title='Clustering options', description=None)
    clusto.add_argument('-w', dest='windowsize', metavar='', type=int,
                        default=200, help='size of window to count successes [200]')
    clusto.add_argument('-u', dest='minsuccesses', metavar='', type=int,
                        default=20, help='minimum success in window [20]')
    clusto.add_argument('-i', dest='minsize', metavar='', type=int,
                        default=1, help='minimum cluster size [1]')
    clusto.add_argument('-c', dest='maxclusters', metavar='', type=int,
                        default=None, help='stop after c clusters [None = infinite]')
    clusto.add_argument('-o', dest='separator', metavar='', type=str,
                        default=None, help='binsplit separator [None = no split]')

    ######################### PRINT HELP IF NO ARGUMENTS ###################
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    outdir: str = os.path.abspath(args.outdir)
    fasta: Optional[str] = args.fasta
    composition: Optional[str] = args.composition
    bamfiles: Optional[list[str]] = args.bamfiles
    rpkm: Optional[str] = args.rpkm
    minlength: int = args.minlength

    if args.minid != 0.0 and bamfiles is None:
        raise argparse.ArgumentTypeError(
            'If minid is set, RPKM must be passed as bam files')

    minid: float = args.minid
    nthreads: int = args.nthreads
    norefcheck: bool = args.norefcheck
    minfasta: Optional[int] = args.minfasta
    noencode: bool = args.noencode
    nhiddens: Optional[list[int]] = args.nhiddens
    nlatent: int = args.nlatent
    alpha: Optional[float] = args.alpha
    beta: float = args.beta
    dropout: Optional[float] = args.dropout
    cuda: bool = args.cuda
    nepochs: int = args.nepochs
    batchsize: int = args.batchsize
    batchsteps: list[int] = args.batchsteps
    lrate: float = args.lrate
    windowsize: int = args.windowsize
    minsuccesses: int = args.minsuccesses
    minsize: int = args.minsize
    maxclusters: Optional[int] = args.maxclusters
    separator: Optional[str] = args.separator

    ######################### CHECK INPUT/OUTPUT FILES #####################

    # Outdir does not exist
    if os.path.exists(outdir):
        raise FileExistsError(outdir)

    # Outdir is in an existing parent dir
    parentdir = os.path.dirname(outdir)
    if parentdir and not os.path.isdir(parentdir):
        raise NotADirectoryError(parentdir)

    # Make sure only one TNF input is there
    if not (composition is None) ^ (fasta is None):
        raise argparse.ArgumentTypeError(
            'Must specify either FASTA or composition path')

    for path in (fasta, composition):
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError(path)

    # Make sure only one RPKM input is there
    if not ((bamfiles is None) ^ (rpkm is None)):
        raise argparse.ArgumentTypeError(
            'Must specify exactly one of BAM files or RPKM input')

    if rpkm is not None and not os.path.isfile(rpkm):
        raise FileNotFoundError('Not an existing non-directory file: ' + rpkm)

    if bamfiles is not None:
        for bampath in bamfiles:
            if not os.path.isfile(bampath):
                raise FileNotFoundError(
                    'Not an existing non-directory file: ' + bampath)

            # Check this early, since I expect users will forget about this
            if not vamb.parsebam.pycoverm.is_bam_sorted(bampath):
                raise ValueError(
                    f'BAM file {bampath} is not sorted by reference.')

    # Check minfasta settings
    if minfasta is not None and fasta is None:
        raise argparse.ArgumentTypeError('If minfasta is not None, '
                                         'input fasta file must be given explicitly')

    if minfasta is not None and minfasta < 0:
        raise argparse.ArgumentTypeError(
            'Minimum FASTA output size must be nonnegative')

    ####################### CHECK ARGUMENTS FOR TNF AND BAMFILES ###########
    if minlength < 250:
        raise argparse.ArgumentTypeError(
            'Minimum contig length must be at least 250')

    if not isfinite(minid) or minid < 0.0 or minid > 1.0:
        raise argparse.ArgumentTypeError(
            'Minimum nucleotide ID must be in [0,1]')

    if nthreads < 1:
        raise argparse.ArgumentTypeError(
            'Zero or negative subprocesses requested')

    ####################### CHECK VAE OPTIONS ################################
    if nhiddens is not None and any(i < 1 for i in nhiddens):
        raise argparse.ArgumentTypeError(
            f'Minimum 1 neuron per layer, not {min(nhiddens)}')

    if nlatent < 1:
        raise argparse.ArgumentTypeError(
            f'Minimum 1 latent neuron, not {nlatent}')

    if alpha is not None and (alpha <= 0 or alpha >= 1):
        raise argparse.ArgumentTypeError('alpha must be above 0 and below 1')

    if beta <= 0:
        raise argparse.ArgumentTypeError('beta cannot be negative or zero')

    if dropout is not None and (dropout < 0 or dropout >= 1):
        raise argparse.ArgumentTypeError('dropout must be in 0 <= d < 1')

    if cuda and not torch.cuda.is_available():
        raise ModuleNotFoundError(
            'Cuda is not available on your PyTorch installation')

    ###################### CHECK TRAINING OPTIONS ####################
    if nepochs < 1:
        raise argparse.ArgumentTypeError(f'Minimum 1 epoch, not {nepochs}')

    if batchsize < 1:
        raise argparse.ArgumentTypeError(
            f'Minimum batchsize of 1, not {batchsize}')

    batchsteps = sorted(set(batchsteps))
    if max(batchsteps, default=0) >= nepochs:
        raise argparse.ArgumentTypeError(
            'All batchsteps must be less than nepochs')

    if min(batchsteps, default=1) < 1:
        raise argparse.ArgumentTypeError('All batchsteps must be 1 or higher')

    if lrate <= 0:
        raise argparse.ArgumentTypeError('Learning rate must be positive')

    ###################### CHECK CLUSTERING OPTIONS ####################
    if minsize < 1:
        raise argparse.ArgumentTypeError(
            'Minimum cluster size must be at least 0')

    if windowsize < 1:
        raise argparse.ArgumentTypeError('Window size must be at least 1')

    if minsuccesses < 1 or minsuccesses > windowsize:
        raise argparse.ArgumentTypeError(
            'Minimum cluster size must be in 1:windowsize')

    if separator is not None and len(separator) == 0:
        raise argparse.ArgumentTypeError(
            'Binsplit separator cannot be an empty string')

    ###################### SET UP LAST PARAMS ############################

    # This doesn't actually work, but maybe the PyTorch folks will fix it sometime.
    torch.set_num_threads(nthreads)

    ################### RUN PROGRAM #########################
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass
    except:
        raise

    logpath = os.path.join(outdir, 'log.txt')

    with open(logpath, 'w') as logfile:
        run(
            outdir,
            fasta,
            composition,
            bamfiles,
            rpkm,
            mincontiglength=minlength,
            norefcheck=norefcheck,
            noencode=noencode,
            minid=minid,
            nthreads=nthreads,
            nhiddens=nhiddens,
            nlatent=nlatent,
            nepochs=nepochs,
            batchsize=batchsize,
            cuda=cuda,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            lrate=lrate,
            batchsteps=batchsteps,
            windowsize=windowsize,
            minsuccesses=minsuccesses,
            minclustersize=minsize,
            separator=separator,
            maxclusters=maxclusters,
            minfasta=minfasta,
            logfile=logfile
        )


if __name__ == '__main__':
    main()
