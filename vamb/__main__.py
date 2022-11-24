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
from typing import Optional, IO, Union
from pathlib import Path

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


class FASTAPath(type(Path())):
    pass



class CompositionPath(type(Path())):
    pass


class CompositionOptions:
    __slots__ = ["path", "min_contig_length"]

    def __init__(
        self, fastapath: Optional[Path], npzpath: Optional[Path], min_contig_length: int
    ):
        assert isinstance(fastapath, (Path, type(None)))
        assert isinstance(npzpath, (Path, type(None)))
        assert isinstance(min_contig_length, int)

        if min_contig_length < 250:
            raise argparse.ArgumentTypeError(
                "Minimum contig length must be at least 250"
            )

        if not (fastapath is None) ^ (npzpath is None):
            raise argparse.ArgumentTypeError(
                "Must specify either FASTA or composition path"
            )

        for path in (fastapath, npzpath):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)

        if fastapath is not None:
            self.path = FASTAPath(fastapath)
        else:
            assert npzpath is not None
            self.path = CompositionPath(npzpath)
        self.min_contig_length = min_contig_length


class AbundancePath(type(Path())):
    pass


class JGIPath(type(Path())):
    pass


class AbundanceOptions:
    __slots__ = ["path", "min_alignment_id", "refcheck"]

    def __init__(
        self,
        bampaths: Optional[list[Path]],
        abundancepath: Optional[Path],
        jgipath: Optional[Path],
        min_alignment_id: Optional[float],
        refcheck: bool,
    ):
        assert isinstance(bampaths, (list, type(None)))
        assert isinstance(abundancepath, (Path, type(None)))
        assert isinstance(jgipath, (Path, type(None)))
        assert isinstance(min_alignment_id, (float, type(None)))
        assert isinstance(refcheck, bool)

        # Make sure only one RPKM input is there
        if not ((bampaths is None) + (abundancepath is None) + (jgipath is None)) != 1:
            raise argparse.ArgumentTypeError(
                "Must specify exactly one of BAM files, abundance NPZ or JGI file input"
            )

        if abundancepath is not None:
            if not abundancepath.is_file():
                raise FileNotFoundError(
                    f'Not an existing non-directory file: "{str(abundancepath)}"'
                )
            self.path = AbundancePath(abundancepath)

        elif bampaths is not None:
            for bampath in bampaths:
                if not bampath.is_file():
                    raise FileNotFoundError(
                        f'Not an existing non-directory file: "{str(bampath)}"'
                    )
            self.path = bampaths

        if jgipath is not None:
            if not jgipath.is_file():
                raise FileNotFoundError(
                    f'Not an existing non-directory file: "{str(jgipath)}"'
                )
            self.path = JGIPath(jgipath)

        if min_alignment_id is not None:
            if bampaths is None:
                raise argparse.ArgumentTypeError(
                    "If minid is set, RPKM must be passed as bam files"
                )
            if (
                not isfinite(min_alignment_id)
                or min_alignment_id < 0.0
                or min_alignment_id > 1.0
            ):
                raise argparse.ArgumentTypeError(
                    "Minimum nucleotide ID must be in [0,1]"
                )
            self.min_alignment_id = min_alignment_id
        else:
            self.min_alignment_id = 0.0

        self.refcheck = refcheck


class VAEOptions:
    __slots__ = ["nhiddens", "nlatent", "alpha", "beta", "dropout", "cuda"]

    def __init__(
        self,
        nhiddens: Optional[list[int]],
        nlatent: int,
        alpha: Optional[float],
        beta: float,
        dropout: Optional[float],
    ):
        assert isinstance(nhiddens, (list, type(None)))
        assert isinstance(nlatent, int)
        assert isinstance(alpha, (float, type(None)))
        assert isinstance(beta, float)
        assert isinstance(dropout, (float, type(None)))

        if nhiddens is not None and any(i < 1 for i in nhiddens):
            raise argparse.ArgumentTypeError(
                f"Minimum 1 neuron per layer, not {min(nhiddens)}"
            )
        self.nhiddens = nhiddens

        if nlatent < 1:
            raise argparse.ArgumentTypeError(f"Minimum 1 latent neuron, not {nlatent}")
        self.nlatent = nlatent

        if alpha is not None and (alpha <= 0 or alpha) >= 1:
            raise argparse.ArgumentTypeError("alpha must be above 0 and below 1")
        self.alpha = alpha

        if beta <= 0 or not isfinite(beta):
            raise argparse.ArgumentTypeError("beta cannot be negative or zero")
        self.beta = beta

        if dropout is not None and (dropout < 0 or dropout >= 1):
            raise argparse.ArgumentTypeError("dropout must be in 0 <= d < 1")
        self.dropout = dropout


class TrainingOptions:
    __slots__ = ["nepochs", "batchsize", "batchsteps", "lrate"]

    def __init__(
        self, nepochs: int, batchsize: int, batchsteps: list[int], lrate: float
    ):
        assert isinstance(nepochs, int)
        assert isinstance(batchsize, int)
        assert isinstance(batchsteps, list)
        assert isinstance(lrate, float)

        if nepochs < 1:
            raise argparse.ArgumentTypeError(f"Minimum 1 epoch, not {nepochs}")
        self.nepochs = nepochs

        if batchsize < 1:
            raise argparse.ArgumentTypeError(f"Minimum batchsize of 1, not {batchsize}")
        self.batchsize = batchsize

        batchsteps = sorted(set(batchsteps))
        if max(batchsteps, default=0) >= self.nepochs:
            raise argparse.ArgumentTypeError("All batchsteps must be less than nepochs")

        if min(batchsteps, default=1) < 1:
            raise argparse.ArgumentTypeError("All batchsteps must be 1 or higher")
        self.batchsteps = batchsteps

        if lrate <= 0.0:
            raise argparse.ArgumentTypeError("Learning rate must be positive")
        self.lrate = lrate


class ClusterOptions:
    __slots__ = [
        "window_size",
        "min_successes",
        "min_cluster_size",
        "max_clusters",
        "binsplit_separator",
    ]

    def __init__(
        self,
        window_size: int,
        min_successes: int,
        min_cluster_size: int,
        max_clusters: Optional[int],
        binsplit_separator: Optional[str],
    ):
        assert isinstance(window_size, int)
        assert isinstance(min_successes, int)
        assert isinstance(min_cluster_size, int)
        assert isinstance(max_clusters, (int, type(None)))
        assert isinstance(binsplit_separator, (str, type(None)))

        if window_size < 1:
            raise argparse.ArgumentTypeError("Window size must be at least 1")
        self.window_size = window_size

        if min_cluster_size < 1:
            raise argparse.ArgumentTypeError("Minimum cluster size must be at least 0")
        self.min_cluster_size = min_cluster_size

        if min_successes < 1 or min_successes > window_size:
            raise argparse.ArgumentTypeError(
                "Minimum cluster size must be in 1:windowsize"
            )
        self.min_successes = min_successes

        if max_clusters is not None and max_clusters < 1:
            raise argparse.ArgumentTypeError("Max clusters must be at least 1")
        self.max_clusters = max_clusters

        if binsplit_separator is not None and len(binsplit_separator) == 0:
            raise argparse.ArgumentTypeError(
                "Binsplit separator cannot be an empty string"
            )
        self.binsplit_separator = binsplit_separator


class VambOptions:
    __slots__ = [
        "out_dir",
        "n_threads",
        "comp_options",
        "min_fasta_output_size",
        "cuda",
    ]

    def __init__(
        self,
        out_dir: Path,
        n_threads: int,
        comp_options: CompositionOptions,
        min_fasta_output_size: Optional[int],
        cuda: bool,
    ):
        assert isinstance(out_dir, Path)
        assert isinstance(n_threads, int)
        assert isinstance(comp_options, CompositionOptions)
        assert isinstance(min_fasta_output_size, (int, type(None)))
        assert isinstance(cuda, bool)

        # Outdir does not exist
        if out_dir.exists():
            raise FileExistsError(out_dir)

        # Outdir is in an existing parent dir
        if not out_dir.parent.is_dir():
            raise NotADirectoryError(parentdir)
        self.out_dir = out_dir

        if n_threads < 1:
            raise ValueError(f"Must pass at least 1 thread, not {n_threads}")
        self.n_threads = n_threads

        if min_fasta_output_size is not None:
            if not isinstance(comp_options.path, FASTAPath):
                raise argparse.ArgumentTypeError(
                    "If minfasta is not None, "
                    "input fasta file must be given explicitly"
                )
            if min_fasta_output_size < 0:
                raise argparse.ArgumentTypeError(
                    "Minimum FASTA output size must be nonnegative"
                )
        self.min_fasta_output_size = min_fasta_output_size

        if cuda and not torch.cuda.is_available():
            raise ModuleNotFoundError(
                "Cuda is not available on your PyTorch installation"
            )
        self.cuda = cuda


def log(string: str, logfile: IO[str], indent: int = 0):
    print(("\t" * indent) + string, file=logfile)
    logfile.flush()


def calc_tnf(
    options: CompositionOptions,
    outdir: Path,
    logfile: IO[str],
) -> vamb.parsecontigs.Composition:
    begintime = time.time()
    log("\nLoading TNF", logfile, 0)
    log(f"Minimum sequence length: {options.min_contig_length}", logfile, 1)

    path = options.path

    if isinstance(path, CompositionPath):
        log(f"Loading composition from npz {path}", logfile, 1)
        composition = vamb.parsecontigs.Composition.load(str(path))
        composition.filter_min_length(options.min_contig_length)
    else:
        assert isinstance(path, FASTAPath)
        log(f"Loading data from FASTA file {path}", logfile, 1)
        with vamb.vambtools.Reader(str(path)) as file:
            composition = vamb.parsecontigs.Composition.from_file(
                file, minlength=options.min_contig_length
            )
        composition.save(outdir.joinpath("composition.npz"))

    elapsed = round(time.time() - begintime, 2)
    print("", file=logfile)
    log(
        f"Kept {composition.count_bases()} bases in {composition.nseqs} sequences",
        logfile,
        1,
    )
    log(f"Processed TNF in {elapsed} seconds", logfile, 1)

    return composition


def calc_rpkm(
    abundance_options: AbundanceOptions,
    outdir: Path,
    comp_metadata: vamb.parsecontigs.CompositionMetaData,
    nthreads: int,
    logfile: IO[str],
) -> vamb.parsebam.Abundance:

    begintime = time.time()
    log("\nLoading depths", logfile)
    log(
        f'Reference hash: {comp_metadata.refhash.hex() if abundance_options.refcheck else "None"}',
        logfile,
        1,
    )

    path = abundance_options.path
    if isinstance(path, AbundancePath):
        log(f"Loading depths from npz array {str(path)}", logfile, 1)
        abundance = vamb.parsebam.Abundance.load(
            path, comp_metadata.refhash if abundance_options.refcheck else None
        )
        # I don't want this check in any constructors of abundance, since the constructors
        # should be able to skip this check in case comp and abundance are independent.
        # But when running the main Vamb workflow, we need to assert this.
        if abundance.nseqs != comp_metadata.nseqs:
            assert not abundance_options.refcheck
            raise ValueError(
                f"Loaded abundance has {abundance.nseqs} sequences, "
                f"but composition has {comp_metadata.nseqs}."
            )

    elif isinstance(path, JGIPath):
        log(f"Loading depths from JGI path {str(path)}", logfile, 1)
        with open(path) as file:
            abundance = vamb.parsebam.Abundance.from_jgi_filehandle(
                file, comp_metadata, abundance_options.refcheck
            )

    else:
        assert isinstance(path, list)
        log(f"Parsing {len(path)} BAM files with {nthreads} threads", logfile, 1)

        abundance = vamb.parsebam.Abundance.from_files(
            [str(i) for i in path],
            outdir.joinpath("tmp"),
            comp_metadata,
            abundance_options.refcheck,
            abundance_options.min_alignment_id,
            nthreads,
        )
        abundance.save(outdir.joinpath("abundance.npz"))

    log(f"Min identity: {abundance.minid}\n", logfile, 1)
    log("Order of columns is:", logfile, 1)
    log("\n\t".join(abundance.samplenames), logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    print("", file=logfile)
    log(f"Processed RPKM in {elapsed} seconds", logfile, 1)

    return abundance


def trainvae(
    vae_options: VAEOptions,
    training_options: TrainingOptions,
    vamb_options: VambOptions,
    outdir: Path,
    rpkms: np.ndarray,
    tnfs: np.ndarray,
    lengths: np.ndarray,
    logfile: IO[str],
) -> tuple[np.ndarray, np.ndarray]:

    begintime = time.time()
    log("\nCreating and training VAE", logfile)

    assert len(rpkms) == len(tnfs)

    nsamples = rpkms.shape[1]
    vae = vamb.encode.VAE(
        nsamples,
        nhiddens=vae_options.nhiddens,
        nlatent=vae_options.nlatent,
        alpha=vae_options.alpha,
        beta=vae_options.beta,
        dropout=vae_options.dropout,
        cuda=vamb_options.cuda,
    )

    log("Created VAE", logfile, 1)
    dataloader, mask = vamb.encode.make_dataloader(
        rpkms,
        tnfs,
        lengths,
        training_options.batchsize,
        destroy=True,
        cuda=vamb_options.cuda,
    )
    log("Created dataloader and mask", logfile, 1)
    vamb.vambtools.write_npz(os.path.join(outdir, "mask.npz"), mask)
    n_discarded = len(mask) - mask.sum()
    log(f"Number of sequences unsuitable for encoding: {n_discarded}", logfile, 1)
    log(f"Number of sequences remaining: {len(mask) - n_discarded}", logfile, 1)
    print("", file=logfile)

    modelpath = outdir.joinpath("model.pt")
    vae.trainmodel(
        dataloader,
        nepochs=training_options.nepochs,
        lrate=training_options.lrate,
        batchsteps=training_options.batchsteps,
        logfile=logfile,
        modelfile=modelpath,
    )

    print("", file=logfile)
    log("Encoding to latent representation", logfile, 1)
    latent = vae.encode(dataloader)
    vamb.vambtools.write_npz(os.path.join(outdir, "latent.npz"), latent)
    del vae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    log(f"Trained VAE and encoded in {elapsed} seconds", logfile, 1)

    return mask, latent


def cluster(
    cluster_options: ClusterOptions,
    clusterspath: Path,
    latent: np.ndarray,
    contignames: np.ndarray,  # of dtype object
    cuda: bool,
    logfile: IO[str],
) -> None:
    begintime = time.time()

    log("\nClustering", logfile)
    log(f"Windowsize: {cluster_options.window_size}", logfile, 1)
    log(
        f"Min successful thresholds detected: {cluster_options.min_successes}",
        logfile,
        1,
    )
    log(f"Max clusters: {cluster_options.max_clusters}", logfile, 1)
    log(f"Min cluster size: {cluster_options.min_cluster_size}", logfile, 1)
    log(f"Use CUDA for clustering: {cuda}", logfile, 1)
    log(
        "Separator: {}".format(
            None
            if cluster_options.binsplit_separator is None
            else ('"' + cluster_options.binsplit_separator + '"')
        ),
        logfile,
        1,
    )

    cluster_generator = vamb.cluster.ClusterGenerator(
        latent,
        windowsize=cluster_options.window_size,
        minsuccesses=cluster_options.min_successes,
        destroy=True,
        normalized=False,
        cuda=cuda,
    )

    renamed = (
        (str(cluster_index + 1), {contignames[i] for i in members})
        for (cluster_index, (_, members)) in enumerate(
            map(lambda x: x.as_tuple(), cluster_generator)
        )
    )

    # Binsplit if given a separator
    if cluster_options.binsplit_separator is not None:
        maybe_split = vamb.vambtools.binsplit(
            renamed, cluster_options.binsplit_separator
        )
    else:
        maybe_split = renamed

    with open(clusterspath, "w") as clustersfile:
        clusternumber, ncontigs = vamb.vambtools.write_clusters(
            clustersfile,
            maybe_split,
            max_clusters=cluster_options.max_clusters,
            min_size=cluster_options.min_cluster_size,
            rename=False,
        )

    print("", file=logfile)
    log(f"Clustered {ncontigs} contigs in {clusternumber} bins", logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    log(f"Clustered contigs in {elapsed} seconds", logfile, 1)


def write_fasta(
    outdir: Path,
    clusterspath: Path,
    fastapath: Path,
    contignames: np.ndarray,  # of object
    contiglengths: np.ndarray,
    minfasta: int,
    logfile: IO[str],
) -> None:
    begintime = time.time()

    log("\nWriting FASTA files", logfile)
    log("Minimum FASTA size: {minfasta}", logfile, 1)
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
            outdir.joinpath("bins"), filtered_clusters, file, maxbins=None
        )

    ncontigs = sum(map(len, filtered_clusters.values()))
    nfiles = len(filtered_clusters)
    print("", file=logfile)
    log(f"Wrote {ncontigs} contigs to {nfiles} FASTA files", logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    log(f"Wrote FASTA in {elapsed} seconds", logfile, 1)


def run(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    vae_options: VAEOptions,
    training_options: TrainingOptions,
    cluster_options: ClusterOptions,
    logfile: IO[str],
):

    log("Starting Vamb version " + ".".join(map(str, vamb.__version__)), logfile)
    log("Date and time is " + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()

    # Get TNFs, save as npz
    composition = calc_tnf(comp_options, vamb_options.out_dir, logfile)

    # Parse BAMs, save as npz
    abundance = calc_rpkm(
        abundance_options,
        vamb_options.out_dir,
        composition.metadata,
        vamb_options.n_threads,
        logfile,
    )

    # Train, save model
    mask, latent = trainvae(
        vae_options,
        training_options,
        vamb_options,
        vamb_options.out_dir,
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        logfile,
    )

    # Free up memory
    comp_metadata = composition.metadata
    del composition, abundance

    comp_metadata.filter_mask(mask)  # type: ignore
    assert comp_metadata.nseqs == len(latent)

    # Cluster, save tsv file
    clusterspath = vamb_options.out_dir.joinpath("clusters.tsv")
    cluster(
        cluster_options,
        clusterspath,
        latent,
        comp_metadata.identifiers,
        vamb_options.cuda,
        logfile,
    )

    del latent

    if vamb_options.min_fasta_output_size is not None:
        path = comp_options.path
        assert isinstance(path, FASTAPath)
        write_fasta(
            vamb_options.out_dir,
            clusterspath,
            path,
            comp_metadata.identifiers,
            comp_metadata.lengths,
            vamb_options.min_fasta_output_size,
            logfile,
        )

    elapsed = round(time.time() - begintime, 2)
    log(f"\nCompleted Vamb in {elapsed} seconds", logfile)


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
        add_help=False,
    )

    # Help
    helpos = parser.add_argument_group(title="Help and version", description=None)
    helpos.add_argument("-h", "--help", help="print help and exit", action="help")
    helpos.add_argument(
        "--version",
        action="version",
        version=f'Vamb {".".join(map(str, vamb.__version__))}',
    )

    # Positional arguments
    reqos = parser.add_argument_group(title="Output (required)", description=None)
    reqos.add_argument(
        "--outdir",
        metavar="",
        type=Path,
        required=True,
        help="output directory to create",
    )

    # TNF arguments
    tnfos = parser.add_argument_group(
        title="TNF input (either fasta or all .npz files required)"
    )
    tnfos.add_argument("--fasta", metavar="", type=Path, help="path to fasta file")
    tnfos.add_argument(
        "--composition", metavar="", type=Path, help="path to .npz of composition"
    )

    # RPKM arguments
    rpkmos = parser.add_argument_group(
        title="RPKM input (either BAMs or .npz required)"
    )
    rpkmos.add_argument(
        "--bamfiles",
        metavar="",
        dest="bampaths",
        type=Path,
        help="paths to (multiple) BAM files",
        nargs="+",
    )
    rpkmos.add_argument(
        "--rpkm",
        metavar="",
        dest="abundancepath",
        type=Path,
        help="path to .npz of RPKM (abundances)",
    )
    rpkmos.add_argument(
        "--jgi",
        metavar="",
        dest="jgipath",
        type=Path,
        help="path to JGI text file of abundances",
    )

    # Optional arguments
    inputos = parser.add_argument_group(title="IO options", description=None)

    inputos.add_argument(
        "-m",
        dest="min_contig_length",
        metavar="",
        type=int,
        default=250,
        help="ignore contigs shorter than this [250]",
    )
    inputos.add_argument(
        "-z",
        dest="min_alignment_id",
        metavar="",
        type=float,
        default=None,
        help="ignore reads with nucleotide identity below this [0.0]",
    )
    inputos.add_argument(
        "-p",
        dest="nthreads",
        metavar="",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "number of threads to use " "[min(" + str(DEFAULT_THREADS) + ", nbamfiles)]"
        ),
    )
    inputos.add_argument(
        "--norefcheck",
        help="skip reference name hashing check [False]",
        action="store_true",
    )
    inputos.add_argument(
        "--minfasta",
        dest="min_fasta_output_size",
        metavar="",
        type=int,
        default=None,
        help="minimum bin size to output as fasta [None = no files]",
    )

    # VAE arguments
    vaeos = parser.add_argument_group(title="VAE options", description=None)

    vaeos.add_argument(
        "-n",
        dest="nhiddens",
        metavar="",
        type=int,
        nargs="+",
        default=None,
        help="hidden neurons [Auto]",
    )
    vaeos.add_argument(
        "-l",
        dest="nlatent",
        metavar="",
        type=int,
        default=32,
        help="latent neurons [32]",
    )
    vaeos.add_argument(
        "-a",
        dest="alpha",
        metavar="",
        type=float,
        default=None,
        help="alpha, weight of TNF versus depth loss [Auto]",
    )
    vaeos.add_argument(
        "-b",
        dest="beta",
        metavar="",
        type=float,
        default=200.0,
        help="beta, capacity to learn [200.0]",
    )
    vaeos.add_argument(
        "-d",
        dest="dropout",
        metavar="",
        type=float,
        default=None,
        help="dropout [Auto]",
    )
    vaeos.add_argument(
        "--cuda", help="use GPU to train & cluster [False]", action="store_true"
    )

    trainos = parser.add_argument_group(title="Training options", description=None)

    trainos.add_argument(
        "-e", dest="nepochs", metavar="", type=int, default=300, help="epochs [300]"
    )
    trainos.add_argument(
        "-t",
        dest="batchsize",
        metavar="",
        type=int,
        default=256,
        help="starting batch size [256]",
    )
    trainos.add_argument(
        "-q",
        dest="batchsteps",
        metavar="",
        type=int,
        nargs="*",
        default=[25, 75, 150, 225],
        help="double batch size at epochs [25 75 150 225]",
    )
    trainos.add_argument(
        "-r",
        dest="lrate",
        metavar="",
        type=float,
        default=1e-3,
        help="learning rate [0.001]",
    )

    # Clustering arguments
    clusto = parser.add_argument_group(title="Clustering options", description=None)
    clusto.add_argument(
        "-w",
        dest="window_size",
        metavar="",
        type=int,
        default=200,
        help="size of window to count successes [200]",
    )
    clusto.add_argument(
        "-u",
        dest="min_successes",
        metavar="",
        type=int,
        default=20,
        help="minimum success in window [20]",
    )
    clusto.add_argument(
        "-i",
        dest="min_cluster_size",
        metavar="",
        type=int,
        default=1,
        help="minimum cluster size [1]",
    )
    clusto.add_argument(
        "-c",
        dest="max_clusters",
        metavar="",
        type=int,
        default=None,
        help="stop after c clusters [None = infinite]",
    )
    clusto.add_argument(
        "-o",
        dest="binsplit_separator",
        metavar="",
        type=str,
        default=None,
        help="binsplit separator [None = no split]",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
    comp_options = CompositionOptions(
        args.fasta, args.composition, args.min_contig_length
    )

    abundance_options = AbundanceOptions(
        args.bampaths,
        args.abundancepath,
        args.jgipath,
        args.min_alignment_id,
        not args.norefcheck,
    )

    vae_options = VAEOptions(
        args.nhiddens,
        args.nlatent,
        args.alpha,
        args.beta,
        args.dropout,
    )

    training_options = TrainingOptions(
        args.nepochs, args.batchsize, args.batchsteps, args.lrate
    )

    cluster_options = ClusterOptions(
        args.window_size,
        args.min_successes,
        args.min_cluster_size,
        args.max_clusters,
        args.binsplit_separator,
    )

    vamb_options = VambOptions(
        args.outdir,
        args.nthreads,
        comp_options,
        args.min_fasta_output_size,
        args.cuda
    )

    torch.set_num_threads(vamb_options.n_threads)

    try:
        os.mkdir(vamb_options.out_dir)
    except FileExistsError:
        pass
    except:
        raise

    with open(vamb_options.out_dir.joinpath("log.txt"), "w") as logfile:
        run(
            vamb_options,
            comp_options,
            abundance_options,
            vae_options,
            training_options,
            cluster_options,
            logfile=logfile,
        )


if __name__ == "__main__":
    main()
