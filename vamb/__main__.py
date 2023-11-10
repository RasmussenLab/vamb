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
import random
from math import isfinite
from typing import Optional, IO, Tuple, Union
from pathlib import Path
from collections.abc import Sequence
from collections import defaultdict
from torch.utils.data import DataLoader
import pandas as pd

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


def try_make_dir(name: Union[Path, str]):
    try:
        os.mkdir(name)
    except FileExistsError:
        pass


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


class AbundanceOptions:
    __slots__ = ["path", "min_alignment_id", "refcheck"]

    def __init__(
        self,
        bampaths: Optional[list[Path]],
        abundancepath: Optional[Path],
        min_alignment_id: Optional[float],
        refcheck: bool,
    ):
        assert isinstance(bampaths, (list, type(None)))
        assert isinstance(abundancepath, (Path, type(None)))
        assert isinstance(min_alignment_id, (float, type(None)))
        assert isinstance(refcheck, bool)

        # Make sure only one RPKM input is there
        if not (bampaths is not None) + (abundancepath is not None) == 1:
            raise argparse.ArgumentTypeError(
                "Must specify exactly one of BAM files or abundance NPZ file input"
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
    __slots__ = ["nhiddens", "nlatent", "beta", "dropout"]

    def __init__(
        self,
        nhiddens: Optional[list[int]],
        nlatent: int,
        beta: float,
        dropout: Optional[float],
    ):
        assert isinstance(nhiddens, (list, type(None)))
        assert isinstance(nlatent, int)

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

        if beta <= 0 or not isfinite(beta):
            raise argparse.ArgumentTypeError("beta cannot be negative or zero")
        self.beta = beta

        if dropout is not None and (dropout < 0 or dropout >= 1):
            raise argparse.ArgumentTypeError("dropout must be in 0 <= d < 1")
        self.dropout = dropout


class AAEOptions:
    def __init__(
        self,
        nhiddens: int,
        nlatent_z: int,
        nlatent_y: int,
        sl: float,
        slr: float,
    ):
        assert isinstance(nhiddens, int)
        assert isinstance(nlatent_z, int)
        assert isinstance(nlatent_y, int)
        assert isinstance(sl, float)
        assert isinstance(slr, float)

        self.nhiddens = nhiddens
        self.nlatent_z = nlatent_z
        self.nlatent_y = nlatent_y
        self.sl = sl
        self.slr = slr


class TaxonomyOptions:
    __slots__ = [
        "taxonomy_path",
        "taxonomy_predictions_path",
        "no_predictor",
    ]

    def __init__(
        self,
        taxonomy_path: Optional[Path],
        taxonomy_predictions_path: Optional[Path],
        no_predictor: bool,
    ):
        assert isinstance(taxonomy_path, (Path, type(None)))
        assert isinstance(taxonomy_predictions_path, (Path, type(None)))

        if taxonomy_path is None and taxonomy_predictions_path is None and no_predictor:
            raise argparse.ArgumentTypeError(
                "You must specify either --taxonomy_path or --taxonomy_predictions_path or run without --no_predictor flag"
            )

        if (
            taxonomy_path is None
            and taxonomy_predictions_path is None
            and not no_predictor
        ):
            raise argparse.ArgumentTypeError(
                "The taxonomy predictor needs --taxonomy_path for training"
            )

        self.taxonomy_path = taxonomy_path
        self.taxonomy_predictions_path = taxonomy_predictions_path
        self.no_predictor = no_predictor


class ReclusteringOptions:
    __slots__ = [
        "latent_path",
        "clusters_path",
        "hmmout_path",
        "binsplit_separator",
        "algorithm",
    ]

    def __init__(
        self,
        latent_path: Path,
        clusters_path: Path,
        hmmout_path: Path,
        binsplit_separator: Optional[str],
        algorithm: str,
    ):
        assert isinstance(latent_path, Path)
        assert isinstance(clusters_path, Path)
        assert isinstance(hmmout_path, Path)

        for path in [latent_path, clusters_path, hmmout_path]:
            if not path.is_file():
                raise FileNotFoundError(path)

        if algorithm not in ("kmeans", "dbscan"):
            raise ValueError("Algortihm must be 'kmeans' or 'dbscan'")

        self.latent_path = latent_path
        self.clusters_path = clusters_path
        self.binsplit_separator = binsplit_separator
        self.hmmout_path = hmmout_path
        self.algorithm = algorithm


class EncoderOptions:
    __slots__ = ["vae_options", "aae_options", "alpha"]

    def __init__(
        self,
        vae_options: Optional[VAEOptions],
        aae_options: Optional[AAEOptions],
        alpha: Optional[float],
    ):
        assert isinstance(alpha, (float, type(None)))
        assert isinstance(vae_options, (VAEOptions, type(None)))
        assert isinstance(aae_options, (AAEOptions, type(None)))

        if alpha is not None and (alpha <= 0 or alpha) >= 1:
            raise argparse.ArgumentTypeError("alpha must be above 0 and below 1")
        self.alpha = alpha

        if vae_options is None and aae_options is None:
            raise argparse.ArgumentTypeError(
                "You must specify either VAE or AAE or both as encoders, cannot run with no encoders."
            )

        self.vae_options = vae_options
        self.aae_options = aae_options


class VAETrainingOptions:
    def __init__(self, nepochs: int, batchsize: int, batchsteps: list[int]):
        assert isinstance(nepochs, int)
        assert isinstance(batchsize, int)
        assert isinstance(batchsteps, list)

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


class PredictorTrainingOptions(VAETrainingOptions):
    def __init__(
        self,
        nepochs: int,
        batchsize: int,
        batchsteps: list[int],
        softmax_threshold: float,
        ploss: str,
    ):
        losses = ["flat_softmax", "cond_softmax", "soft_margin", "random_cut"]
        if (softmax_threshold > 1) or (softmax_threshold < 0):
            raise argparse.ArgumentTypeError(
                f"Softmax threshold should be between 0 and 1, currently {softmax_threshold}"
            )
        if ploss not in losses:
            raise argparse.ArgumentTypeError(
                f"Predictor loss needs to be one of {losses}"
            )
        self.softmax_threshold = softmax_threshold
        self.ploss = ploss
        super(PredictorTrainingOptions, self).__init__(
            nepochs,
            batchsize,
            batchsteps,
        )


class AAETrainingOptions:
    def __init__(
        self,
        nepochs: int,
        batchsize: int,
        batchsteps: list[int],
        temp: Optional[float],
    ):
        assert isinstance(nepochs, int)
        assert isinstance(batchsize, int)
        assert isinstance(batchsteps, list)

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

        assert isinstance(temp, (float, type(None)))
        self.temp = temp


class TrainingOptions:
    def __init__(
        self,
        encoder_options: EncoderOptions,
        vae_options: Optional[VAETrainingOptions],
        aae_options: Optional[AAETrainingOptions],
        lrate: float,
    ):
        assert isinstance(lrate, float)

        assert (encoder_options.vae_options is None) == (vae_options is None)
        assert (encoder_options.aae_options is None) == (aae_options is None)

        if lrate <= 0.0:
            raise argparse.ArgumentTypeError("Learning rate must be positive")
        self.lrate = lrate

        self.vae_options = vae_options
        self.aae_options = aae_options
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
        "seed",
        "cuda",
    ]

    def __init__(
        self,
        out_dir: Path,
        n_threads: int,
        comp_options: CompositionOptions,
        min_fasta_output_size: Optional[int],
        seed: int,
        cuda: bool,
    ):
        assert isinstance(out_dir, Path)
        assert isinstance(n_threads, int)
        assert isinstance(comp_options, CompositionOptions)
        assert isinstance(min_fasta_output_size, (int, type(None)))
        assert isinstance(seed, int)
        assert isinstance(cuda, bool)

        # Outdir does not exist
        if out_dir.exists():
            raise FileExistsError(out_dir)

        # Outdir is in an existing parent dir
        parent_dir = out_dir.parent
        if not parent_dir.is_dir():
            raise NotADirectoryError(parent_dir)
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
        self.seed = seed
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
                file, minlength=options.min_contig_length, logfile=logfile
            )
        composition.save(outdir.joinpath("composition.npz"))

    elapsed = round(time.time() - begintime, 2)
    print("", file=logfile)
    log(
        f"Kept {composition.count_bases()} bases in {composition.nseqs} sequences",
        logfile,
        1,
    )
    log(f"Processed TNF in {elapsed} seconds.", logfile, 1)

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
        if hasattr(abundance, "nseqs") and abundance.nseqs != comp_metadata.nseqs:
            assert not abundance_options.refcheck
            raise ValueError(
                f"Loaded abundance has {abundance.nseqs} sequences, "
                f"but composition has {comp_metadata.nseqs}."
            )
    else:
        assert isinstance(path, list)
        log(f"Parsing {len(path)} BAM files with {nthreads} threads", logfile, 1)

        abundance = vamb.parsebam.Abundance.from_files(
            path,
            outdir.joinpath("tmp").joinpath("pycoverm"),
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
    log(f"Processed RPKM in {elapsed} seconds.", logfile, 1)

    return abundance


def trainvae(
    vae_options: VAEOptions,
    training_options: VAETrainingOptions,
    vamb_options: VambOptions,
    lrate: float,
    alpha: Optional[float],
    data_loader: DataLoader,
    logfile: IO[str],
) -> np.ndarray:
    begintime = time.time()
    log("\nCreating and training VAE", logfile)

    nsamples = data_loader.dataset.tensors[0].shape[1]  # type:ignore
    vae = vamb.encode.VAE(
        nsamples,
        nhiddens=vae_options.nhiddens,
        nlatent=vae_options.nlatent,
        alpha=alpha,
        beta=vae_options.beta,
        dropout=vae_options.dropout,
        cuda=vamb_options.cuda,
        seed=vamb_options.seed,
    )

    log("Created VAE", logfile, 1)
    modelpath = vamb_options.out_dir.joinpath("model.pt")
    vae.trainmodel(
        vamb.encode.set_batchsize(data_loader, training_options.batchsize),
        nepochs=training_options.nepochs,
        lrate=lrate,
        batchsteps=training_options.batchsteps,
        logfile=logfile,
        modelfile=modelpath,
    )

    print("", file=logfile)
    log("Encoding to latent representation", logfile, 1)
    latent = vae.encode(data_loader)
    vamb.vambtools.write_npz(vamb_options.out_dir.joinpath("latent.npz"), latent)
    del vae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    log(f"Trained VAE and encoded in {elapsed} seconds.", logfile, 1)

    return latent


def trainaae(
    data_loader: DataLoader,
    aae_options: AAEOptions,
    training_options: AAETrainingOptions,
    vamb_options: VambOptions,
    lrate: float,
    alpha: Optional[float],  # set automatically if None
    logfile: IO[str],
    contignames: Sequence[str],
) -> tuple[np.ndarray, dict[str, set[str]]]:
    begintime = time.time()
    log("\nCreating and training AAE", logfile)
    nsamples = data_loader.dataset.tensors[0].shape[1]  # type:ignore

    aae = vamb.aamb_encode.AAE(
        nsamples,
        aae_options.nhiddens,
        aae_options.nlatent_z,
        aae_options.nlatent_y,
        aae_options.sl,
        aae_options.slr,
        alpha,
        _cuda=vamb_options.cuda,
        seed=vamb_options.seed,
    )

    log("Created AAE", logfile, 1)
    modelpath = os.path.join(vamb_options.out_dir, "aae_model.pt")
    aae.trainmodel(
        vamb.encode.set_batchsize(data_loader, training_options.batchsize),
        training_options.nepochs,
        training_options.batchsteps,
        training_options.temp,
        lrate,
        logfile,
        modelpath,
    )

    print("", file=logfile)
    log("Encoding to latent representation", logfile, 1)
    clusters_y_dict, latent = aae.get_latents(contignames, data_loader)
    vamb.vambtools.write_npz(
        os.path.join(vamb_options.out_dir, "aae_z_latent.npz"), latent
    )

    del aae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    log(f"Trained AAE and encoded in {elapsed} seconds.", logfile, 1)

    return latent, clusters_y_dict


def cluster(
    cluster_options: ClusterOptions,
    clusterspath: Path,
    latent: np.ndarray,
    contignames: Sequence[str],  # of dtype object
    lengths: Sequence[int],  # of dtype object
    vamb_options: VambOptions,
    logfile: IO[str],
    cluster_prefix: str,
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
    log(f"Use CUDA for clustering: {vamb_options.cuda}", logfile, 1)
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
        lengths,  # type:ignore
        windowsize=cluster_options.window_size,
        minsuccesses=cluster_options.min_successes,
        destroy=True,
        normalized=False,
        # cuda=vamb_options.cuda,
        cuda=False,  # disabled until clustering is fixed
        rng_seed=vamb_options.seed,
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
            cluster_prefix=cluster_prefix,
        )

    print("", file=logfile)
    log(f"Clustered {ncontigs} contigs in {clusternumber} bins", logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    log(f"Clustered contigs in {elapsed} seconds.", logfile, 1)


def write_fasta(
    outdir: Path,
    clusterspath: Path,
    fastapath: Path,
    contignames: Sequence[str],  # of object
    contiglengths: np.ndarray,
    minfasta: int,
    logfile: IO[str],
    separator: Optional[str],
) -> None:
    begintime = time.time()

    log("\nWriting FASTA files", logfile)
    log("Minimum FASTA size: " + str(minfasta), logfile, 1)
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
            outdir.joinpath("bins"),
            filtered_clusters,
            file,
            maxbins=None,
            separator=separator,
        )

    ncontigs = sum(map(len, filtered_clusters.values()))
    nfiles = len(filtered_clusters)
    print("", file=logfile)
    log(f"Wrote {ncontigs} contigs to {nfiles} FASTA files", logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    log(f"Wrote FASTA in {elapsed} seconds.", logfile, 1)


def load_composition_and_abundance(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    logfile: IO[str],
) -> Tuple[vamb.parsecontigs.Composition, vamb.parsebam.Abundance]:
    log("Starting Vamb version " + ".".join(map(str, vamb.__version__)), logfile)
    log("Date and time is " + str(datetime.datetime.now()), logfile, 1)
    log("Random seed is " + str(vamb_options.seed), logfile, 1)
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
    time_generating_input = round(time.time() - begintime, 2)
    log(
        f"\nTNF and coabundances generated in {time_generating_input} seconds.",
        logfile,
        1,
    )
    return (composition, abundance)


def run(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    encoder_options: EncoderOptions,
    training_options: TrainingOptions,
    cluster_options: ClusterOptions,
    logfile: IO[str],
):
    vae_options = encoder_options.vae_options
    aae_options = encoder_options.aae_options
    vae_training_options = training_options.vae_options
    aae_training_options = training_options.aae_options

    begintime = time.time()

    composition, abundance = load_composition_and_abundance(
        vamb_options=vamb_options,
        comp_options=comp_options,
        abundance_options=abundance_options,
        logfile=logfile,
    )
    data_loader = vamb.encode.make_dataloader(
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        256,  # dummy value - we change this before using the actual loader
        destroy=True,
        cuda=vamb_options.cuda,
    )

    latent = None
    if vae_options is not None:
        assert vae_training_options is not None
        # Train, save model
        latent = trainvae(
            vae_options=vae_options,
            training_options=vae_training_options,
            vamb_options=vamb_options,
            lrate=training_options.lrate,
            alpha=encoder_options.alpha,
            data_loader=data_loader,
            logfile=logfile,
        )
        print("", file=logfile)

    clusters_y_dict = None
    latent_z = None
    if aae_options is not None:
        assert aae_training_options is not None
        # Train, save model
        latent_z, clusters_y_dict = trainaae(
            data_loader=data_loader,
            aae_options=aae_options,
            vamb_options=vamb_options,
            training_options=aae_training_options,
            lrate=training_options.lrate,
            alpha=encoder_options.alpha,
            logfile=logfile,
            contignames=composition.metadata.identifiers,  # type:ignore
        )
        print("", file=logfile)

    # Free up memory
    comp_metadata = composition.metadata
    del composition, abundance

    # Write contignames and contiglengths needed for dereplication purposes
    np.savetxt(
        vamb_options.out_dir.joinpath("contignames"),
        comp_metadata.identifiers,
        fmt="%s",
    )
    np.savez(vamb_options.out_dir.joinpath("lengths.npz"), comp_metadata.lengths)

    if vae_options is not None:
        assert latent is not None
        assert comp_metadata.nseqs == len(latent)
        # Cluster, save tsv file
        clusterspath = vamb_options.out_dir.joinpath("vae_clusters.tsv")
        cluster(
            cluster_options,
            clusterspath,
            latent,
            comp_metadata.identifiers,  # type:ignore
            comp_metadata.lengths,  # type:ignore
            vamb_options,
            logfile,
            "vae_",
        )
        log("VAE latent clustered", logfile, 1)

        del latent
        fin_cluster_latent = time.time()

        if vamb_options.min_fasta_output_size is not None:
            path = comp_options.path
            assert isinstance(path, FASTAPath)
            write_fasta(
                vamb_options.out_dir,
                clusterspath,
                path,
                comp_metadata.identifiers,  # type:ignore
                comp_metadata.lengths,
                vamb_options.min_fasta_output_size,
                logfile,
                separator=cluster_options.binsplit_separator,
            )

        writing_bins_time = round(time.time() - fin_cluster_latent, 2)
        log(f"VAE bins written in {writing_bins_time} seconds.", logfile, 1)

    if aae_options is not None:
        assert latent_z is not None
        assert clusters_y_dict is not None
        assert comp_metadata.nseqs == len(latent_z)
        # Cluster, save tsv file
        clusterspath = vamb_options.out_dir.joinpath("aae_z_clusters.tsv")

        cluster(
            cluster_options,
            clusterspath,
            latent_z,
            comp_metadata.identifiers,  # type:ignore
            comp_metadata.lengths,  # type:ignore
            vamb_options,
            logfile,
            "aae_z_",
        )

        fin_cluster_latent_z = time.time()
        log("AAE z latent clustered.", logfile, 1)

        del latent_z

        if vamb_options.min_fasta_output_size is not None:
            path = comp_options.path
            assert isinstance(path, FASTAPath)
            write_fasta(
                vamb_options.out_dir,
                clusterspath,
                path,
                comp_metadata.identifiers,  # type:ignore
                comp_metadata.lengths,
                vamb_options.min_fasta_output_size,
                logfile,
                separator=cluster_options.binsplit_separator,
            )
        time_writing_bins_z = time.time()
        writing_bins_time_z = round(time_writing_bins_z - fin_cluster_latent_z, 2)
        log(f"AAE z bins written in {writing_bins_time_z} seconds.", logfile, 1)

        clusterspath = vamb_options.out_dir.joinpath("aae_y_clusters.tsv")
        # Binsplit if given a separator
        if cluster_options.binsplit_separator is not None:
            maybe_split = vamb.vambtools.binsplit(
                clusters_y_dict.items(), cluster_options.binsplit_separator
            )
        else:
            maybe_split = clusters_y_dict.items()
        with open(clusterspath, "w") as clustersfile:
            clusternumber, ncontigs = vamb.vambtools.write_clusters(
                clustersfile,
                maybe_split,
                max_clusters=cluster_options.max_clusters,
                min_size=cluster_options.min_cluster_size,
                rename=False,
                cluster_prefix="aae_y_",
            )

        print("", file=logfile)
        log(f"Clustered {ncontigs} contigs in {clusternumber} bins", logfile, 1)
        time_start_writin_z_bins = time.time()
        if vamb_options.min_fasta_output_size is not None:
            path = comp_options.path
            assert isinstance(path, FASTAPath)
            write_fasta(
                vamb_options.out_dir,
                clusterspath,
                path,
                comp_metadata.identifiers,  # type:ignore
                comp_metadata.lengths,
                vamb_options.min_fasta_output_size,
                logfile,
                separator=cluster_options.binsplit_separator,
            )
        time_writing_bins_y = time.time()
        writing_bins_time_y = round(time_writing_bins_y - time_start_writin_z_bins, 2)
        log(f"AAE y bins written in {writing_bins_time_y} seconds.", logfile, 1)

    log(f"\nCompleted Vamb in {round(time.time() - begintime, 2)} seconds.", logfile, 0)


def parse_mmseqs_taxonomy(
    taxonomy_path: Path,
    contignames: list[str],  # already masked
    logfile: IO[str],
) -> pd.Series:
    df_mmseq = pd.read_csv(taxonomy_path, delimiter="\t", header=None)
    assert (
        len(df_mmseq.columns) >= 9
    ), f"Too few columns ({len(df_mmseq.columns)}) in mmseqs taxonomy file"
    df_mmseq = df_mmseq[~(df_mmseq[2] == "no rank")]
    log(f"{len(df_mmseq)} lines in taxonomy file", logfile, 1)
    log(f"{len(contignames)} contigs", logfile, 1)
    df_mmseq = df_mmseq[df_mmseq[0].isin(contignames)]

    # TODO: rethink when we start working with other domains
    missing_contigs = set(contignames) - set(df_mmseq[0])
    for c in missing_contigs:
        new_row: dict[int, Union[str, float]] = {i: np.nan for i in range(9)}
        new_row[0] = c
        new_row[2] = "domain"
        new_row[8] = "d_Bacteria"
        df_mmseq = pd.concat([df_mmseq, pd.DataFrame([new_row])], ignore_index=True)

    df_mmseq.set_index(0, inplace=True)
    df_mmseq = df_mmseq.loc[contignames]
    df_mmseq.reset_index(inplace=True)
    graph_column = df_mmseq[8]

    if list(df_mmseq[0]) != list(contignames):
        raise AssertionError(
            "The contig names of taxonomy entries are not the same as in the contigs metadata"
        )

    return graph_column


def predict_taxonomy(
    rpkms: np.ndarray,
    tnfs: np.ndarray,
    lengths: np.ndarray,
    contignames: np.ndarray,
    taxonomy_path: Path,
    out_dir: Path,
    predictor_training_options: PredictorTrainingOptions,
    cuda: bool,
    logfile: IO[str],
):
    begintime = time.time()

    graph_column = parse_mmseqs_taxonomy(
        taxonomy_path=taxonomy_path,
        contignames=contignames,
        logfile=logfile,
    )

    nodes, ind_nodes, table_parent = vamb.h_loss.make_graph(graph_column.unique())
    log(f"{len(nodes)} nodes in the graph", logfile, 1)

    classes_order = np.array(list(graph_column.str.split(";").str[-1]))
    targets = np.array([ind_nodes[i] for i in classes_order])

    model = vamb.h_loss.VAMB2Label(
        rpkms.shape[1],
        len(nodes),
        nodes,
        table_parent,
        nhiddens=[512, 512, 512, 512],
        hier_loss=predictor_training_options.ploss,
        cuda=cuda,
    )

    dataloader_vamb = vamb.encode.make_dataloader(
        rpkms,
        tnfs,
        lengths,
        batchsize=predictor_training_options.batchsize,
    )
    dataloader_joint = vamb.h_loss.make_dataloader_concat_hloss(
        rpkms,
        tnfs,
        lengths,
        targets,
        len(nodes),
        table_parent,
        batchsize=predictor_training_options.batchsize,
    )

    print("", file=logfile)
    log("Created dataloader", logfile, 0)

    predictortime = time.time()
    log(
        f"Starting training the taxonomy predictor at {str(datetime.datetime.now())}",
        logfile,
        0,
    )
    log(f"Using threshold {predictor_training_options.softmax_threshold}", logfile, 0)

    model_path = out_dir.joinpath("predictor_model.pt")
    with open(model_path, "wb") as modelfile:
        model.trainmodel(
            dataloader_joint,
            nepochs=predictor_training_options.nepochs,
            modelfile=modelfile,
            logfile=logfile,
            batchsteps=predictor_training_options.batchsteps,
        )
    log(
        f"Finished training the taxonomy predictor in {round(time.time() - predictortime, 2)} seconds.",
        logfile,
        0,
    )

    log("Writing the taxonomy predictions", logfile, 0)
    predicted_path = out_dir.joinpath("results_taxonomy_predictor.csv")
    nodes_ar = np.array(nodes)

    log(f"Using threshold {predictor_training_options.softmax_threshold}", logfile, 0)
    row = 0
    for predicted_vector, predicted_labels in model.predict(dataloader_vamb):
        N = len(predicted_vector)
        predictions: list[str] = []
        probs: list[float] = []
        for i in range(predicted_vector.shape[0]):
            label = predicted_labels[i]
            pred_labels: list[str] = [label]
            while table_parent[label] != -1:
                pred_labels.append(table_parent[label])
                label = table_parent[label]
            pred_labels = ";".join([nodes_ar[label] for label in pred_labels][::-1])
            threshold_mask = (
                predicted_vector[i] > predictor_training_options.softmax_threshold
            )
            pred_line = ";".join(nodes_ar[threshold_mask][1:])
            predictions.append(pred_line)
            absolute_probs = predicted_vector[i][threshold_mask]
            absolute_prob = ";".join(map(str, absolute_probs[1:]))
            probs.append(absolute_prob)
        df_gt = pd.DataFrame(
            {"contigs": contignames[row : row + N], "lengths": lengths[row : row + N]}
        )
        df_gt["predictions"] = predictions
        df_gt["probabilities"] = probs
        df_gt.to_csv(
            predicted_path,
            mode="a",
            index=None,
            header=not os.path.exists(predicted_path),
        )
        row += N

    log(
        f"\nCompleted taxonomy predictions in {round(time.time() - begintime, 2)} seconds.",
        logfile,
        0,
    )


def extract_and_filter_data(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    logfile: IO[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    composition, abundance = load_composition_and_abundance(
        vamb_options=vamb_options,
        comp_options=comp_options,
        abundance_options=abundance_options,
        logfile=logfile,
    )

    log(f"{len(composition.metadata.identifiers)} contig names", logfile, 0)

    return (
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        composition.metadata.identifiers,
    )


def run_taxonomy_predictor(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    predictor_training_options: PredictorTrainingOptions,
    taxonomy_options: TaxonomyOptions,
    logfile: IO[str],
):
    rpkms, tnfs, lengths, contignames = extract_and_filter_data(
        vamb_options=vamb_options,
        comp_options=comp_options,
        abundance_options=abundance_options,
        logfile=logfile,
    )
    assert taxonomy_options.taxonomy_path is not None
    predict_taxonomy(
        rpkms=rpkms,
        tnfs=tnfs,
        lengths=lengths,
        contignames=contignames,
        taxonomy_path=taxonomy_options.taxonomy_path,
        out_dir=vamb_options.out_dir,
        predictor_training_options=predictor_training_options,
        cuda=vamb_options.cuda,
        logfile=logfile,
    )


def run_vaevae(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    vae_training_options: VAETrainingOptions,
    predictor_training_options: PredictorTrainingOptions,
    taxonomy_options: TaxonomyOptions,
    cluster_options: ClusterOptions,
    encoder_options: EncoderOptions,
    logfile: IO[str],
):
    vae_options = encoder_options.vae_options
    rpkms, tnfs, lengths, contignames = extract_and_filter_data(
        vamb_options=vamb_options,
        comp_options=comp_options,
        abundance_options=abundance_options,
        logfile=logfile,
    )

    if taxonomy_options.taxonomy_path is not None and not taxonomy_options.no_predictor:
        log("Predicting missing values from mmseqs taxonomy", logfile, 0)
        predict_taxonomy(
            rpkms=rpkms,
            tnfs=tnfs,
            lengths=lengths,
            contignames=contignames,
            taxonomy_path=taxonomy_options.taxonomy_path,
            out_dir=vamb_options.out_dir,
            predictor_training_options=predictor_training_options,
            cuda=vamb_options.cuda,
            logfile=logfile,
        )
        predictions_path = vamb_options.out_dir.joinpath(
            "results_taxonomy_predictor.csv"
        )
    elif taxonomy_options.taxonomy_predictions_path is not None:
        log("mmseqs taxonomy predictions are provided", logfile, 0)
        predictions_path = taxonomy_options.taxonomy_predictions_path
    else:
        log("Not predicting the taxonomy", logfile, 0)
        predictions_path = None

    if predictions_path is not None:
        df_gt = pd.read_csv(predictions_path)
        df_gt.loc[
            df_gt["predictions"].isna(), "predictions"
        ] = "d_Bacteria"  # TODO: it's a hack
        graph_column = df_gt["predictions"]
    elif taxonomy_options.no_predictor:
        log("Using mmseqs taxonomy for semisupervised learning", logfile, 0)
        assert taxonomy_options.taxonomy_path is not None
        graph_column = parse_mmseqs_taxonomy(
            taxonomy_path=taxonomy_options.taxonomy_path,
            contignames=contignames,  # type:ignore
            logfile=logfile,
        )
    else:
        raise argparse.ArgumentTypeError("One of the taxonomy arguments is missing")

    nodes, ind_nodes, table_parent = vamb.h_loss.make_graph(graph_column.unique())

    classes_order = np.array(list(graph_column.str.split(";").str[-1]))
    targets = np.array([ind_nodes[i] for i in classes_order])

    assert vae_options is not None
    vae = vamb.h_loss.VAEVAEHLoss(
        rpkms.shape[1],
        len(nodes),
        nodes,
        table_parent,
        nhiddens=vae_options.nhiddens,
        nlatent=vae_options.nlatent,
        alpha=encoder_options.alpha,
        beta=vae_options.beta,
        dropout=vae_options.dropout,
        cuda=vamb_options.cuda,
        logfile=logfile,
    )

    dataloader_vamb = vamb.encode.make_dataloader(
        rpkms,
        tnfs,
        lengths,
        batchsize=vae_training_options.batchsize,
        cuda=vamb_options.cuda,
    )
    dataloader_joint = vamb.h_loss.make_dataloader_concat_hloss(
        rpkms,
        tnfs,
        lengths,
        targets,
        len(nodes),
        table_parent,
        batchsize=vae_training_options.batchsize,
        cuda=vamb_options.cuda,
    )
    dataloader_labels = vamb.h_loss.make_dataloader_labels_hloss(
        rpkms,
        tnfs,
        lengths,
        targets,
        len(nodes),
        table_parent,
        batchsize=vae_training_options.batchsize,
        cuda=vamb_options.cuda,
    )

    shapes = (rpkms.shape[1], 103, 1, len(nodes))
    dataloader = vamb.h_loss.make_dataloader_semisupervised_hloss(
        dataloader_joint,
        dataloader_vamb,
        dataloader_labels,
        len(nodes),
        table_parent,
        shapes,
        vamb_options.seed,
        batchsize=vae_training_options.batchsize,
        cuda=vamb_options.cuda,
    )
    model_path = vamb_options.out_dir.joinpath("vaevae_model.pt")
    with open(model_path, "wb") as modelfile:
        vae.trainmodel(
            dataloader,
            nepochs=vae_training_options.nepochs,
            modelfile=modelfile,
            logfile=logfile,
            batchsteps=vae_training_options.batchsteps,
        )

    latent_both = vae.VAEJoint.encode(dataloader_joint)
    log(
        f"{latent_both.shape} embedding shape",
        logfile,
        0,
    )

    LATENT_PATH = vamb_options.out_dir.joinpath("vaevae_latent.npy")
    np.save(LATENT_PATH, latent_both)

    # Cluster, save tsv file
    clusterspath = vamb_options.out_dir.joinpath("vaevae_clusters.tsv")
    cluster(
        cluster_options,
        clusterspath,
        latent_both,
        contignames,
        lengths,
        vamb_options,
        logfile,
        "vaevae_",
    )
    log("VAEVAE latent clustered", logfile, 1)

    del latent_both
    fin_cluster_latent = time.time()

    if vamb_options.min_fasta_output_size is not None:
        path = comp_options.path
        assert isinstance(path, FASTAPath)
        write_fasta(
            vamb_options.out_dir,
            clusterspath,
            path,
            contignames,
            lengths,
            vamb_options.min_fasta_output_size,
            logfile,
            separator=cluster_options.binsplit_separator,
        )

    writing_bins_time = round(time.time() - fin_cluster_latent, 2)
    log(f"VAEVAE bins written in {writing_bins_time} seconds.", logfile, 1)


def run_reclustering(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    reclustering_options: ReclusteringOptions,
    taxonomy_options: TaxonomyOptions,
    logfile: IO[str],
):
    composition, _ = load_composition_and_abundance(
        vamb_options=vamb_options,
        comp_options=comp_options,
        abundance_options=abundance_options,
        logfile=logfile,
    )

    log(
        f"{len(composition.metadata.identifiers)} contig names after filtering",
        logfile,
        0,
    )
    log("mmseqs taxonomy predictions are provided", logfile, 0)
    predictions_path = taxonomy_options.taxonomy_predictions_path

    reclustered = vamb.reclustering.recluster_bins(
        logfile,
        reclustering_options.clusters_path,
        reclustering_options.latent_path,
        str(comp_options.path),
        composition.metadata.identifiers,
        out_dir=vamb_options.out_dir,
        minfasta=0,
        binned_length=1000,
        num_process=40,
        random_seed=int(random.uniform(0, 2**32 - 1)),
        hmmout_path=reclustering_options.hmmout_path,
        algorithm=reclustering_options.algorithm,
        predictions_path=predictions_path,
    )

    cluster_dict = defaultdict(set)
    for k, v in zip(reclustered, composition.metadata.identifiers):
        cluster_dict[k].add(v)
    clusterspath = vamb_options.out_dir.joinpath("clusters_reclustered.tsv")
    if reclustering_options.binsplit_separator is not None:
        maybe_split = vamb.vambtools.binsplit(
            cluster_dict.items(), reclustering_options.binsplit_separator
        )
    else:
        maybe_split = cluster_dict.items()
    with open(clusterspath, "w") as clustersfile:
        clusternumber, ncontigs = vamb.vambtools.write_clusters(
            clustersfile,
            maybe_split,
            rename=False,
            cluster_prefix="recluster",
        )

    print("", file=logfile)
    log(f"Clustered {ncontigs} contigs in {clusternumber} bins", logfile, 1)

    fin_cluster_latent = time.time()

    if vamb_options.min_fasta_output_size is not None:
        path = comp_options.path
        assert isinstance(path, FASTAPath)
        write_fasta(
            vamb_options.out_dir,
            clusterspath,
            path,
            composition.metadata.identifiers,
            composition.metadata.lengths,
            vamb_options.min_fasta_output_size,
            logfile,
            separator=reclustering_options.binsplit_separator,
        )

    writing_bins_time = round(time.time() - fin_cluster_latent, 2)
    log(f"Reclustered bins written in {writing_bins_time} seconds.", logfile, 1)


class BasicArguments(object):
    LOGS_PATH = "log.txt"

    def __init__(self, args):
        self.args = args
        self.comp_options = CompositionOptions(
            self.args.fasta, self.args.composition, self.args.minlength
        )
        self.abundance_options = AbundanceOptions(
            self.args.bampaths,
            self.args.abundancepath,
            self.args.min_alignment_id,
            not self.args.norefcheck,
        )
        self.vamb_options = VambOptions(
            self.args.outdir,
            self.args.nthreads,
            self.comp_options,
            self.args.min_fasta_output_size,
            self.args.seed,
            self.args.cuda,
        )

    def run_inner(self, logfile):
        raise NotImplementedError

    def run(self):
        torch.set_num_threads(self.vamb_options.n_threads)
        try_make_dir(self.vamb_options.out_dir)
        with open(self.vamb_options.out_dir.joinpath(self.LOGS_PATH), "w") as logfile:
            self.run_inner(logfile)


class TaxometerArguments(BasicArguments):
    def __init__(self, args):
        super(TaxometerArguments, self).__init__(args)
        self.taxonomy_options = TaxonomyOptions(
            taxonomy_predictions_path=None,
            taxonomy_path=args.taxonomy,
            no_predictor=None,
        )
        self.predictor_training_options = PredictorTrainingOptions(
            nepochs=args.pred_nepochs,
            batchsize=args.pred_batchsize,
            batchsteps=args.pred_batchsteps,
            softmax_threshold=args.pred_softmax_threshold,
            ploss=args.ploss,
        )

    def run_inner(self, logfile):
        run_taxonomy_predictor(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            taxonomy_options=self.taxonomy_options,
            predictor_training_options=self.predictor_training_options,
            logfile=logfile,
        )


class BinnerArguments(BasicArguments):
    def __init__(self, args):
        super(BinnerArguments, self).__init__(args)
        self.vae_options = VAEOptions(
            nhiddens=args.nhiddens,
            nlatent=args.nlatent,
            beta=args.beta,
            dropout=args.dropout,
        )
        self.vae_training_options = VAETrainingOptions(
            nepochs=args.nepochs,
            batchsize=args.batchsize,
            batchsteps=args.batchsteps,
        )
        self.aae_options = None
        self.aae_training_options = None
        self.cluster_options = ClusterOptions(
            args.window_size,
            args.min_successes,
            args.min_cluster_size,
            args.max_clusters,
            args.binsplit_separator,
        )


class VAEAAEArguments(BinnerArguments):
    def __init__(self, args):
        super(VAEAAEArguments, self).__init__(args)

        # For the VAE-AAE workflow, we must cluster the full set of sequences,
        # else the cluster dereplication step makes no sense, as the different
        # clusterings will contain different sets of clusters.
        # So, enforce this here
        if self.cluster_options.max_clusters is not None:
            raise ValueError(
                "When using the VAE-AAE model, `max_clusters` (option `-c`) "
                "must not be set."
            )

        self.aae_options = AAEOptions(
            nhiddens=args.nhiddens_aae,
            nlatent_z=args.nlatent_aae_z,
            nlatent_y=args.nlatent_aae_y,
            sl=args.sl,
            slr=args.slr,
        )
        self.aae_training_options = AAETrainingOptions(
            nepochs=args.nepochs_aae,
            batchsize=args.batchsize_aae,
            batchsteps=args.batchsteps_aae,
            temp=args.temp,
        )

    def init_encoder_and_training(self):
        self.encoder_options = EncoderOptions(
            vae_options=self.vae_options,
            aae_options=self.aae_options,
            alpha=self.args.alpha,
        )
        self.training_options = TrainingOptions(
            encoder_options=self.encoder_options,
            vae_options=self.vae_training_options,
            aae_options=self.aae_training_options,
            lrate=self.args.lrate,
        )

    def run_inner(self, logfile: Optional[IO[str]]):
        run(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            encoder_options=self.encoder_options,
            training_options=self.training_options,
            cluster_options=self.cluster_options,
            logfile=logfile,
        )


class VAEArguments(BinnerArguments):
    def __init__(self, args):
        super(VAEArguments, self).__init__(args)
        self.encoder_options = EncoderOptions(
            vae_options=self.vae_options,
            aae_options=None,
            alpha=self.args.alpha,
        )
        self.training_options = TrainingOptions(
            encoder_options=self.encoder_options,
            vae_options=self.vae_training_options,
            aae_options=None,
            lrate=self.args.lrate,
        )

    def run_inner(self, logfile: Optional[IO[str]]):
        run(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            encoder_options=self.encoder_options,
            training_options=self.training_options,
            cluster_options=self.cluster_options,
            logfile=logfile,
        )


class VAEVAEArguments(BinnerArguments):
    def __init__(self, args):
        super(VAEVAEArguments, self).__init__(args)
        self.encoder_options = EncoderOptions(
            vae_options=self.vae_options,
            aae_options=None,
            alpha=self.args.alpha,
        )
        self.training_options = TrainingOptions(
            encoder_options=self.encoder_options,
            vae_options=self.vae_training_options,
            aae_options=None,
            lrate=self.args.lrate,
        )
        self.taxonomy_options = TaxonomyOptions(
            taxonomy_path=self.args.taxonomy,
            taxonomy_predictions_path=self.args.taxonomy_predictions,
            no_predictor=self.args.no_predictor,
        )
        self.predictor_training_options = PredictorTrainingOptions(
            nepochs=self.args.pred_nepochs,
            batchsize=self.args.pred_batchsize,
            batchsteps=self.args.pred_batchsteps,
            softmax_threshold=self.args.pred_softmax_threshold,
            ploss=self.args.ploss,
        )

    def run_inner(self, logfile: Optional[IO[str]]):
        run_vaevae(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            taxonomy_options=self.taxonomy_options,
            vae_training_options=self.vae_training_options,
            predictor_training_options=self.predictor_training_options,
            cluster_options=self.cluster_options,
            encoder_options=self.encoder_options,
            logfile=logfile,
        )


class ReclusteringArguments(BasicArguments):
    def __init__(self, args):
        super(ReclusteringArguments, self).__init__(args)
        self.reclustering_options = ReclusteringOptions(
            latent_path=args.latent_path,
            clusters_path=args.clusters_path,
            binsplit_separator=args.binsplit_separator_recluster,
            hmmout_path=args.hmmout_path,
            algorithm=args.algorithm,
        )
        if self.args.algorithm == "dbscan":
            self.preds = self.args.taxonomy_predictions
        else:
            self.preds = self.args.clusters_path
        self.taxonomy_options = TaxonomyOptions(
            taxonomy_path=None,
            taxonomy_predictions_path=self.preds,
            no_predictor=None,
        )

    def run_inner(self, logfile: Optional[IO[str]]):
        run_reclustering(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            reclustering_options=self.reclustering_options,
            taxonomy_options=self.taxonomy_options,
            logfile=logfile,
        )


def add_input_output_arguments(subparser):
    """Add TNFs and RPKMs arguments to a subparser."""
    # TNF arguments
    tnfos = subparser.add_argument_group(
        title="TNF input (either fasta or all .npz files required)"
    )
    tnfos.add_argument("--fasta", metavar="", type=Path, help="path to fasta file")
    tnfos.add_argument(
        "--composition", metavar="", type=Path, help="path to .npz of composition"
    )

    # RPKM arguments
    rpkmos = subparser.add_argument_group(
        title="RPKM input (either BAMs or .npz required)"
    )
    rpkmos.add_argument(
        "--bamfiles",
        dest="bampaths",
        metavar="",
        type=Path,
        help="paths to (multiple) BAM files",
        nargs="+",
    )
    rpkmos.add_argument(
        "--rpkm",
        metavar="",
        dest="abundancepath",
        type=Path,
        help="path to .npz of RPKM abundances",
    )

    reqos = subparser.add_argument_group(title="Output (required)", description=None)
    reqos.add_argument(
        "--outdir",
        metavar="",
        type=Path,
        # required=True, # Setting this flag breaks the argparse with positional arguments. The existence of the folder is checked elsewhere
        help="output directory to create",
    )

    # # Optional arguments
    inputos = subparser.add_argument_group(title="IO options", description=None)
    inputos.add_argument(
        "-m",
        dest="minlength",
        metavar="",
        type=int,
        default=2000,
        help="ignore contigs shorter than this [2000]",
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
    inputos.add_argument(
        "--cuda", help="use GPU to train & cluster [False]", action="store_true"
    )

    # Other arguments
    otheros = subparser.add_argument_group(title="Other options", description=None)
    otheros.add_argument(
        "--seed",
        metavar="",
        type=int,
        default=int.from_bytes(os.urandom(8), "little"),
        help="Random seed random determinism not guaranteed",
    )
    return subparser


def add_vae_arguments(subparser):
    # VAE arguments
    vaeos = subparser.add_argument_group(title="VAE options", description=None)

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

    trainos = subparser.add_argument_group(title="Training options", description=None)

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
    return subparser


def add_predictor_arguments(subparser):
    # Train predictor arguments
    pred_trainos = subparser.add_argument_group(
        title="Training options for the taxonomy predictor", description=None
    )

    pred_trainos.add_argument(
        "-pe",
        dest="pred_nepochs",
        metavar="",
        type=int,
        default=100,
        help="taxonomy predictor epochs [100]",
    )
    pred_trainos.add_argument(
        "-pt",
        dest="pred_batchsize",
        metavar="",
        type=int,
        default=256,
        help="starting batch size for the taxonomy predictor [256]",
    )
    pred_trainos.add_argument(
        "-pq",
        dest="pred_batchsteps",
        metavar="",
        type=int,
        nargs="*",
        default=[],
        help="double batch size at epochs for the taxonomy predictor [25 75]",
    )
    pred_trainos.add_argument(
        "-pr",
        dest="pred_lrate",
        metavar="",
        type=float,
        default=1e-3,
        help="learning rate for the taxonomy predictor [0.001]",
    )
    pred_trainos.add_argument(
        "-pthr",
        dest="pred_softmax_threshold",
        metavar="",
        type=float,
        default=0.5,
        help="conditional probability threshold for accepting the taxonomic prediction [0.5]",
    )
    pred_trainos.add_argument(
        "-ploss",
        dest="ploss",
        metavar="",
        type=str,
        default="flat_softmax",
        help='Hierarchical loss one of flat_softmax, cond_softmax, soft_margin, random_cut ["flat_softmax"]',
    )
    return subparser


def add_clustering_arguments(subparser):
    # Clustering arguments
    clusto = subparser.add_argument_group(title="Clustering options", description=None)
    clusto.add_argument(
        "-w",
        dest="window_size",
        metavar="",
        type=int,
        default=300,
        help="size of window to count successes [300]",
    )
    clusto.add_argument(
        "-u",
        dest="min_successes",
        metavar="",
        type=int,
        default=15,
        help="minimum success in window [15]",
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
    return subparser


def add_model_selection_arguments(subparser):
    # Model selection argument
    model_selection = subparser.add_argument_group(
        title="Model selection", description=None
    )

    model_selection.add_argument(
        "--model",
        dest="model",
        metavar="",
        type=str,
        choices=[
            "vae",
            "aae",
            "vae-aae",
            "vaevae",
        ],
        default="vae",
        help="Choose which model to run; only vae (vae); only aae (aae); the combination of vae and aae (vae-aae); taxonomy_predictor; vaevae; reclustering [vae]",
    )
    return subparser


def add_aae_arguments(subparser):
    # AAE arguments
    aaeos = subparser.add_argument_group(title="AAE options", description=None)

    aaeos.add_argument(
        "--n_aae",
        dest="nhiddens_aae",
        metavar="",
        type=int,
        default=547,
        help="hidden neurons AAE [547]",
    )
    aaeos.add_argument(
        "--z_aae",
        dest="nlatent_aae_z",
        metavar="",
        type=int,
        default=283,
        help="latent neurons AAE continuous latent space  [283]",
    )
    aaeos.add_argument(
        "--y_aae",
        dest="nlatent_aae_y",
        metavar="",
        type=int,
        default=700,
        help="latent neurons AAE categorical latent space [700]",
    )
    aaeos.add_argument(
        "--sl_aae",
        dest="sl",
        metavar="",
        type=float,
        default=0.00964,
        help="loss scale between reconstruction loss and adversarial losses [0.00964]",
    )
    aaeos.add_argument(
        "--slr_aae",
        dest="slr",
        metavar="",
        type=float,
        default=0.5,
        help="loss scale between reconstruction adversarial losses [0.5] ",
    )
    aaeos.add_argument(
        "--aae_temp",
        dest="temp",
        metavar="",
        type=float,
        default=0.1596,
        help=" Temperature of the softcategorical prior [0.1596]",
    )

    aaetrainos = subparser.add_argument_group(
        title="Training options AAE", description=None
    )

    aaetrainos.add_argument(
        "--e_aae",
        dest="nepochs_aae",
        metavar="",
        type=int,
        default=70,
        help="epochs AAE [70]",
    )
    aaetrainos.add_argument(
        "--t_aae",
        dest="batchsize_aae",
        metavar="",
        type=int,
        default=256,
        help="starting batch size AAE [256]",
    )
    aaetrainos.add_argument(
        "--q_aae",
        dest="batchsteps_aae",
        metavar="",
        type=int,
        nargs="*",
        default=[25, 50],
        help="double batch size at epochs AAE [25,50]",
    )
    return subparser


def add_taxonomy_arguments(subparser, taxonomy_only=False, predictions_only=False):
    # Taxonomy arguments
    taxonomys = subparser.add_argument_group(title="Taxonomy")
    if not predictions_only:
        taxonomys.add_argument(
            "--taxonomy",
            metavar="",
            type=Path,
            help="path to taxonomy tsv file, structured like the output of MMSeqs2 search",
        )
    if not taxonomy_only:
        taxonomys.add_argument(
            "--taxonomy_predictions",
            metavar="",
            type=Path,
            help="path to results_taxonomy_predictor.csv file, output of taxonomy_predictor model",
        )
    if not taxonomy_only and not predictions_only:
        taxonomys.add_argument(
            "--no_predictor",
            help="do not complete mmseqs search with taxonomy predictions [False]",
            action="store_true",
        )
    return subparser


def add_reclustering_arguments(subparser):
    # Reclustering arguments
    reclusters = subparser.add_argument_group(title="k-means reclustering arguments")
    reclusters.add_argument(
        "--latent_path",
        metavar="",
        type=Path,
        help="path to a latent space",
    )
    reclusters.add_argument(
        "--clusters_path",
        metavar="",
        type=Path,
        help="path to a cluster file corresponding to the latent space",
    )
    reclusters.add_argument(
        "--hmmout_path",
        metavar="",
        type=Path,
        default=None,
        help="path to markers.hmmout path generated for the same set of contigs",
    )
    reclusters.add_argument(
        "-ro",
        dest="binsplit_separator_recluster",
        metavar="",
        type=str,
        default="C",
        help="binsplit separator for reclustering [C]",
    )
    reclusters.add_argument(
        "--algorithm",
        metavar="",
        type=str,
        default="kmeans",
        help="which reclustering algorithm to use ('kmeans', 'dbscan'). DBSCAN requires a taxonomy predictions file [kmeans]",
    )
    return subparser


def main():
    doc = f"""
    Version: {'.'.join([str(i) for i in vamb.__version__])}

    Default use, good for most datasets:
    vamb bin default --outdir out --fasta my_contigs.fna --bamfiles *.bam -o C

    Find the latest updates and documentation at https://github.com/RasmussenLab/vamb"""
    parser = argparse.ArgumentParser(
        prog="vamb",
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # usage="%(prog)s outdir tnf_input rpkm_input [options]",
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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    subparsers = parser.add_subparsers(dest="subcommand")

    TAXOMETER = "taxometer"
    BIN = "bin"
    VAMB = "default"
    TAXVAMB = "taxvamb"
    AVAMB = "avamb"
    RECLUSTER = "recluster"

    vaevae_parserbin_parser = subparsers.add_parser(
        BIN,
        help="""
        VAMB, TaxVAMB, AVAMB binners
        """,
    )
    subparsers_model = vaevae_parserbin_parser.add_subparsers(dest="model_subcommand")

    vae_parser = subparsers_model.add_parser(
        VAMB,
        help="""
        default binner based on a variational autoencoder. 
        See the paper "Improved metagenome binning and assembly using deep variational autoencoders" 
        (https://www.nature.com/articles/s41587-020-00777-4)
        """,
    )
    add_input_output_arguments(vae_parser)
    add_vae_arguments(vae_parser)
    add_clustering_arguments(vae_parser)

    vaevae_parser = subparsers_model.add_parser(
        TAXVAMB,
        help="""
        taxonomy informed binner based on a bi-modal variational autoencoder. 
        See the paper "TaxVAMB: taxonomic annotations improve metagenome binning" 
        (link)
        """,
    )
    add_input_output_arguments(vaevae_parser)
    add_vae_arguments(vaevae_parser)
    add_clustering_arguments(vaevae_parser)
    add_predictor_arguments(vaevae_parser)
    add_taxonomy_arguments(vaevae_parser)

    vaeaae_parser = subparsers_model.add_parser(
        AVAMB,
        help="""
        ensemble model of an adversarial autoencoder and a variational autoencoder. 
        See the paper "Adversarial and variational autoencoders improve metagenomic binning" 
        (https://www.nature.com/articles/s42003-023-05452-3). 
        WARNING: recommended use is through the Snakemake pipeline
        """,
    )
    add_input_output_arguments(vaeaae_parser)
    add_vae_arguments(vaeaae_parser)
    add_aae_arguments(vaeaae_parser)
    add_clustering_arguments(vaeaae_parser)

    predict_parser = subparsers.add_parser(
        TAXOMETER,
        help="""
        refines taxonomic annotations of any metagenome classifier. 
        See the paper "Taxometer: deep learning improves taxonomic classification of contigs using binning features and a hierarchical loss" 
        (link)
        """,
    )
    add_input_output_arguments(predict_parser)
    add_taxonomy_arguments(predict_parser, taxonomy_only=True)
    add_predictor_arguments(predict_parser)

    recluster_parser = subparsers.add_parser(
        RECLUSTER,
        help="""
        reclustering using single-copy genes for the binning results of VAMB, TaxVAMB or AVAMB
        """,
    )
    add_input_output_arguments(recluster_parser)
    add_reclustering_arguments(recluster_parser)
    add_taxonomy_arguments(recluster_parser, predictions_only=True)

    args = parser.parse_args()

    if args.subcommand == TAXOMETER:
        runner = TaxometerArguments(args)
    elif args.subcommand == BIN:
        if args.model_subcommand is None:
            vaevae_parserbin_parser.print_help()
            sys.exit(1)
        classes_map = {
            VAMB: VAEArguments,
            AVAMB: VAEAAEArguments,
            TAXVAMB: VAEVAEArguments,
        }
        runner = classes_map[args.model_subcommand](args)
    elif args.subcommand == RECLUSTER:
        runner = ReclusteringArguments(args)
    else:
        # There are no more subcommands
        assert False
    runner.run()


if __name__ == "__main__":
    main()
