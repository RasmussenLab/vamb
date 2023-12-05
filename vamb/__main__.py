#!/usr/bin/env python3

# More imports below, but the user's choice of processors must be parsed before
# numpy can be imported.
import vamb
import numpy as np
import sys
import os
import argparse
import torch
import time
import random
import pycoverm
import itertools
from math import isfinite
from typing import Optional, Tuple, Union, cast
from pathlib import Path
from collections.abc import Sequence
from collections import defaultdict
from torch.utils.data import DataLoader
from loguru import logger
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import datetime
import networkx as nx

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


def format_log(record) -> str:
    colors = {"WARNING": "red", "INFO": "green", "DEBUG": "blue", "ERROR": "red"}
    L = colors.get(record["level"].name, "blue")
    T = "red" if record["level"].name in ("WARNING", "ERROR") else "cyan"
    message = "<red>{message}</red>" if T == "red" else "{message}"
    time = f"<{T}>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</{T}>"
    level = f"<b><{L}>{{level:<7}}</{L}></b>"
    return f"{time} | {level} | {message}\n{{exception}}"


def try_make_dir(name: Union[Path, str]):
    try:
        os.mkdir(name)
    except FileExistsError:
        pass


class FASTAPath(type(Path())):
    __slots__ = []
    pass


class CompositionPath(type(Path())):
    __slots__ = []
    pass


class EmbeddingsPath(type(Path())):
    pass


class CompositionOptions:
    __slots__ = ["path", "min_contig_length", "warn_on_few_seqs"]

    def __init__(
        self,
        fastapath: Optional[Path],
        npzpath: Optional[Path],
        min_contig_length: int,
        warn_on_few_seqs: bool,
    ):
        assert isinstance(fastapath, (Path, type(None)))
        assert isinstance(npzpath, (Path, type(None)))
        assert isinstance(min_contig_length, int)
        assert isinstance(warn_on_few_seqs, bool)

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
        self.warn_on_few_seqs = warn_on_few_seqs


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
                if not pycoverm.is_bam_sorted(str(bampath)):
                    raise ValueError(f"Path {bampath} is not sorted by reference.")
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


class EmbeddingsOptions:
    __slots__ = [
        "path",
        "embeddedcontigspath",
        "shuffle_embeds",
        "embeds_loss",
        "radius",
        "gamma",
        "neighs_object_path",
        "embeddings_mask_path",
        "embeddings_processed_path",
        "margin",
    ]

    def __init__(
        self,
        embeddingspath: Path,
        embeddedcontigspath: Path,
        shuffle_embeds: bool,
        embeds_loss: str,
        radius: float,
        gamma: float,
        neighs_object_path: Optional[Path],
        embeddings_mask_path: Optional[Path],
        embeddings_processed_path: Optional[Path],
        margin: float = 0.01,
    ):
        assert isinstance(embeddingspath, (Path, type(None)))

        self.path = EmbeddingsPath(embeddingspath)
        self.embeddedcontigspath = EmbeddingsPath(embeddedcontigspath)
        self.shuffle_embeds = shuffle_embeds
        self.embeds_loss = embeds_loss
        self.radius = radius
        self.gamma = gamma
        self.margin = margin

        if neighs_object_path is not None:
            assert (
                embeddings_mask_path is not None
                and embeddings_processed_path is not None
            )
            self.neighs_object_path = neighs_object_path  # np.load(neighs_object_path,allow_pickle=True)["arr_0"]
            self.embeddings_mask_path = embeddings_mask_path
            self.embeddings_processed_path = embeddings_processed_path


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
        "no_predictor",
        "taxonomy_path",
        "column_contigs",
        "column_taxonomy",
        "delimiter_taxonomy",
    ]

    def __init__(
        self,
        no_predictor: Optional[bool],
        taxonomy_path: Optional[Path],
        column_contigs: Optional[str],
        column_taxonomy: Optional[str],
        delimiter_taxonomy: Optional[str],
    ):
        assert isinstance(taxonomy_path, (Path, type(None)))

        if taxonomy_path is None and no_predictor:
            raise argparse.ArgumentTypeError(
                "You must specify either --taxonomy_path or --taxonomy_predictions_path or run without --no_predictor flag"
            )

        if taxonomy_path is None and not no_predictor:
            raise argparse.ArgumentTypeError(
                "The taxonomy predictor needs --taxonomy_path for training"
            )

        if taxonomy_path is not None and not taxonomy_path.is_file():
            raise FileNotFoundError(taxonomy_path)

        self.taxonomy_path = taxonomy_path
        self.no_predictor = no_predictor
        self.column_contigs = column_contigs
        self.column_taxonomy = column_taxonomy
        self.delimiter_taxonomy = delimiter_taxonomy


class ReclusteringOptions:
    __slots__ = [
        "latent_path",
        "clusters_path",
        "hmmout_path",
        "binsplitter",
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
        self.hmmout_path = hmmout_path
        self.algorithm = algorithm
        self.binsplitter = vamb.vambtools.BinSplitter(binsplit_separator)


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


class Encodern2vOptions:
    __slots__ = ["vae_options", "alpha"]

    def __init__(
        self,
        vae_options: Optional[VAEOptions],
        alpha: Optional[float],
    ):
        assert isinstance(alpha, (float, type(None)))
        assert isinstance(vae_options, (VAEOptions, type(None)))

        if alpha is not None and (alpha <= 0 or alpha) >= 1:
            raise argparse.ArgumentTypeError("alpha must be above 0 and below 1")
        self.alpha = alpha

        if vae_options is None:
            raise argparse.ArgumentTypeError(
                "You must specify VAE as encoders, cannot run with no encoders."
            )

        self.vae_options = vae_options


class PredictorTrainingOptions(VAETrainingOptions):
    def __init__(
        self,
        nepochs: int,
        batchsize: int,
        batchsteps: list[int],
        softmax_threshold: float,
        ploss: str,
    ):
        losses = ["flat_softmax", "cond_softmax", "soft_margin"]
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
        "max_clusters",
        "binsplitter",
    ]

    def __init__(
        self,
        window_size: int,
        min_successes: int,
        max_clusters: Optional[int],
        binsplit_separator: Optional[str],
    ):
        assert isinstance(window_size, int)
        assert isinstance(min_successes, int)
        assert isinstance(max_clusters, (int, type(None)))
        assert isinstance(binsplit_separator, (str, type(None)))

        if window_size < 1:
            raise argparse.ArgumentTypeError("Window size must be at least 1")
        self.window_size = window_size

        if min_successes < 1 or min_successes > window_size:
            raise argparse.ArgumentTypeError(
                "Minimum cluster size must be in 1:windowsize"
            )
        self.min_successes = min_successes

        if max_clusters is not None and max_clusters < 1:
            raise argparse.ArgumentTypeError("Max clusters must be at least 1")
        self.max_clusters = max_clusters
        self.binsplitter = vamb.vambtools.BinSplitter(binsplit_separator)


class DBSCluster_Options:
    __slots__ = [
        "min_samples",
        "epsilon",
        "metric",
        "binsplit_separator",
    ]

    def __init__(
        self,
        binsplit_separator: Optional[str],
        min_samples: int = 1,
        epsilon: float = 0.001,
        metric: str = "cosine",
    ):
        assert isinstance(min_samples, int)
        assert isinstance(epsilon, float)
        assert isinstance(metric, str)
        assert isinstance(binsplit_separator, (str, type(None)))

        self.min_samples = min_samples
        self.epsilon = epsilon
        self.metric = metric

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


def calc_tnf(
    options: CompositionOptions,
    outdir: Path,
    binsplitter: vamb.vambtools.BinSplitter,
) -> vamb.parsecontigs.Composition:
    begintime = time.time()
    logger.info("Loading TNF")
    logger.info(f"\tMinimum sequence length: {options.min_contig_length}")

    path = options.path

    if isinstance(path, CompositionPath):
        logger.info(f"\tLoading composition from npz at: {path}")
        composition = vamb.parsecontigs.Composition.load(str(path))
        composition.filter_min_length(options.min_contig_length)
    else:
        assert isinstance(path, FASTAPath)
        logger.info(f"\tLoading data from FASTA file {path}")
        with vamb.vambtools.Reader(str(path)) as file:
            composition = vamb.parsecontigs.Composition.from_file(
                file, minlength=options.min_contig_length
            )
        composition.save(outdir.joinpath("composition.npz"))

    binsplitter.initialize(composition.metadata.identifiers)

    if options.warn_on_few_seqs and composition.nseqs < 20_000:
        message = (
            f"Kept only {composition.nseqs} sequences from FASTA file. "
            "We normally expect 20,000 sequences or more to prevent overfitting. "
            "As a deep learning model, VAEs are prone to overfitting with too few sequences. "
            "You may want to  bin more samples as a time, lower the beta parameter, "
            "or use a different binner altogether.\n"
        )
        logger.opt(raw=True).info("\n")
        logger.warning(message)

    # Warn the user if any contigs have been observed, which is smaller
    # than the threshold.
    if not np.all(composition.metadata.mask):
        n_removed = len(composition.metadata.mask) - np.sum(composition.metadata.mask)
        message = (
            f"The minimum sequence length has been set to {options.min_contig_length}, "
            f"but {n_removed} sequences fell below this threshold and was filtered away."
            "\nBetter results are obtained if the sequence file is filtered to the minimum "
            "sequence length before mapping.\n"
        )
        logger.opt(raw=True).info("\n")
        logger.warning(message)

    elapsed = round(time.time() - begintime, 2)
    logger.info(
        f"\tKept {composition.count_bases()} bases in {composition.nseqs} sequences"
    )
    logger.info(f"\tProcessed TNF in {elapsed} seconds.\n")

    return composition


def calc_rpkm(
    abundance_options: AbundanceOptions,
    outdir: Path,
    comp_metadata: vamb.parsecontigs.CompositionMetaData,
    nthreads: int,
) -> vamb.parsebam.Abundance:
    begintime = time.time()
    logger.info("Loading depths")
    logger.info(
        f'\tReference hash: {comp_metadata.refhash.hex() if abundance_options.refcheck else "None"}'
    )

    path = abundance_options.path
    if isinstance(path, AbundancePath):
        logger.info(f"\tLoading depths from npz at: {str(path)}")

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
    else:
        assert isinstance(path, list)
        logger.info(f"\tParsing {len(path)} BAM files with {nthreads} threads")

        abundance = vamb.parsebam.Abundance.from_files(
            path,
            outdir.joinpath("tmp").joinpath("pycoverm"),
            comp_metadata,
            abundance_options.refcheck,
            abundance_options.min_alignment_id,
            nthreads,
        )
        abundance.save(outdir.joinpath("abundance.npz"))

        logger.info(f"\tMin identity: {abundance.minid}")
        logger.info("\tOrder of columns is:")
        for samplename in abundance.samplenames:
            logger.info(f"\t\t{samplename}")

    elapsed = round(time.time() - begintime, 2)
    logger.info(f"\tProcessed RPKM in {elapsed} seconds.\n")

    return abundance


def trainvae(
    vae_options: VAEOptions,
    training_options: VAETrainingOptions,
    vamb_options: VambOptions,
    lrate: float,
    alpha: Optional[float],
    data_loader: DataLoader,
) -> np.ndarray:
    begintime = time.time()
    logger.info("Creating and training VAE")

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

    logger.info("\tCreated VAE")
    modelpath = vamb_options.out_dir.joinpath("model.pt")
    vae.trainmodel(
        vamb.encode.set_batchsize(data_loader, training_options.batchsize),
        nepochs=training_options.nepochs,
        lrate=lrate,
        batchsteps=training_options.batchsteps,
        modelfile=modelpath,
    )

    logger.info("\tEncoding to latent representation")
    latent = vae.encode(data_loader)
    vamb.vambtools.write_npz(vamb_options.out_dir.joinpath("latent.npz"), latent)
    del vae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    logger.info(f"\tTrained VAE and encoded in {elapsed} seconds.\n")

    return latent


def trainvae_n2v_asimetric(
    vae_options: VAEOptions,
    training_options: VAETrainingOptions,
    vamb_options: VambOptions,
    embeddings_options: EmbeddingsOptions,
    lrate: float,
    alpha: Optional[float],
    data_loader: DataLoader,
    neighs_object: np.ndarray,
) -> np.ndarray:
    begintime = time.time()
    logger.info("Creating and training VAE")

    n_embedding = data_loader.dataset.tensors[2].shape[1]
    nsamples = data_loader.dataset.tensors[0].shape[1]
    ncontigs = data_loader.dataset.tensors[0].shape[0]

    vae = vamb.encode_n2v_asimetric.VAE(
        nsamples,
        ncontigs,
        n_embedding,
        neighs_object,
        nhiddens=vae_options.nhiddens,
        nlatent=vae_options.nlatent,
        alpha=alpha,
        beta=vae_options.beta,
        gamma=embeddings_options.gamma,
        dropout=vae_options.dropout,
        cuda=vamb_options.cuda,
        seed=vamb_options.seed,
        margin=embeddings_options.margin,
        embs_loss=embeddings_options.embeds_loss,
    )

    logger.info("\nCreated VAE with embeddings")
    modelpath = vamb_options.out_dir.joinpath("model.pt")
    vae.trainmodel(
        vamb.encode.set_batchsize(data_loader, training_options.batchsize),
        nepochs=training_options.nepochs,
        lrate=lrate,
        batchsteps=training_options.batchsteps,
        modelfile=modelpath,
        # warmup=training_options.warmup,
    )

    logger.info("\nEncoding to latent representation")
    latent = vae.encode(data_loader)
    vamb.vambtools.write_npz(vamb_options.out_dir.joinpath("latent.npz"), latent)
    del vae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    logger.info(f"\nTrained VAE and encoded in {elapsed} seconds.")

    return latent


def trainaae(
    data_loader: DataLoader,
    aae_options: AAEOptions,
    training_options: AAETrainingOptions,
    vamb_options: VambOptions,
    lrate: float,
    alpha: Optional[float],  # set automatically if None
    contignames: Sequence[str],
) -> tuple[np.ndarray, dict[str, set[str]]]:
    begintime = time.time()
    logger.info("Creating and training AAE")
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

    logger.info("\tCreated AAE")
    modelpath = os.path.join(vamb_options.out_dir, "aae_model.pt")
    aae.trainmodel(
        vamb.encode.set_batchsize(data_loader, training_options.batchsize),
        training_options.nepochs,
        training_options.batchsteps,
        training_options.temp,
        lrate,
        modelpath,
    )

    logger.info("\tEncoding to latent representation")
    clusters_y_dict, latent = aae.get_latents(contignames, data_loader)
    vamb.vambtools.write_npz(
        os.path.join(vamb_options.out_dir, "aae_z_latent.npz"), latent
    )

    del aae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    logger.info(f"\tTrained AAE and encoded in {elapsed} seconds.\n")

    return latent, clusters_y_dict


def cluster_and_write_files(
    vamb_options: VambOptions,
    cluster_options: ClusterOptions,
    base_clusters_name: str,  # e.g. /foo/bar/vae -> /foo/bar/vae_unsplit.tsv
    bins_dir: Path,
    fasta_catalogue: Path,
    latent: np.ndarray,
    sequence_names: Sequence[str],
    sequence_lens: Sequence[int],
):
    begintime = time.time()
    # Create cluser iterator
    logger.info("Clustering")
    logger.info(f"\tWindowsize: {cluster_options.window_size}")
    logger.info(
        f"\tMin successful thresholds detected: {cluster_options.min_successes}",
    )
    logger.info(f"\tMax clusters: {cluster_options.max_clusters}")
    logger.info(f"\tUse CUDA for clustering: {vamb_options.cuda}")
    logger.info(f"\tBinsplitter: {cluster_options.binsplitter.log_string()}")

    cluster_generator = vamb.cluster.ClusterGenerator(
        latent,
        sequence_lens,  # type:ignore
        windowsize=cluster_options.window_size,
        minsuccesses=cluster_options.min_successes,
        destroy=True,
        normalized=False,
        cuda=vamb_options.cuda,
        rng_seed=vamb_options.seed,
    )
    # This also works correctly when max_clusters is None
    clusters = itertools.islice(cluster_generator, cluster_options.max_clusters)
    cluster_dict: dict[str, set[str]] = dict()

    # Write the cluster metadata to file
    with open(Path(base_clusters_name + "_metadata.tsv"), "w") as file:
        print("name\tradius\tpeak valley ratio\tkind\tbp\tncontigs", file=file)
        for i, cluster in enumerate(clusters):
            cluster_dict[str(i + 1)] = {
                sequence_names[cast(int, i)] for i in cluster.members
            }
            print(
                str(i + 1),
                None if cluster.radius is None else round(cluster.radius, 3),
                None
                if cluster.observed_pvr is None
                else round(cluster.observed_pvr, 2),
                cluster.kind_str,
                sum(sequence_lens[i] for i in cluster.members),
                len(cluster.members),
                file=file,
                sep="\t",
            )

    elapsed = round(time.time() - begintime, 2)

    write_clusters_and_bins(
        vamb_options,
        cluster_options.binsplitter,
        base_clusters_name,
        bins_dir,
        fasta_catalogue,
        cluster_dict,
        sequence_names,
        sequence_lens,
    )
    logger.info(f"\tClustered contigs in {elapsed} seconds.\n")


def cluster_dbs(
    dbs_cluster_options: DBSCluster_Options,
    clusterspath: Path,
    latent: np.ndarray,
    contignames: Sequence[str],
) -> None:
    begintime = time.time()

    logger.info("\nClustering with DBSCAN")
    logger.info(f"Min_samples : {dbs_cluster_options.min_samples}")
    logger.info(f"Epsilon: {dbs_cluster_options.epsilon}")
    logger.info(
        "Separator: {}".format(
            None
            if dbs_cluster_options.binsplit_separator is None
            else ('"' + dbs_cluster_options.binsplit_separator + '"')
        )
    )
    db = DBSCAN(
        min_samples=dbs_cluster_options.min_samples,
        eps=dbs_cluster_options.epsilon,
        metric="cosine",
    )
    db.fit(latent)

    print(db.labels_)

    np.savetxt(
        clusterspath,
        np.array([db.labels_, contignames], dtype=object).T,
        delimiter="\t",
        fmt="%s",
    )

    logger.info(f"Clustered {len(contignames)} contigs in {len(set(db.labels_))} bins")

    elapsed = round(time.time() - begintime, 2)
    logger.info(f"Clustered contigs in {elapsed} seconds with DBSCAN.")


def write_clusters_and_bins(
    vamb_options: VambOptions,
    binsplitter: vamb.vambtools.BinSplitter,
    base_clusters_name: str,  # e.g. /foo/bar/vae -> /foo/bar/vae_unsplit.tsv
    bins_dir: Path,
    fasta_catalogue: Path,
    clusters: dict[str, set[str]],
    sequence_names: Sequence[str],
    sequence_lens: Sequence[int],
):
    # Write unsplit clusters to file
    begintime = time.time()
    unsplit_path = Path(base_clusters_name + "_unsplit.tsv")
    with open(unsplit_path, "w") as file:
        (n_unsplit_clusters, n_contigs) = vamb.vambtools.write_clusters(
            file, clusters.items()
        )

    # Open unsplit clusters and split them
    if binsplitter.splitter is not None:
        split_path = Path(base_clusters_name + "_split.tsv")
        clusters = dict(binsplitter.binsplit(clusters.items()))
        with open(split_path, "w") as file:
            (n_split_clusters, _) = vamb.vambtools.write_clusters(
                file, clusters.items()
            )
        msg = f"\tClustered {n_contigs} contigs in {n_split_clusters} split bins ({n_unsplit_clusters} clusters)"
    else:
        msg = f"\tClustered {n_contigs} contigs in {n_unsplit_clusters} unsplit bins"

    logger.info(msg)
    elapsed = round(time.time() - begintime, 2)
    logger.info(f"\tWrote cluster file(s) in {elapsed} seconds.")

    # Write bins, if necessary
    if vamb_options.min_fasta_output_size is not None:
        starttime = time.time()
        filtered_clusters: dict[str, set[str]] = dict()
        assert len(sequence_lens) == len(sequence_names)
        sizeof = dict(zip(sequence_names, sequence_lens))
        for binname, contigs in clusters.items():
            if sum(sizeof[c] for c in contigs) >= vamb_options.min_fasta_output_size:
                filtered_clusters[binname] = contigs

        with vamb.vambtools.Reader(fasta_catalogue) as file:
            vamb.vambtools.write_bins(
                bins_dir,
                filtered_clusters,
                file,
                None,
            )
        elapsed = round(time.time() - starttime, 2)
        n_bins = len(filtered_clusters)
        n_contigs = sum(len(v) for v in filtered_clusters.values())
        logger.info(
            f"\tWrote {n_bins} bins with {n_contigs} sequences in {elapsed} seconds."
        )


def load_composition_and_abundance(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    binsplitter: vamb.vambtools.BinSplitter,
) -> Tuple[vamb.parsecontigs.Composition, vamb.parsebam.Abundance]:
    composition = calc_tnf(comp_options, vamb_options.out_dir, binsplitter)
    abundance = calc_rpkm(
        abundance_options,
        vamb_options.out_dir,
        composition.metadata,
        vamb_options.n_threads,
    )
    return (composition, abundance)


def find_neighbours(
    embeddings,
    radius=0.2,
):
    # Compute pairwise cosine distances
    cosine_distances = cdist(embeddings, embeddings, metric="cosine")

    # Set the diagonal to a value greater than the threshold to avoid self-selection
    np.fill_diagonal(cosine_distances, 1)

    # Find the indices of rows that are closer than the threshold in cosine distance
    # closer_indices = np.where(cosine_distances < radius)

    # Print pairs of indices
    idxs_with_neighs = []

    neighs = list()
    for i in range(len(cosine_distances)):
        neighs_i = list()
        for j in range(len(cosine_distances)):
            if cosine_distances[i, j] < radius:
                neighs_i.append(j)
        neighs.append(neighs_i)
        if len(neighs_i) > 0:
            idxs_with_neighs.append(i)
    return neighs, idxs_with_neighs


def load_composition_and_abundance_and_embeddings(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    embeddings_options: EmbeddingsOptions,
    binsplitter: vamb.vambtools.BinSplitter,
) -> Tuple[vamb.parsecontigs.Composition, vamb.parsebam.Abundance, np.ndarray]:
    logger.info(
        "Starting Vamb (with node2vec embeddings) version "
        + ".".join(map(str, vamb.__version__))
    )
    logger.info("Date and time is " + str(datetime.datetime.now()))
    begintime = time.time()

    composition = calc_tnf(comp_options, vamb_options.out_dir, binsplitter)
    abundance = calc_rpkm(
        abundance_options,
        vamb_options.out_dir,
        composition.metadata,
        vamb_options.n_threads,
    )

    time_generating_input = round(time.time() - begintime, 2)
    logger.info(f"\nTNF and coabundances generated in {time_generating_input} seconds.")

    if embeddings_options.neighs_object_path is None:
        logger.info("\nComputing contig embedding neighbours")

        embeddings = np.load(embeddings_options.path)["arr_0"]
        if embeddings_options.shuffle_embeds == True:
            np.random.shuffle(embeddings)
        logger.info("Embeddings loaded from %s " % str(embeddings_options.path))

        contigs_embedded_all = np.loadtxt(
            embeddings_options.embeddedcontigspath, dtype=object
        )
        # first mask embeddings and contigs_embedded by the contigs that used for binningcomposition.metadata.identifiers

        mask_contigs_embedded_all = np.array(
            [
                1 if c in composition.metadata.identifiers else 0
                for c in contigs_embedded_all
            ],
            dtype=bool,
        )
        contigs_embedded_2k_incomplete = contigs_embedded_all[mask_contigs_embedded_all]
        embeddings_2k_incomplete = embeddings[mask_contigs_embedded_all]

        # # now mask embeddings_2k by contigs that have neighs within radius
        neighs, idxs_with_neighs = find_neighbours(
            embeddings_2k_incomplete, radius=embeddings_options.radius
        )
        contigs_embedds_d = {
            c: neighs_c for c, neighs_c in zip(contigs_embedded_all, neighs)
        }

        logger.info(
            "%i/%i contigs have neighbours within %.2f "
            % (
                len(idxs_with_neighs),
                len(composition.metadata.identifiers),
                embeddings_options.radius,
            )
        )
        neighs = np.array(
            [
                np.array(contigs_embedds_d[c])
                if c in contigs_embedds_d.keys()
                else np.array([])
                for c in composition.metadata.identifiers
            ],
            dtype=object,
        )

        contigs_embedded_2k_with_neighs = contigs_embedded_2k_incomplete[
            idxs_with_neighs
        ]
        embeddings_2k_with_neighs = embeddings_2k_incomplete[idxs_with_neighs, :]
        c2kincomplete_embed_d = {
            c: e
            for c, e, in zip(contigs_embedded_2k_with_neighs, embeddings_2k_with_neighs)
        }

        # now we have an object with a list of contigs used for binning that have neighs, but we might still miss some contigs here since
        # when the embeddings where genearted, only contigs part of connected components where saved
        embeddings_2k_complete = np.array(
            [
                c2kincomplete_embed_d[c]
                if c in c2kincomplete_embed_d.keys()
                else [55.0] * embeddings.shape[1]
                for c in composition.metadata.identifiers
            ]
        )
        contigs_2k_complete_w_neighs_mask = np.array(
            [
                1 if c in c2kincomplete_embed_d.keys() else 0
                for c in composition.metadata.identifiers
            ],
            dtype=bool,
        )

        time_generating_embedds = round(time.time() - begintime, 2)

        logger.info(f"\nEmbeddings processed in {time_generating_embedds} seconds.")
    else:
        logger.info(f"\nLoading neighs from  {embeddings_options.neighs_object_path}.")
        embeddings_2k_complete = np.load(embeddings_options.embeddings_processed_path)[
            "arr_0"
        ]
        contigs_2k_complete_w_neighs_mask = np.load(
            embeddings_options.embeddings_mask_path
        )["arr_0"]
        neighs = np.load(embeddings_options.neighs_object_path, allow_pickle=True)[
            "arr_0"
        ]
        contigs_with_neighs_n = np.sum([1 for neighs in neighs if len(neighs) > 0])
        logger.info(f"\nContigs with neighs   {contigs_with_neighs_n}.")
    return (
        composition,
        abundance,
        embeddings_2k_complete,
        contigs_2k_complete_w_neighs_mask,
        neighs,
    )


def run(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    encoder_options: EncoderOptions,
    training_options: TrainingOptions,
    cluster_options: ClusterOptions,
):
    vae_options = encoder_options.vae_options
    aae_options = encoder_options.aae_options
    vae_training_options = training_options.vae_options
    aae_training_options = training_options.aae_options

    composition, abundance = load_composition_and_abundance(
        vamb_options=vamb_options,
        comp_options=comp_options,
        abundance_options=abundance_options,
        binsplitter=cluster_options.binsplitter,
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
        )

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
            contignames=composition.metadata.identifiers,  # type:ignore
        )

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
        cluster_and_write_files(
            vamb_options,
            cluster_options,
            str(vamb_options.out_dir.joinpath("vae_clusters")),
            vamb_options.out_dir.joinpath("bins"),
            comp_options.path,
            latent,
            comp_metadata.identifiers,  # type:ignore
            comp_metadata.lengths,  # type:ignore
        )
        del latent

    if aae_options is not None:
        assert latent_z is not None
        assert clusters_y_dict is not None
        assert comp_metadata.nseqs == len(latent_z)
        cluster_and_write_files(
            vamb_options,
            cluster_options,
            str(vamb_options.out_dir.joinpath("aae_z_clusters")),
            vamb_options.out_dir.joinpath("bins"),
            comp_options.path,
            latent_z,
            comp_metadata.identifiers,  # type:ignore
            comp_metadata.lengths,  # type:ignore
        )
        del latent_z
        # We enforce this in the VAEAAEOptions constructor, see comment there
        assert cluster_options.max_clusters is None
        write_clusters_and_bins(
            vamb_options,
            cluster_options.binsplitter,
            str(vamb_options.out_dir.joinpath("aae_y_clusters")),
            vamb_options.out_dir.joinpath("bins"),
            comp_options.path,
            clusters_y_dict,
            comp_metadata.identifiers,  # type:ignore
            comp_metadata.lengths,  # type:ignore
        )


def parse_taxonomy(
    taxonomy_path: Path,
    contignames: Sequence[str],
    column_contigs: Optional[str],
    column_taxonomy: Optional[str],
    delimiter_taxonomy: Optional[str],
) -> pd.Series:
    if column_contigs is None or column_taxonomy is None:
        df_taxonomy = pd.read_csv(taxonomy_path, delimiter="\t", header=None)
        column_contigs = 0
        column_taxonomy = 8
        if len(df_taxonomy.columns) != 9:
            raise ValueError(
                "The contigs column or the taxonomy column are not defined, but the taxonomy file is not in MMseqs2 format"
            )
    else:
        df_taxonomy = pd.read_csv(taxonomy_path, delimiter=delimiter_taxonomy)
    logger.info(f"{len(df_taxonomy)} lines in taxonomy file")
    logger.info(f"{len(contignames)} contigs")

    df_taxonomy = df_taxonomy.fillna(np.nan)
    df_taxonomy = df_taxonomy[df_taxonomy[column_contigs].isin(contignames)]

    missing_contigs = set(contignames) - set(df_taxonomy[column_contigs])
    new_rows = []
    for c in missing_contigs:
        new_row: dict[int, Union[str, float]] = {i: np.nan for i in df_taxonomy.columns}
        new_row[column_contigs] = c
        new_rows.append(new_row)

    df_taxonomy = pd.concat([df_taxonomy, pd.DataFrame(new_rows)], ignore_index=True)

    df_taxonomy.set_index(column_contigs, inplace=True)
    df_taxonomy = df_taxonomy.loc[contignames]
    df_taxonomy.reset_index(inplace=True)
    graph_column = df_taxonomy[column_taxonomy].astype(str)

    if list(df_taxonomy[column_contigs]) != list(contignames):
        raise AssertionError(
            "The contig names of taxonomy entries are not the same as in the contigs metadata"
        )

    return graph_column


def predict_taxonomy(
    rpkms: np.ndarray,
    tnfs: np.ndarray,
    lengths: np.ndarray,
    contignames: np.ndarray,
    taxonomy_options: TaxonomyOptions,
    out_dir: Path,
    predictor_training_options: PredictorTrainingOptions,
    cuda: bool,
):
    begintime = time.time()

    graph_column = parse_taxonomy(
        taxonomy_path=taxonomy_options.taxonomy_path,
        contignames=contignames,  # type:ignore
        column_contigs=taxonomy_options.column_contigs,
        column_taxonomy=taxonomy_options.column_taxonomy,
        delimiter_taxonomy=taxonomy_options.delimiter_taxonomy,
    )
    graph_column.loc[graph_column == ""] = np.nan
    nodes, ind_nodes, table_parent = vamb.taxvamb_encode.make_graph(
        graph_column.unique()
    )
    logger.info(f"{len(nodes)} nodes in the graph")
    graph_column = graph_column.fillna("Domain")
    classes_order = np.array(list(graph_column.str.split(";").str[-1]))
    targets = np.array([ind_nodes[i] for i in classes_order])

    model = vamb.taxvamb_encode.VAMB2Label(
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
    dataloader_joint = vamb.taxvamb_encode.make_dataloader_concat_hloss(
        rpkms,
        tnfs,
        lengths,
        targets,
        len(nodes),
        table_parent,
        batchsize=predictor_training_options.batchsize,
    )

    logger.info("\tCreated dataloader")

    predictortime = time.time()
    logger.info("Starting training the taxonomy predictor")
    logger.info(f"Using threshold {predictor_training_options.softmax_threshold}")

    model_path = out_dir.joinpath("predictor_model.pt")
    with open(model_path, "wb") as modelfile:
        model.trainmodel(
            dataloader_joint,
            nepochs=predictor_training_options.nepochs,
            modelfile=modelfile,
            batchsteps=predictor_training_options.batchsteps,
        )
    logger.info(
        f"Finished training the taxonomy predictor in {round(time.time() - predictortime, 2)} seconds."
    )

    logger.info("Writing the taxonomy predictions")
    predicted_path = out_dir.joinpath("results_taxometer.csv")
    nodes_ar = np.array(nodes)

    logger.info(f"Using threshold {predictor_training_options.softmax_threshold}")
    row = 0
    output_exists = False
    for predicted_vector, predicted_labels in model.predict(dataloader_vamb):
        N = len(predicted_vector)
        predictions: list[str] = []
        probs: list[str] = []
        for i in range(predicted_vector.shape[0]):
            label = predicted_labels[i]
            while table_parent[label] != -1:
                label = table_parent[label]
            threshold_mask = (
                predicted_vector[i] > predictor_training_options.softmax_threshold
            )
            pred_line = ";".join(nodes_ar[threshold_mask][1:])
            predictions.append(pred_line)
            absolute_probs = predicted_vector[i][threshold_mask]
            absolute_prob = ";".join(map(str, absolute_probs[1:]))
            probs.append(absolute_prob)

        with open(predicted_path, "a") as file:
            if not output_exists:
                print("contigs,lengths,predictions,scores", file=file)
            for contig, length, prediction, prob in zip(
                contignames[row : row + N], lengths[row : row + N], predictions, probs
            ):
                print(contig, length, prediction, prob, file=file, sep=",")
        output_exists = True
        row += N

    logger.info(
        f"Completed taxonomy predictions in {round(time.time() - begintime, 2)} seconds."
    )


def run_taxonomy_predictor(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    predictor_training_options: PredictorTrainingOptions,
    taxonomy_options: TaxonomyOptions,
):
    composition, abundance = load_composition_and_abundance(
        vamb_options=vamb_options,
        comp_options=comp_options,
        abundance_options=abundance_options,
        binsplitter=vamb.vambtools.BinSplitter.inert_splitter(),
    )
    rpkms, tnfs, lengths, contignames = (
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        composition.metadata.identifiers,
    )
    assert taxonomy_options.taxonomy_path is not None
    predict_taxonomy(
        rpkms=rpkms,
        tnfs=tnfs,
        lengths=lengths,
        contignames=contignames,
        taxonomy_options=taxonomy_options,
        out_dir=vamb_options.out_dir,
        predictor_training_options=predictor_training_options,
        cuda=vamb_options.cuda,
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
):
    vae_options = encoder_options.vae_options
    composition, abundance = load_composition_and_abundance(
        vamb_options=vamb_options,
        comp_options=comp_options,
        abundance_options=abundance_options,
        binsplitter=vamb.vambtools.BinSplitter.inert_splitter(),
    )
    rpkms, tnfs, lengths, contignames = (
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        composition.metadata.identifiers,
    )

    if taxonomy_options.taxonomy_path is not None and not taxonomy_options.no_predictor:
        logger.info("Predicting missing values from mmseqs taxonomy")
        predict_taxonomy(
            rpkms=rpkms,
            tnfs=tnfs,
            lengths=lengths,
            contignames=contignames,
            taxonomy_options=taxonomy_options,
            out_dir=vamb_options.out_dir,
            predictor_training_options=predictor_training_options,
            cuda=vamb_options.cuda,
        )
        predictions_path = vamb_options.out_dir.joinpath("results_taxometer.csv")
        column_contigs = "contigs"
        column_taxonomy = "predictions"
        delimiter_taxonomy = ","
    else:
        logger.info("Not predicting the taxonomy")
        predictions_path = None

    if predictions_path is not None:
        graph_column = parse_taxonomy(
            taxonomy_path=predictions_path,
            contignames=contignames,  # type:ignore
            column_contigs=column_contigs,
            column_taxonomy=column_taxonomy,
            delimiter_taxonomy=delimiter_taxonomy,
        )
    elif taxonomy_options.no_predictor:
        logger.info("Using mmseqs taxonomy for semisupervised learning")
        assert taxonomy_options.taxonomy_path is not None
        graph_column = parse_taxonomy(
            taxonomy_path=taxonomy_options.taxonomy_path,
            contignames=contignames,  # type:ignore
            column_contigs=taxonomy_options.column_contigs,
            column_taxonomy=taxonomy_options.column_taxonomy,
            delimiter_taxonomy=taxonomy_options.delimiter_taxonomy,
        )
    else:
        raise argparse.ArgumentTypeError("One of the taxonomy arguments is missing")

    graph_column.loc[graph_column == ""] = np.nan
    nodes, ind_nodes, table_parent = vamb.taxvamb_encode.make_graph(
        graph_column.unique()
    )
    graph_column = graph_column.fillna("Domain")
    classes_order = np.array(list(graph_column.str.split(";").str[-1]))
    targets = np.array([ind_nodes[i] for i in classes_order])

    assert vae_options is not None
    vae = vamb.taxvamb_encode.VAEVAEHLoss(
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
    )

    dataloader_vamb = vamb.encode.make_dataloader(
        rpkms,
        tnfs,
        lengths,
        batchsize=vae_training_options.batchsize,
        cuda=vamb_options.cuda,
    )
    dataloader_joint = vamb.taxvamb_encode.make_dataloader_concat_hloss(
        rpkms,
        tnfs,
        lengths,
        targets,
        len(nodes),
        table_parent,
        batchsize=vae_training_options.batchsize,
        cuda=vamb_options.cuda,
    )
    dataloader_labels = vamb.taxvamb_encode.make_dataloader_labels_hloss(
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
    dataloader = vamb.taxvamb_encode.make_dataloader_semisupervised_hloss(
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
            batchsteps=vae_training_options.batchsteps,
        )

    latent_both = vae.VAEJoint.encode(dataloader_joint)
    logger.info(f"{latent_both.shape} embedding shape")

    LATENT_PATH = vamb_options.out_dir.joinpath("vaevae_latent.npy")
    np.save(LATENT_PATH, latent_both)

    # Cluster, save tsv file
    cluster_and_write_files(
        vamb_options,
        cluster_options,
        str(vamb_options.out_dir.joinpath("vaevae_clusters")),
        vamb_options.out_dir.joinpath("bins"),
        comp_options.path,
        latent_both,
        contignames,  # type:ignore
        lengths,  # type:ignore
    )


def run_reclustering(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    reclustering_options: ReclusteringOptions,
    taxonomy_options: TaxonomyOptions,
):
    composition, _ = load_composition_and_abundance(
        vamb_options=vamb_options,
        comp_options=comp_options,
        abundance_options=abundance_options,
        binsplitter=reclustering_options.binsplitter,
    )

    logger.info(f"{len(composition.metadata.identifiers)} contig names after filtering")
    logger.info("mmseqs taxonomy predictions are provided")
    predictions_path = taxonomy_options.taxonomy_predictions_path

    reclustered = vamb.reclustering.recluster_bins(
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

    cluster_dict: defaultdict[str, set[str]] = defaultdict(set)
    for k, v in zip(reclustered, composition.metadata.identifiers):
        cluster_dict[k].add(v)

    write_clusters_and_bins(
        vamb_options,
        reclustering_options.binsplitter,
        str(vamb_options.out_dir.joinpath("clusters_reclustered")),
        vamb_options.out_dir.joinpath("bins"),
        comp_options.path,
        cluster_dict,
        composition.metadata.identifiers,  # type:ignore
        composition.metadata.lengths,  # type:ignore
    )


def run_n2v_asimetric(
    vamb_options: VambOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    embeddings_options: EmbeddingsOptions,
    encoder_options: Encodern2vOptions,
    training_options: TrainingOptions,
    cluster_options: ClusterOptions,
    dbs_cluster_options: DBSCluster_Options,
):
    # I HAVE TO MAKE CHANGES ACCORDING TO THE CHANGES IN THE ENCODE_N2V_ASIMETRIC
    # I ALSO HAVE TO IMPLEMENT A FUNCTION THAT GENERATES AN OBJECT INDICATING THE NEIGHBOURS OF EACH CONTIG
    # BAED ON EMBEDDING SPACE AND radius DISTANCE
    vae_options = encoder_options.vae_options
    vae_training_options = training_options.vae_options

    begintime = time.time()

    (
        composition,
        abundance,
        embeddings,
        embeddings_mask,
        neighs_object,
    ) = load_composition_and_abundance_and_embeddings(
        vamb_options=vamb_options,
        comp_options=comp_options,
        abundance_options=abundance_options,
        embeddings_options=embeddings_options,
        binsplitter=cluster_options.binsplitter,
    )

    # Write contignames and contiglengths needed for dereplication purposes
    np.savetxt(
        vamb_options.out_dir.joinpath("contignames"),
        composition.metadata.identifiers,
        fmt="%s",
    )
    np.savez(vamb_options.out_dir.joinpath("lengths.npz"), composition.metadata.lengths)
    vamb.vambtools.write_npz(
        vamb_options.out_dir.joinpath("embeddings.npz"), embeddings
    )
    vamb.vambtools.write_npz(
        vamb_options.out_dir.joinpath("embeddings_mask.npz"), embeddings_mask
    )
    vamb.vambtools.write_npz(
        vamb_options.out_dir.joinpath("neighs_object.npz"),
        neighs_object,
    )

    logger.info(
        "Creating dataloader and mask (including embeddings and neighbours objects)"
    )

    (
        data_loader,
        neighs_object_after_dataloader,
        # mask,
    ) = vamb.encode_n2v_asimetric.make_dataloader_n2v(
        abundance.matrix.astype(np.float32),
        composition.matrix.astype(np.float32),
        embeddings.astype(np.float32),
        embeddings_mask.astype(np.float32),
        neighs_object,
        composition.metadata.lengths,
        256,  # dummy value - we change this before using the actual loader
        destroy=True,
        cuda=vamb_options.cuda,
    )
    # composition.metadata.filter_mask(mask)

    # logger.info(
    #     "Created dataloader and mask (including embeddings and neighbours objects)"
    # )
    # if embeddings_options.shuffle_embeds:
    #     logger.info("embeddings rows shuffled")

    # vamb.vambtools.write_npz(vamb_options.out_dir.joinpath("mask.npz"), mask)
    # n_discarded = len(mask) - mask.sum()
    # logger.info(f"Number of sequences unsuitable for encoding: {n_discarded}")
    # logger.info(f"Number of sequences remaining: {len(mask) - n_discarded}")

    latent = None
    # Train, save model
    latent = trainvae_n2v_asimetric(
        vae_options=vae_options,
        training_options=vae_training_options,
        vamb_options=vamb_options,
        embeddings_options=embeddings_options,
        lrate=training_options.lrate,
        alpha=encoder_options.alpha,
        data_loader=data_loader,
        neighs_object=neighs_object_after_dataloader,
    )

    # Free up memory
    comp_metadata = composition.metadata
    del composition, abundance

    del embeddings

    assert latent is not None
    assert comp_metadata.nseqs == len(latent)
    # Cluster with DBSCAN, save tsv file
    # clusterspath = vamb_options.out_dir.joinpath("vae_dbscan_clusters.tsv")
    # cluster_dbs(
    #     dbs_cluster_options,
    #     clusterspath,
    #     latent,
    #     comp_metadata.identifiers,
    # )

    # Cluster first contigs within the neighbourhoods margins

    withinmarginclusters_cs_d, mask_contigs_already_clustered = cluster_neighs_based(
        neighs_object=neighs_object,
        latents=latent,
        contignames=comp_metadata.identifiers,
    )
    # save clusters within margins ONLY THERE WILL BE MISSING CONTIGS HERE
    clusterspath = vamb_options.out_dir.joinpath("vae_clusters_within_margins.tsv")

    write_clusters_and_bins(
        vamb_options,
        cluster_options.binsplitter,
        str(vamb_options.out_dir.joinpath("vae_clusters_within_margins")),
        vamb_options.out_dir.joinpath("bins_within_margins"),
        comp_options.path,
        withinmarginclusters_cs_d,
        comp_metadata.identifiers[mask_contigs_already_clustered],
        comp_metadata.lengths[mask_contigs_already_clustered],
    )
    logger.info(f"\tClustered within neighs contigs.\n")

    # now cluster the remaining contigs not within margins
    clusterspath = vamb_options.out_dir.joinpath("vae_clusters_outside_margins.tsv")
    cluster_and_write_files(
        vamb_options,
        cluster_options,
        str(vamb_options.out_dir.joinpath("vae_clusters_outside_margins")),
        vamb_options.out_dir.joinpath("bins_outside_margins"),
        comp_options.path,
        latent[~mask_contigs_already_clustered],
        comp_metadata.identifiers[~mask_contigs_already_clustered],  # type:ignore
        comp_metadata.lengths[~mask_contigs_already_clustered],  # type:ignore
    )
    # merge the within and outside clusters and ensure tehre are not repeats and missing contigs
    clusterspath = vamb_options.out_dir.joinpath(
        "vae_clusters_outside_margins_unsplit.tsv"
    )
    merged_cl_cs_d = withinmarginclusters_cs_d.copy()
    cl_c_ar = np.loadtxt(clusterspath, delimiter="\t", dtype=object)
    for cl in set(cl_c_ar[:, 0]):
        merged_cl_cs_d[cl] = set()

    for cl, c in cl_c_ar:
        merged_cl_cs_d[cl].add(c)

    assert len(np.sum(set([c for cs in merged_cl_cs_d.values() for c in cs]))) == len(
        comp_metadata.identifiers
    )
    # save clusters within margins and outside margins
    clusterspath = vamb_options.out_dir.joinpath(
        "vae_clusters_within_margins_complete.tsv"
    )

    write_clusters_and_bins(
        vamb_options,
        cluster_options.binsplitter,
        str(vamb_options.out_dir.joinpath("vae_clusters_within_margins_complete")),
        vamb_options.out_dir.joinpath("bins_within_margins_complete"),
        comp_options.path,
        merged_cl_cs_d,
        comp_metadata.identifiers,
        comp_metadata.lengths,
    )
    logger.info(f"\tClustered contigs within neighs and outside neighs.\n")

    # Cluster, save tsv file
    clusterspath = vamb_options.out_dir.joinpath("vae_clusters.tsv")
    cluster_and_write_files(
        vamb_options,
        cluster_options,
        str(vamb_options.out_dir.joinpath("vae_clusters")),
        vamb_options.out_dir.joinpath("bins"),
        comp_options.path,
        latent,
        comp_metadata.identifiers,  # type:ignore
        comp_metadata.lengths,  # type:ignore
    )
    del latent


def cluster_neighs_based(neighs_object, latents, contignames, min_neighbourhood_size=2):
    c2idx = {contigname: i for i, contigname in enumerate(contignames)}
    idx2c = {i: contigname for i, contigname in enumerate(contignames)}

    mask_contigs_already_clustered = np.zeros(len(contignames), dtype=bool)

    neighbourhoods_graph = nx.Graph()
    for c_i, neigh_idxs in enumerate(neighs_object):
        if c_i in neigh_idxs and len(neigh_idxs) == 1:
            continue

        if c_i not in neigh_idxs and len(neigh_idxs) == 0:
            continue

        # assert c_i in neigh_idxs

        for n_i in neigh_idxs:
            neighbourhoods_graph.add_edge(idx2c[c_i], idx2c[n_i])
            mask_contigs_already_clustered[n_i] = True
            mask_contigs_already_clustered[c_i] = True
    contigs_in_neighbourhoods = [
        c for cs in nx.connected_components(neighbourhoods_graph) for c in cs
    ]
    assert len(contigs_in_neighbourhoods) == len(set(contigs_in_neighbourhoods))

    print("Contigs in neighbourhoods: %i" % len(contigs_in_neighbourhoods))

    withinmarginclusters_g = nx.Graph()

    for cs in nx.connected_components(neighbourhoods_graph):
        # print(cs)
        if len(cs) < 2:
            continue
        cs_list = list(cs)

        for i, c_i in enumerate(cs_list):
            # mask_contigs_already_clustered[c2idx[c_i]] = True
            for c_j in cs_list[i + 1 :]:
                assert c2idx[c_i] != c2idx[c_j]
                withinmarginclusters_g.add_edge(c_i, c_j)
                # print(c_i, c_j)
            #    mask_contigs_already_clustered[c2idx[c_j]] = True
        for c in cs_list:
            assert c in withinmarginclusters_g.nodes()
        for c in cs:
            assert c in withinmarginclusters_g.nodes()
        # break
        idxs_cs = [c2idx[c] for c in cs]

        limit_h = np.max(latents[idxs_cs], axis=0)
        limit_l = np.min(latents[idxs_cs], axis=0)

        limits = [(l, h) for h, l in zip(limit_h, limit_l)]

        idxs_not_cs = set(np.arange(len(contignames))) - set(idxs_cs)

        for idx_not_cs in idxs_not_cs:
            if all(
                limits[i][0] <= latents[idx_not_cs][i] <= limits[i][1]
                for i in range(latents.shape[1])
            ):
                extra_contig = contignames[idx_not_cs]

                withinmarginclusters_g.add_edge(extra_contig, cs_list[-1])
                mask_contigs_already_clustered[idx_not_cs] = True

    for c in contignames[mask_contigs_already_clustered]:
        assert c in withinmarginclusters_g.nodes()

    withinmarginclusters_cs_d = {
        "nneighs_%i" % i: cs
        for i, cs in enumerate(nx.connected_components(withinmarginclusters_g))
    }
    print(
        "Contigs already clustered mask sum %i" % np.sum(mask_contigs_already_clustered)
    )
    print(
        "Contigs in withinmargin_d %i, %i, %i"
        % (
            len(set([c for cs in withinmarginclusters_cs_d.values() for c in cs])),
            len([c for cs in withinmarginclusters_cs_d.values() for c in cs]),
            len(withinmarginclusters_g.nodes()),
        )
    )

    return withinmarginclusters_cs_d, mask_contigs_already_clustered


class BasicArguments(object):
    LOGS_PATH = "log.txt"

    def __init__(self, args):
        self.args = args
        self.comp_options = CompositionOptions(
            self.args.fasta,
            self.args.composition,
            self.args.minlength,
            args.warn_on_few_seqs,
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

    def run_inner(self):
        raise NotImplementedError

    @logger.catch()
    def run(self):
        torch.set_num_threads(self.vamb_options.n_threads)
        try_make_dir(self.vamb_options.out_dir)
        logger.add(
            self.vamb_options.out_dir.joinpath(self.LOGS_PATH), format=format_log
        )
        logger.add(sys.stderr, format=format_log)
        begintime = time.time()
        logger.info("Starting Vamb version " + ".".join(map(str, vamb.__version__)))
        logger.info("Random seed is " + str(self.vamb_options.seed))
        self.run_inner()
        logger.info(f"Completed Vamb in {round(time.time() - begintime, 2)} seconds.")


class TaxometerArguments(BasicArguments):
    def __init__(self, args):
        super(TaxometerArguments, self).__init__(args)
        self.taxonomy_options = TaxonomyOptions(
            taxonomy_path=args.taxonomy,
            no_predictor=False,
            column_contigs=args.column_contigs,
            column_taxonomy=args.column_taxonomy,
            delimiter_taxonomy=args.delimiter_taxonomy,
        )
        self.predictor_training_options = PredictorTrainingOptions(
            nepochs=args.pred_nepochs,
            batchsize=args.pred_batchsize,
            batchsteps=[],
            softmax_threshold=args.pred_softmax_threshold,
            ploss=args.ploss,
        )

    def run_inner(self):
        run_taxonomy_predictor(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            taxonomy_options=self.taxonomy_options,
            predictor_training_options=self.predictor_training_options,
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

    def run_inner(self):
        run(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            encoder_options=self.encoder_options,
            training_options=self.training_options,
            cluster_options=self.cluster_options,
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

    def run_inner(self):
        run(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            encoder_options=self.encoder_options,
            training_options=self.training_options,
            cluster_options=self.cluster_options,
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
            no_predictor=self.args.no_predictor,
            column_contigs=args.column_contigs,
            column_taxonomy=args.column_taxonomy,
            delimiter_taxonomy=args.delimiter_taxonomy,
        )
        self.predictor_training_options = PredictorTrainingOptions(
            nepochs=self.args.pred_nepochs,
            batchsize=self.args.pred_batchsize,
            batchsteps=[],
            softmax_threshold=self.args.pred_softmax_threshold,
            ploss=self.args.ploss,
        )

    def run_inner(self):
        run_vaevae(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            taxonomy_options=self.taxonomy_options,
            vae_training_options=self.vae_training_options,
            predictor_training_options=self.predictor_training_options,
            cluster_options=self.cluster_options,
            encoder_options=self.encoder_options,
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
            taxonomy_path=self.preds,
            no_predictor=False,
            column_contigs=args.column_contigs,
            column_taxonomy=args.column_taxonomy,
            delimiter_taxonomy=args.delimiter_taxonomy,
        )

    def run_inner(self):
        run_reclustering(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            reclustering_options=self.reclustering_options,
            taxonomy_options=self.taxonomy_options,
        )


class VAEASYarguments(BinnerArguments):
    def __init__(self, args):
        super(VAEASYarguments, self).__init__(args)
        self.encoder_options = EncoderOptions(
            vae_options=self.vae_options,
            aae_options=None,
            alpha=self.args.alpha,
        )

        self.embeddings_options = EmbeddingsOptions(
            embeddingspath=args.embeds,
            embeddedcontigspath=args.contigs_embedded,
            shuffle_embeds=args.shuffle_embeds,
            embeds_loss=args.embeds_loss,
            radius=args.R,
            gamma=args.gamma,
            neighs_object_path=args.neighs,
            embeddings_mask_path=args.embeds_mask,
            embeddings_processed_path=args.embeds_processed,
            margin=args.margin,
        )

        self.training_options = TrainingOptions(
            encoder_options=self.encoder_options,
            vae_options=self.vae_training_options,
            aae_options=None,
            lrate=self.args.lrate,
        )

        self.dbs_cluster_options = DBSCluster_Options(
            args.binsplit_separator,
        )

    def run_inner(self):
        run_n2v_asimetric(
            vamb_options=self.vamb_options,
            comp_options=self.comp_options,
            abundance_options=self.abundance_options,
            embeddings_options=self.embeddings_options,
            encoder_options=self.encoder_options,
            training_options=self.training_options,
            cluster_options=self.cluster_options,
            dbs_cluster_options=self.dbs_cluster_options,
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
        default=int.from_bytes(os.urandom(7), "little"),
        help="Random seed (random determinism not guaranteed)",
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


def add_vae_n2v_asy_arguments(subparser):
    # Embeddings arguments
    vae_asy_os = subparser.add_argument_group(title="Asymmetric vae arguments")
    vae_asy_os.add_argument(
        "--embeds",
        metavar="",
        type=Path,
        help="paths to graph embeddings ",
    )
    vae_asy_os.add_argument(
        "--contigs_embedded",
        metavar="",
        type=Path,
        help="paths to contigs that were embedded",
    )
    vae_asy_os.add_argument(
        "--shuffle_embeds",
        help="shuffle embeddings [False]",
        action="store_true",
    )
    vae_asy_os.add_argument(
        "--embeds_loss",
        help="Reconstruction loss for the embeddings [cos]",
        type=str,
        default="cos",
    )

    vae_asy_os.add_argument(
        "-R",
        help="radius neighbours embeddings",
        type=float,
        default=0.2,
    )
    vae_asy_os.add_argument(
        "--embeds_processed",
        metavar="",
        type=Path,
        help="paths to graph embeddings already proprocessed, so they match the contigs dimeniosn",
    )

    vae_asy_os.add_argument(
        "--embeds_mask",
        metavar="",
        type=Path,
        help="paths to graph embeddings mask so True if contig has neighbour within radious ",
    )
    vae_asy_os.add_argument(
        "--neighs",
        metavar="",
        type=Path,
        help="paths to graph neighs obj, where each row (contig) indicates the neibouring contig indices",
    )

    vae_asy_os.add_argument(
        "--gamma",
        help="Weight for the embeddings contrastive losss",
        type=float,
        default=0.1,
    )
    vae_asy_os.add_argument(
        "--margin",
        help="Margin where cosine distances stop being penalized",
        type=float,
        default=0.01,
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
        default=1024,
        help="batch size for the taxonomy predictor [1024]",
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
        help='Hierarchical loss (one of flat_softmax, cond_softmax, soft_margin) ["flat_softmax"]',
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
        # This means: None is not passed, "" if passed but empty, e.g. "-o -c 5"
        # otherwise a string.
        default=None,
        const="",
        nargs="?",
        help="binsplit separator [C] (pass empty string to disable)",
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
        choices=["vae", "aae", "vae-aae", "vaevae", "vae_asy"],
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


def add_taxonomy_arguments(subparser, taxonomy_only=False):
    # Taxonomy arguments
    taxonomys = subparser.add_argument_group(title="Taxonomy")
    taxonomys.add_argument(
        "--taxonomy",
        metavar="",
        type=Path,
        help="path to the taxonomy file",
    )
    taxonomys.add_argument(
        "--column_contigs",
        metavar="",
        type=str,
        default=None,
        help="the column in the provided taxonomy file that contains contig names, not needed if the file is in MMseqs2 output format [None]",
    )
    taxonomys.add_argument(
        "--column_taxonomy",
        metavar="",
        type=str,
        default=None,
        help='the column in the provided taxonomy file that contains taxonomy labels from domain to species or subspecies, separated with ";", not needed if the file is in MMseqs2 output format [None]',
    )
    taxonomys.add_argument(
        "--delimiter_taxonomy",
        metavar="",
        type=str,
        default="\t",
        help="the delimiter of the provided taxonomy file [\\t]",
    )
    if not taxonomy_only:
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


def add_vae_n2v_asy_arguments(subparser):
    # Embeddings arguments
    vae_asy_os = subparser.add_argument_group(title="Asymmetric vae arguments")
    vae_asy_os.add_argument(
        "--embeds",
        metavar="",
        type=Path,
        help="paths to graph embeddings ",
    )
    vae_asy_os.add_argument(
        "--contigs_embedded",
        metavar="",
        type=Path,
        help="paths to contigs that were embedded",
    )
    vae_asy_os.add_argument(
        "--shuffle_embeds",
        help="shuffle embeddings [False]",
        action="store_true",
    )
    vae_asy_os.add_argument(
        "--embeds_loss",
        help="Reconstruction loss for the embeddings [cos]",
        type=str,
        default="cos",
    )

    vae_asy_os.add_argument(
        "-R",
        help="radius neighbours embeddings",
        type=float,
        default=0.2,
    )
    vae_asy_os.add_argument(
        "--embeds_processed",
        metavar="",
        type=Path,
        help="paths to graph embeddings already proprocessed, so they match the contigs dimeniosn",
    )

    vae_asy_os.add_argument(
        "--embeds_mask",
        metavar="",
        type=Path,
        help="paths to graph embeddings mask so True if contig has neighbour within radious ",
    )
    vae_asy_os.add_argument(
        "--neighs",
        metavar="",
        type=Path,
        help="paths to graph neighs obj, where each row (contig) indicates the neibouring contig indices",
    )

    vae_asy_os.add_argument(
        "--gamma",
        help="Weight for the embeddings contrastive losss",
        type=float,
        default=0.1,
    )
    vae_asy_os.add_argument(
        "--margin",
        help="Margin where cosine distances stop being penalized",
        type=float,
        default=0.01,
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
    VAMB_ASY = "vae_asy"

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
    add_taxonomy_arguments(recluster_parser)

    vae_asy_parser = subparsers_model.add_parser(
        VAMB_ASY,
        help="""
        Variational Autoencoder binner that besides coabundances and TNF also leverages
        assembly graph information.
        """,
    )
    add_input_output_arguments(vae_asy_parser)
    add_vae_arguments(vae_asy_parser)
    add_vae_n2v_asy_arguments(vae_asy_parser)
    add_clustering_arguments(vae_asy_parser)

    args = parser.parse_args()

    if args.subcommand == TAXOMETER:
        args.warn_on_few_seqs = True
        runner = TaxometerArguments(args)
    elif args.subcommand == BIN:
        args.warn_on_few_seqs = True
        if args.model_subcommand is None:
            vaevae_parserbin_parser.print_help()
            sys.exit(1)
        classes_map = {
            VAMB: VAEArguments,
            AVAMB: VAEAAEArguments,
            TAXVAMB: VAEVAEArguments,
            VAMB_ASY: VAEASYarguments,
        }
        runner = classes_map[args.model_subcommand](args)
    elif args.subcommand == RECLUSTER:
        # Uniquely, the reclustering cannot overfit, so we don't need this warning
        args.warn_on_few_seqs = False
        runner = ReclusteringArguments(args)
    else:
        # There are no more subcommands
        assert False
    runner.run()


if __name__ == "__main__":
    main()
