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
import pycoverm
import itertools
from math import isfinite
from typing import Optional, Tuple, Union, cast, Callable, Literal, NamedTuple
from pathlib import Path
from collections.abc import Sequence
from torch.utils.data import DataLoader
from functools import partial
from loguru import logger
import datetime
import networkx as nx

_ncpu = os.cpu_count()
DEFAULT_THREADS = 8 if _ncpu is None else min(_ncpu, 8)

# This is the smallest number of sequences that can be accepted.
# A smaller number than this cause issues either during normalization of the Abundance
# object, or causes PyTorch to throw dimension errors, as singleton dimensions
# are treated differently.
MINIMUM_SEQS = 2

# These MUST be set before importing numpy
# I know this is a shitty hack, see https://github.com/numpy/numpy/issues/11826
os.environ["MKL_NUM_THREADS"] = str(DEFAULT_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(DEFAULT_THREADS)
os.environ["OMP_NUM_THREADS"] = str(DEFAULT_THREADS)

# Append vamb to sys.path to allow vamb import even if vamb was not installed
# using pip
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)


def check_existing_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def format_log(record) -> str:
    colors = {"WARNING": "red", "INFO": "green", "DEBUG": "blue", "ERROR": "red"}
    L = colors.get(record["level"].name, "blue")
    T = "red" if record["level"].name in ("WARNING", "ERROR") else "cyan"
    message = "<red>{message}</red>" if T == "red" else "{message}"
    time = f"<{T}>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</{T}>"
    level = f"<b><{L}>{{level:<7}}</{L}></b>"
    return f"{time} | {level} | {message}\n{{exception}}"


def typeasserted(x, Ty: Union[type, tuple[type, ...]]):
    assert isinstance(x, Ty)
    return x


def try_make_dir(name: Union[Path, str]):
    try:
        os.mkdir(name)
    except FileExistsError:
        pass


class FASTAPath:
    def __init__(self, path: Path):
        self.path = check_existing_file(path)


class CompositionPath:
    def __init__(self, path: Path):
        self.path = check_existing_file(path)


class CompositionOptions:
    __slots__ = ["path"]

    @staticmethod
    def are_args_present(args: argparse.Namespace) -> bool:
        return isinstance(args.fasta, Path) or isinstance(args.composition, Path)

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(
            typeasserted(args.fasta, (Path, type(None))),
            typeasserted(args.composition, (Path, type(None))),
        )

    def __init__(
        self,
        fastapath: Optional[Path],
        npzpath: Optional[Path],
    ):
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


class AbundancePath:
    def __init__(self, path: Path):
        self.path = check_existing_file(path)


class BAMPaths:
    @classmethod
    def from_dir(cls, dir: Path, min_alignment_id: Optional[float]):
        if not dir.is_dir():
            raise NotADirectoryError(dir)
        paths = [b for b in dir.iterdir() if b.is_file() and b.suffix == ".bam"]
        return cls(paths, min_alignment_id)

    def __init__(self, paths: list[Path], min_alignment_id: Optional[float]):
        if len(paths) == 0:
            raise ValueError(
                "No BAM files passed. If `--bamdir` was used, check directory is not empty, "
                "and all BAM files ends with the extension `.bam`"
            )
        for path in paths:
            check_existing_file(path)
            if not pycoverm.is_bam_sorted(str(path)):
                raise ValueError(f"Path {path} is not sorted by reference.")

        if min_alignment_id is not None:
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
        self.paths = paths


class NeighsPath:
    def __init__(self, path: Path):
        self.path = check_existing_file(path)


class AbundanceTSVPath:
    __slots__ = ["path"]

    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path


class AbundanceOptions:
    __slots__ = ["paths"]

    @staticmethod
    def are_args_present(args: argparse.Namespace) -> bool:
        return (
            isinstance(args.bampaths, list)
            or isinstance(args.bamdir, Path)
            or isinstance(args.abundance_tsv, Path)
            or isinstance(args.abundancepath, Path)
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(
            typeasserted(args.bampaths, (list, type(None))),
            typeasserted(args.bamdir, (Path, type(None))),
            typeasserted(args.abundance_tsv, (Path, type(None))),
            typeasserted(args.abundancepath, (Path, type(None))),
            typeasserted(args.min_alignment_id, (float, type(None))),
        )
    def __init__(
        self,
        bampaths: Optional[list[Path]],
        bamdir: Optional[Path],
        abundance_tsv: Optional[Path],
        abundancepath: Optional[Path],
        min_alignment_id: Optional[float],
    ):
        # Make sure only one abundance input is there
        if (
            (bampaths is not None)
            + (abundancepath is not None)
            + (bamdir is not None)
            + (abundance_tsv is not None)
        ) != 1:
            raise argparse.ArgumentTypeError(
                "Must specify exactly one of BAM files, BAM dir, TSV file or abundance NPZ file input"
            )

        if abundancepath is not None:
            self.paths = AbundancePath(abundancepath)
        elif bamdir is not None:
            self.paths = BAMPaths.from_dir(bamdir, min_alignment_id)
        elif bampaths is not None:
            logger.warning(
                "The --bamfiles argument is deprecated. It works, but might be removed in future versions of Vamb. Please use --bamdir instead"
            )
            self.paths = BAMPaths(bampaths, min_alignment_id)
        elif abundance_tsv is not None:
            self.paths = AbundanceTSVPath(abundance_tsv)

class NeighsOptions:
    __slots__ = [
        "neighs_path",
        "gamma",
        "margin",
        "radius_clustering",
        "top_neighbours",
        "max_neighs_r",
    ]
    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
    ):
        return cls(
            typeasserted(args.neighs_path, Path),
            typeasserted(args.gamma, float),
            typeasserted(args.margin, float),
            typeasserted(args.radius_clustering, float),
            typeasserted(args.top_neighbours, int),
            typeasserted(args.max_neighs_r, float),
        )



    def __init__(
        self,           
        neighs_path: Path,
        gamma: float =0.1,
        margin: float = 0.01,
        radius_clustering: float = 0.01,
        top_neighbours: int = 50,
        max_neighs_r: float = 0.2,
    ):
        
        assert isinstance(neighs_path, Path)
        assert isinstance(gamma, float)
        assert isinstance(margin, float)
        assert isinstance(radius_clustering, float)
        assert isinstance(top_neighbours, int)
        assert isinstance(max_neighs_r, float)
        
        self.margin = margin
        self.radius_clustering = radius_clustering
        self.neighs_path = neighs_path
        self.gamma = gamma
        self.top_neighbours = top_neighbours
        self.max_neighs_r = max_neighs_r


class BasicTrainingOptions:
    __slots__ = ["num_epochs", "starting_batch_size", "batch_steps"]

    @classmethod
    def from_args_vae(cls, args: argparse.Namespace):
        return cls(
            typeasserted(args.nepochs, int),
            typeasserted(args.batchsize, int),
            typeasserted(args.batchsteps, list),
        )

    @classmethod
    def from_args_aae(cls, args: argparse.Namespace):
        return cls(
            typeasserted(args.nepochs_aae, int),
            typeasserted(args.batchsize_aae, int),
            typeasserted(args.batchsteps_aae, list),
        )

    @classmethod
    def from_args_taxometer(cls, args: argparse.Namespace):
        return cls(
            typeasserted(args.pred_nepochs, int),
            typeasserted(args.pred_batchsize, int),
            [],
        )


    def __init__(
        self, 
        num_epochs: int, 
        starting_batch_size: int, 
        batch_steps: list[int]
    ):
        if num_epochs < 1:
            raise argparse.ArgumentTypeError(f"Minimum 1 epoch, not {num_epochs}")
        self.num_epochs = num_epochs

        if starting_batch_size < 1:
            raise argparse.ArgumentTypeError(
                f"Minimum batchsize of 1, not {starting_batch_size}"
            )
        self.starting_batch_size = starting_batch_size

        batch_steps = sorted(set(batch_steps))
        if max(batch_steps, default=0) >= self.num_epochs:
            raise argparse.ArgumentTypeError("All batchsteps must be less than nepochs")

        if min(batch_steps, default=1) < 1:
            raise argparse.ArgumentTypeError("All batchsteps must be 1 or higher")
        self.batch_steps = batch_steps


# class NeighsOptions:
#     __slots__ = [
#         "neighs_path",
#         "gamma",
#         "margin",
#         "radius_clustering",
#         "top_neighbours",
#         "max_neighs_r",
#     ]

#     def __init__(
#         self,           
#         neighs_path: Path,
#         gamma: float =0.1 ,
#         margin: float = 0.01,
#         radius_clustering: float = 0.01,
#         top_neighbours: int = 50,
#         max_neighs_r: float = 0.2,
#     ):
        
#         assert isinstance(neighs_path, Path)
#         assert isinstance(gamma, float)
#         assert isinstance(margin, float)
#         assert isinstance(radius_clustering, float)
#         assert isinstance(top_neighbours, int)
#         assert isinstance(max_neighs_r, float)
        
#         self.margin = margin
#         self.radius_clustering = radius_clustering
#         self.neighs_path = NeighsPath(neighs_path)
#         self.gamma = gamma
#         self.top_neighbours = top_neighbours
#         self.max_neighs_r = max_neighs_r

class VAEOptions:
    __slots__ = ["nhiddens", "nlatent", "alpha",
                "beta", "dropout", "basic_options"]

    @classmethod
    def from_args(cls, basic: BasicTrainingOptions, args: argparse.Namespace):
        return cls(
            basic,
            typeasserted(args.nhiddens, (list, type(None))),
            typeasserted(args.nlatent, int),
            typeasserted(args.alpha, (float, type(None))),
            typeasserted(args.beta, float),
            typeasserted(args.dropout, (float, type(None))),
        )

    def __init__(
        self,
        basic_options: BasicTrainingOptions,
        nhiddens: Optional[list[int]],
        nlatent: int,
        alpha: Optional[float],
        beta: float,
        dropout: Optional[float],
    ):
        self.basic_options = basic_options

        if nhiddens is not None and (
            len(nhiddens) == 0 or any(i < 1 for i in nhiddens)
        ):
            raise argparse.ArgumentTypeError(
                f"Minimum 1 neuron per layer, not {min(nhiddens)}"
            )
        self.nhiddens = nhiddens

        if nlatent < 1:
            raise argparse.ArgumentTypeError(f"Minimum 1 latent neuron, not {nlatent}")
        self.nlatent = nlatent

        if alpha is not None and ((not isfinite(alpha)) or alpha <= 0 or alpha >= 1):
            raise argparse.ArgumentTypeError("alpha must be above 0 and below 1")
        self.alpha = alpha

        if beta <= 0 or not isfinite(beta):
            raise argparse.ArgumentTypeError("beta cannot be negative or zero")
        self.beta = beta

        if dropout is not None and (dropout < 0 or dropout >= 1):
            raise argparse.ArgumentTypeError("dropout must be in 0 <= d < 1")
        self.dropout = dropout

class GeneralOptions:
    __slots__ = [
        "out_dir",
        "min_contig_length",
        "n_threads",
        "refcheck",
        "seed",
        "cuda",
    ]

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(
            typeasserted(args.outdir, Path),
            typeasserted(args.minlength, int),
            typeasserted(args.nthreads, int),
            not typeasserted(args.norefcheck, bool),
            typeasserted(args.seed, int),
            typeasserted(args.cuda, bool),
        )

    def __init__(
        self,
        out_dir: Path,
        min_contig_length: int,
        n_threads: int,
        refcheck: bool,
        seed: int,
        cuda: bool,
    ):
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

        if min_contig_length < 250:
            raise argparse.ArgumentTypeError(
                "Minimum contig length must be at least 250"
            )
        self.min_contig_length = min_contig_length

        if cuda and not torch.cuda.is_available():
            raise ModuleNotFoundError(
                "Cuda is not available on your PyTorch installation"
            )
        self.seed = seed
        self.cuda = cuda
        self.refcheck = refcheck


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
        binsplit_separator: Optional[str],
        algorithm: str,
        hmmout_path: Path = "",
    ):
        assert isinstance(latent_path, Path)
        assert isinstance(clusters_path, Path)
        #assert isinstance(hmmout_path, Path)

        for path in [latent_path, clusters_path]:
            if not path.is_file():
                raise FileNotFoundError(path)

        if algorithm not in ("kmeans", "dbscan"):
            raise ValueError("Algortihm must be 'kmeans' or 'dbscan'")

        self.latent_path = latent_path
        self.clusters_path = clusters_path
        self.hmmout_path = hmmout_path
        self.algorithm = algorithm
        self.binsplitter = vamb.vambtools.BinSplitter(binsplit_separator)


# class EncoderOptions:
#     __slots__ = ["vae_options", "aae_options", "alpha"]

#     def __init__(
#         self,
#         vae_options: Optional[VAEOptions],
#         aae_options: Optional[AAEOptions],
#         alpha: Optional[float],
#     ):
#         assert isinstance(alpha, (float, type(None)))
#         assert isinstance(vae_options, (VAEOptions, type(None)))
#         assert isinstance(aae_options, (AAEOptions, type(None)))

#         if alpha is not None and (alpha <= 0 or alpha) >= 1:
#             raise argparse.ArgumentTypeError("alpha must be above 0 and below 1")
#         self.alpha = alpha

#         if vae_options is None and aae_options is None:
#             raise argparse.ArgumentTypeError(
#                 "You must specify either VAE or AAE or both as encoders, cannot run with no encoders."
#             )

#         self.vae_options = vae_options
#         self.aae_options = aae_options


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


class TaxonomyBase:
    __slots__ = ["path"]

    def __init__(self, path: Path):
        self.path = path


class RefinedTaxonomy(TaxonomyBase):
    pass


class UnrefinedTaxonomy(TaxonomyBase):
    pass


def get_taxonomy(args: argparse.Namespace) -> Union[RefinedTaxonomy, UnrefinedTaxonomy]:
    path = args.taxonomy
    if path is None:
        raise ValueError(
            "Cannot load taxonomy for Taxometer without specifying --taxonomy"
        )
    with open(check_existing_file(path)) as file:
        try:
            header = next(file).rstrip("\r\n")
        except StopIteration:
            header = None

    if header is None:
        raise ValueError(f'Empty taxonomy path at "{path}"')
    elif header == vamb.taxonomy.TAXONOMY_HEADER:
        return UnrefinedTaxonomy(path)
    elif header == vamb.taxonomy.PREDICTED_TAXONOMY_HEADER:
        return RefinedTaxonomy(path)
    else:
        raise ValueError(
            f'ERROR: When reading taxonomy file at "{path}", '
            f"the first line was not either {repr(vamb.taxonomy.TAXONOMY_HEADER)} "
            f"or {repr(vamb.taxonomy.PREDICTED_TAXONOMY_HEADER)}'"
        )


class TaxometerOptions:
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        tax = get_taxonomy(args)
        if isinstance(tax, RefinedTaxonomy):
            raise ValueError(
                f'Attempted to run Taxometer to refine taxonomy at "{args.taxonomy}", '
                "but this file appears to already be an output of Taxometer"
            )
        return cls(
            GeneralOptions.from_args(args),
            CompositionOptions.from_args(args),
            AbundanceOptions.from_args(args),
            tax,
            BasicTrainingOptions.from_args_taxometer(args),
            typeasserted(args.pred_softmax_threshold, float),
            args.ploss,
        )

    def __init__(
        self,
        general: GeneralOptions,
        composition: CompositionOptions,
        abundance: AbundanceOptions,
        taxonomy: UnrefinedTaxonomy,
        basic: BasicTrainingOptions,
        softmax_threshold: float,
        ploss: Union[
            Literal["float_softmax"], Literal["cond_softmax"], Literal["soft_margin"]
        ],
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
        self.general = general
        self.taxonomy = taxonomy
        self.ploss = ploss
        self.softmax_threshold = softmax_threshold
        self.composition = composition
        self.abundance = abundance
        self.basic = basic


class MarkerPath:
    def __init__(self, path: Path):
        self.path = check_existing_file(path)


class PredictMarkers:
    def __init__(self, hmm: Path, comp: CompositionOptions):
        self.hmm = check_existing_file(hmm)
        if not isinstance(comp.path, FASTAPath):
            raise ValueError(
                "If markers is predicted via --hmm, composition must be passed with --fasta"
            )
        self.fasta = comp.path


class MarkerOptions:
    @classmethod
    def from_args(cls, comp: CompositionOptions, args):
        markers = typeasserted(args.markers, (Path, type(None)))
        hmm = typeasserted(args.hmm_path, (Path, type(None)))
        if not (markers is None) ^ (hmm is None):
            raise ValueError(
                "Exactly one of --markers or --hmm_path must be set to input taxonomic information"
            )
        if markers is not None:
            return cls(MarkerPath(markers))
        else:
            assert isinstance(hmm, Path)  # we just checked this above
            return cls(PredictMarkers(hmm, comp))

    def __init__(self, markers: Union[MarkerPath, PredictMarkers]):
        self.content = markers


class KmeansOptions:
    def __init__(self, clusters: Path):
        self.clusters = check_existing_file(clusters)


class DBScanOptions:
    @classmethod
    def from_args(cls, args: argparse.Namespace, n_threads: int):
        tax = get_taxonomy(args)
        predict = not typeasserted(args.no_predictor, bool)
        if predict:
            if isinstance(tax, RefinedTaxonomy):
                logger.warning(
                    "Flag --no_predictor not set, but the taxonomy passed in "
                    "on --taxonomy is already refined. Skipped refinement."
                )
                return cls(tax, n_threads)
            else:
                if not (
                    CompositionOptions.are_args_present(args)
                    and AbundanceOptions.are_args_present(args)
                ):
                    raise ValueError(
                        "Flag --no_predictor is not set, but abundance "
                        "or composition has not been passed in, so there is no information "
                        "to refine taxonomy with"
                    )
                tax_options = TaxometerOptions(
                    GeneralOptions.from_args(args),
                    CompositionOptions.from_args(args),
                    AbundanceOptions.from_args(args),
                    tax,
                    BasicTrainingOptions.from_args_taxometer(args),
                    typeasserted(args.pred_softmax_threshold, float),
                    args.ploss,
                )
                return cls(tax_options, n_threads)
        else:
            return cls(tax, n_threads)

    def __init__(
        self,
        ops: Union[
            UnrefinedTaxonomy,
            RefinedTaxonomy,
            TaxometerOptions,
        ],
        n_processes: int,
    ):
        self.taxonomy = ops
        self.n_processes = n_processes


class ClusterOptions:
    __slots__ = [
        "window_size",
        "min_successes",
        "max_clusters",
    ]

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(
            typeasserted(args.window_size, int),
            typeasserted(args.min_successes, int),
            typeasserted(args.max_clusters, (int, type(None))),
        )

    def __init__(
        self,
        window_size: int,
        min_successes: int,
        max_clusters: Optional[int],
    ):
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


class AAEOptions:
    @classmethod
    def from_args(
        cls,
        basic: BasicTrainingOptions,
        clustering: ClusterOptions,
        args: argparse.Namespace,
    ):
        return cls(
            basic,
            clustering,
            typeasserted(args.nhiddens_aae, int),
            typeasserted(args.nlatent_aae_z, int),
            typeasserted(args.nlatent_aae_y, int),
            typeasserted(args.sl, float),
            typeasserted(args.slr, float),
            typeasserted(args.temp, float),
        )

    def __init__(
        self,
        basic: BasicTrainingOptions,
        clustering: ClusterOptions,
        nhiddens: int,
        nlatent_z: int,
        nlatent_y: int,
        sl: float,
        slr: float,
        temp: float,
    ):
        # For the VAE-AAE workflow, we must cluster the full set of sequences,
        # else the cluster dereplication step makes no sense, as the different
        # clusterings will contain different sets of clusters.
        # So, enforce this here
        if clustering.max_clusters is not None:
            raise ValueError(
                "When using the VAE-AAE model, `max_clusters` (option `-c`) "
                "must not be set."
            )

        self.basic = basic
        if nhiddens < 1:
            raise argparse.ArgumentTypeError(
                f"Minimum 1 neuron in the hidden layers, not {nhiddens}"
            )

        self.nhiddens = nhiddens

        if nlatent_z < 1:
            raise argparse.ArgumentTypeError(
                f"Minimum 1 latent neuron, not {nlatent_z}"
            )
        self.nlatent_z = nlatent_z

        if nlatent_y < 1:
            raise argparse.ArgumentTypeError(
                f"Minimum 1 latent neuron, not {nlatent_y}"
            )
        self.nlatent_y = nlatent_y
        self.temp = temp
        self.sl = sl
        self.slr = slr


class BinOutputOptions:
    __slots__ = ["binsplitter", "min_fasta_output_size"]

    # We take a composition as arguments, because if min_fasta_output_size is set,
    # we need to guarantee that the composition is passed as fasta.
    # Similarly, we currently don't have any workflows that can use BinOutputOptions
    # without using composition.

    @classmethod
    def from_args(cls, comp: CompositionOptions, args: argparse.Namespace):
        return cls(
            comp,
            typeasserted(args.binsplit_separator, (str, type(None))),
            typeasserted(args.min_fasta_output_size, (int, type(None))),
        )

    def __init__(
        self,
        composition: CompositionOptions,
        binsplit_separator: Optional[str],
        min_fasta_output_size: Optional[int],
    ):
        self.binsplitter = vamb.vambtools.BinSplitter(binsplit_separator)
        if min_fasta_output_size is not None:
            if not isinstance(composition.path, FASTAPath):
                raise argparse.ArgumentTypeError(
                    "If minfasta is not None, input fasta file must be given explicitly"
                )
            if min_fasta_output_size < 0:
                raise argparse.ArgumentTypeError(
                    "Minimum FASTA output size must be nonnegative"
                )
        self.min_fasta_output_size = min_fasta_output_size


@logger.catch(reraise=True)
def run(
    runner: Callable[[], None],
    general: GeneralOptions,
):
    torch.set_num_threads(general.n_threads)
    try_make_dir(general.out_dir)
    logger.add(general.out_dir.joinpath("log.txt"), format=format_log)
    begintime = time.time()
    logger.info("Starting Vamb version " + vamb.__version_str__)
    logger.info("Random seed is " + str(general.seed))
    logger.info(f"Invoked with CLI args: '{' '.join(sys.argv)}'")
    runner()
    logger.info(f"Completed Vamb in {round(time.time() - begintime, 2)} seconds.")


class BinnerCommonOptions:
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        comp = CompositionOptions.from_args(args)
        return cls(
            GeneralOptions.from_args(args),
            comp,
            AbundanceOptions.from_args(args),
            ClusterOptions.from_args(args),
            BinOutputOptions.from_args(comp, args),
        )

    # We do not have BasicTrainingOptions because that is model-specific
    def __init__(
        self,
        general: GeneralOptions,
        comp: CompositionOptions,
        abundance: AbundanceOptions,
        clustering: ClusterOptions,
        output: BinOutputOptions,
    ):
        self.general = general
        self.comp = comp
        self.abundance = abundance
        self.clustering = clustering
        self.output = output


class BinDefaultOptions:
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        common = BinnerCommonOptions.from_args(args)
        basic = BasicTrainingOptions.from_args_vae(args)
        return cls(
            common,
            VAEOptions.from_args(basic, args),
        )

    def __init__(
        self,
        common: BinnerCommonOptions,
        vae: VAEOptions,
    ):
        self.common = common
        self.vae = vae


class BinTaxVambOptions:
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        common = BinnerCommonOptions.from_args(args)
        basic = BasicTrainingOptions.from_args_vae(args)
        vae = VAEOptions.from_args(basic, args)
        taxonomy = get_taxonomy(args)
        predict = not typeasserted(args.no_predictor, bool)
        if predict:
            if isinstance(taxonomy, RefinedTaxonomy):
                logger.warning(
                    "Flag --no_predictor not set, but the taxonomy passed in "
                    "on --taxonomy is already refined. Skipped refinement."
                )
                tax = taxonomy
            else:
                tax = TaxometerOptions(
                    common.general,
                    common.comp,
                    common.abundance,
                    taxonomy,
                    basic,
                    typeasserted(args.pred_softmax_threshold, float),
                    args.ploss,
                )
        else:
            tax = taxonomy
        return cls(common, vae, tax)

    # The VAEVAE models share the same settings as the VAE model, so we just use VAEOptions
    def __init__(
        self,
        common: BinnerCommonOptions,
        vae: VAEOptions,
        taxonomy: Union[RefinedTaxonomy, TaxometerOptions, UnrefinedTaxonomy],
    ):
        self.common = common
        self.vae = vae
        self.taxonomy = taxonomy


class BinAvambOptions:
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        common = BinnerCommonOptions.from_args(args)
        basic_vae = BasicTrainingOptions.from_args_vae(args)
        basic_aae = BasicTrainingOptions.from_args_aae(args)
        clustering = ClusterOptions.from_args(args)
        
        return cls(
            common,
            VAEOptions.from_args(basic_vae, args),
            AAEOptions.from_args(basic_aae, clustering, args),
        )

    def __init__(
        self,
        common: BinnerCommonOptions,
        vae: VAEOptions,
        aae: AAEOptions,
    ):
        self.common = common
        self.vae = vae
        self.aae = aae


class BinContrVambOptions:
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        common = BinnerCommonOptions.from_args(args)
        basic = BasicTrainingOptions.from_args_vae(args)
        neighs = NeighsOptions.from_args(args)
        return cls(
            common,
            VAEOptions.from_args(basic, args),
            neighs
        )

    def __init__(
        self,
        common: BinnerCommonOptions,
        vae: VAEOptions,
        neighs: NeighsOptions,
    ):
        self.common = common
        self.vae = vae
        self.neighs = neighs 


class ReclusteringOptions:
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        general = GeneralOptions.from_args(args)
        latent = args.latent_path
        if latent is None:
            raise ValueError("--latent_path must be set")
        if args.algorithm == "kmeans":
            clusters = args.clusters_path
            if clusters is None:
                raise ValueError(
                    "If --algorithm is set to 'kmeans', --clusters_path must be set"
                )
            algorithm = KmeansOptions(clusters)
        elif args.algorithm == "dbscan":
            algorithm = DBScanOptions.from_args(args, general.n_threads)
        else:
            assert False  # no more algorithms

        # Avoid loading composition again if already loaded in DBScanOptions
        if isinstance(algorithm, DBScanOptions) and isinstance(
            algorithm.taxonomy, tuple
        ):
            comp = algorithm.taxonomy[1]
        else:
            comp = CompositionOptions.from_args(args)

        markers = MarkerOptions.from_args(comp, args)
        output = BinOutputOptions.from_args(comp, args)
        return cls(
            latent,
            comp,
            markers,
            general,
            output,
            algorithm,
        )

    def __init__(
        self,
        latent_path: Path,
        comp: CompositionOptions,
        markers: MarkerOptions,
        general: GeneralOptions,
        output: BinOutputOptions,
        algorithm: Union[KmeansOptions, DBScanOptions],
    ):
        self.latent_path = check_existing_file(latent_path)
        self.composition = comp
        self.markers = markers
        self.general = general
        self.output = output
        self.algorithm = algorithm


def calc_tnf(
    options: CompositionOptions,
    min_contig_length: int,
    outdir: Path,
    binsplitter: vamb.vambtools.BinSplitter,
) -> vamb.parsecontigs.Composition:
    begintime = time.time()
    logger.info("Loading TNF")
    logger.info(f"\tMinimum sequence length: {min_contig_length}")

    path = options.path

    if isinstance(path, CompositionPath):
        logger.info(f'\tLoading composition from npz at: "{path.path}"')
        composition = vamb.parsecontigs.Composition.load(path.path)
        composition.filter_min_length(min_contig_length)
    else:
        assert isinstance(path, FASTAPath)
        logger.info(f"\tLoading data from FASTA file {path.path}")
        with vamb.vambtools.Reader(path.path) as file:
            composition = vamb.parsecontigs.Composition.from_file(
                file, minlength=min_contig_length
            )
        composition.save(outdir.joinpath("composition.npz"))

    binsplitter.initialize(composition.metadata.identifiers)

    if composition.nseqs < MINIMUM_SEQS:
        err = (
            f"Found only {composition.nseqs} contigs, but Vamb currently requires at least "
            f"{MINIMUM_SEQS} to work correctly. "
            "If you have this few sequences in a metagenomic assembly, "
            "it's probably an error somewhere in your workflow."
        )
        logger.error(err)
        raise ValueError(err)

    # Warn the user if any contigs have been observed, which is smaller
    # than the threshold.
    if not np.all(composition.metadata.mask):
        n_removed = len(composition.metadata.mask) - np.sum(composition.metadata.mask)
        message = (
            f"The minimum sequence length has been set to {min_contig_length}, "
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


def calc_abundance(
    abundance_options: AbundanceOptions,
    outdir: Path,
    refcheck: bool,
    comp_metadata: vamb.parsecontigs.CompositionMetaData,
    nthreads: int,
) -> vamb.parsebam.Abundance:
    begintime = time.time()
    logger.info("Loading depths")
    logger.info(
        f"\tReference hash: {comp_metadata.refhash.hex() if refcheck else 'None'}"
    )

    paths = abundance_options.paths
    if isinstance(paths, AbundancePath):
        logger.info(f'\tLoading depths from npz at: "{str(paths.path)}"')

        abundance = vamb.parsebam.Abundance.load(
            paths.path,
            comp_metadata.refhash if refcheck else None,
        )
        # I don't want this check in any constructors of abundance, since the constructors
        # should be able to skip this check in case comp and abundance are independent.
        # But when running the main Vamb workflow, we need to assert this.
        if abundance.nseqs != comp_metadata.nseqs:
            assert not refcheck
            raise ValueError(
                f"Loaded abundance has {abundance.nseqs} sequences, "
                f"but composition has {comp_metadata.nseqs}."
            )
    elif isinstance(paths, BAMPaths):
        logger.info(f"\tParsing {len(paths.paths)} BAM files with {nthreads} threads")
        logger.info(f"\tMin identity: {paths.min_alignment_id}")

        abundance = vamb.parsebam.Abundance.from_files(
            list(paths.paths),
            outdir.joinpath("tmp").joinpath("pycoverm"),
            comp_metadata,
            refcheck,
            paths.min_alignment_id,
            nthreads,
        )
        abundance.save(outdir.joinpath("abundance.npz"))

        logger.info("\tOrder of columns is:")
        for i, samplename in enumerate(abundance.samplenames):
            logger.info(f"\t{i:>6}: {samplename}")
    elif isinstance(paths, AbundanceTSVPath):
        logger.info(f'\tParsing abundance from TSV at "{paths.path}"')
        abundance = vamb.parsebam.Abundance.from_tsv(
            paths.path,
            comp_metadata,
        )
        abundance.save(outdir.joinpath("abundance.npz"))
        logger.info("\tOrder of columns is:")
        for i, samplename in enumerate(abundance.samplenames):
            logger.info(f"\t{i:>6}: {samplename}")

    else:
        assert False

    elapsed = round(time.time() - begintime, 2)
    logger.info(f"\tProcessed abundance in {elapsed} seconds.\n")

    return abundance


def load_composition_and_abundance(
    vamb_options: GeneralOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    binsplitter: vamb.vambtools.BinSplitter,
) -> Tuple[vamb.parsecontigs.Composition, vamb.parsebam.Abundance]:
    composition = calc_tnf(
        comp_options, vamb_options.min_contig_length, vamb_options.out_dir, binsplitter
    )
    abundance = calc_abundance(
        abundance_options,
        vamb_options.out_dir,
        vamb_options.refcheck,
        composition.metadata,
        vamb_options.n_threads,
    )
    return (composition, abundance)


def load_markers(
    options: MarkerOptions,
    comp_metadata: vamb.parsecontigs.CompositionMetaData,
    existing_outdir: Path,
    n_threads: int,
) -> vamb.parsemarkers.Markers:
    logger.info("Loading markers")
    begin_time = time.time()
    if isinstance(options.content, MarkerPath):
        path = options.content.path
        logger.info(f'\tLoading markers from existing `markers.npz` at "{path}"')
        markers = vamb.parsemarkers.Markers.load(path, comp_metadata.refhash)
    elif isinstance(options.content, PredictMarkers):
        logger.info("\tPredicting markers. This might take some time")
        logger.info(f"\t\tFASTA file located at {options.content.fasta.path}")
        logger.info(
            f"\t\tHMM profile file (.hmm file) located at {options.content.hmm}"
        )
        markers = vamb.parsemarkers.Markers.from_files(
            options.content.fasta.path,
            options.content.hmm,
            cast(Sequence[str], comp_metadata.identifiers),
            existing_outdir.joinpath("tmp_markers"),
            n_threads,
            comp_metadata.refhash,
        )
        markers.save(existing_outdir.joinpath("markers.npz"))
    else:
        assert False

    elapsed = round(time.time() - begin_time, 2)
    logger.info(f"\tProcessed markers in {elapsed} seconds.\n")
    return markers


def trainvae(
    vae_options: VAEOptions,
    vamb_options: GeneralOptions,
    data_loader: DataLoader,
) -> np.ndarray:
    begintime = time.time()
    logger.info("Creating and training VAE")

    nsamples = data_loader.dataset.tensors[0].shape[1]  # type:ignore
    vae = vamb.encode.VAE(
        nsamples,
        nhiddens=vae_options.nhiddens,
        nlatent=vae_options.nlatent,
        alpha=vae_options.alpha,
        beta=vae_options.beta,
        dropout=vae_options.dropout,
        cuda=vamb_options.cuda,
        seed=vamb_options.seed,
    )

    logger.info("\tCreated VAE")
    modelpath = vamb_options.out_dir.joinpath("model.pt")
    vae.trainmodel(
        vamb.encode.set_batchsize(
            data_loader, vae_options.basic_options.starting_batch_size
        ),
        nepochs=vae_options.basic_options.num_epochs,
        batchsteps=vae_options.basic_options.batch_steps,
        modelfile=modelpath,
    )

    logger.info("\tEncoding to latent representation")
    latent = vae.encode(data_loader)
    vamb.vambtools.write_npz(vamb_options.out_dir.joinpath("latent.npz"), latent)
    del vae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    logger.info(f"\tTrained VAE and encoded in {elapsed} seconds.\n")

    return latent


def train_contr_vae(
    vae_options: VAEOptions,
    #training_options: VAETrainingOptions,
    vamb_options: GeneralOptions,
    neighs_options: NeighsOptions,
    #lrate: float,
    #alpha: Optional[float],
    data_loader: DataLoader,
    neighs_object: np.ndarray,
) -> np.ndarray:
    begintime = time.time()
    logger.info("Creating and training contrastive-VAE")

    nsamples = data_loader.dataset.tensors[0].shape[1]
    ncontigs = data_loader.dataset.tensors[0].shape[0]

    vae = vamb.encode_n2v_asimetric.VAE(
        nsamples,
        ncontigs,
        neighs_object,
        nhiddens=vae_options.nhiddens,
        nlatent=vae_options.nlatent,
        alpha=vae_options.alpha,
        beta=vae_options.beta,
        gamma=neighs_options.gamma,
        dropout=vae_options.dropout,
        cuda=vamb_options.cuda,
        seed=vamb_options.seed,
        margin=neighs_options.margin,
    )

    logger.info("Created contrastive-VAE")
    modelpath = vamb_options.out_dir.joinpath("model.pt")
    vae.trainmodel(
        vamb.encode.set_batchsize(
            data_loader, 
            vae_options.basic_options.starting_batch_size
            ),
        nepochs=vae_options.basic_options.num_epochs,
        #lrate=lrate, # use default value [1e-3] since vamb vae uses d-adaptation
        batchsteps=vae_options.basic_options.batch_steps,
        modelfile=modelpath    
        )

    logger.info("\nEncoding to latent representation")
    latent = vae.encode(data_loader)
    vamb.vambtools.write_npz(vamb_options.out_dir.joinpath("latent.npz"), latent)
    del vae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    logger.info(f"\nTrained contrastive-VAE and encoded in {elapsed} seconds.")

    return latent


def trainaae(
    data_loader: DataLoader,
    aae_options: AAEOptions,
    vamb_options: GeneralOptions,
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
        vamb.encode.set_batchsize(data_loader, aae_options.basic.starting_batch_size),
        aae_options.basic.num_epochs,
        aae_options.basic.batch_steps,
        aae_options.temp,
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


class FastaOutput(NamedTuple):
    existing_fasta_path: FASTAPath
    bins_dir_to_populate: Path  # (or to create, if not existing)
    min_fasta_size: int

    @classmethod
    def try_from_common(cls, common: BinnerCommonOptions):
        if common.output.min_fasta_output_size is not None:
            # We have checked this in the BinOutputOptions constructor
            assert isinstance(common.comp.path, FASTAPath)
            return cls(
                common.comp.path,
                common.general.out_dir.joinpath("bins"),
                common.output.min_fasta_output_size,
            )
        else:
            return None


def cluster_and_write_files(
    cluster_options: ClusterOptions,
    binsplitter: vamb.vambtools.BinSplitter,
    latent: np.ndarray,
    sequence_names: Sequence[str],
    sequence_lens: np.ndarray,
    seed: int,
    cuda: bool,
    base_clusters_name: str,  # e.g. /foo/bar/vae -> /foo/bar/vae_unsplit.tsv
    fasta_output: Optional[FastaOutput],
    bin_prefix: Optional[str],  # see write_clusters_and_bins
):
    begintime = time.time()
    # Create cluser iterator
    logger.info("Clustering")
    logger.info(f"\tWindowsize: {cluster_options.window_size}")
    logger.info(
        f"\tMin successful thresholds detected: {cluster_options.min_successes}",
    )
    logger.info(f"\tMax clusters: {cluster_options.max_clusters}")
    logger.info(f"\tUse CUDA for clustering: {cuda}")
    logger.info(f"\tBinsplitter: {binsplitter.log_string()}")

    cluster_generator = vamb.cluster.ClusterGenerator(
        latent,
        sequence_lens,
        windowsize=cluster_options.window_size,
        minsuccesses=cluster_options.min_successes,
        destroy=True,
        normalized=False,
        cuda=cuda,
        rng_seed=seed,
    )
    # This also works correctly when max_clusters is None
    clusters = itertools.islice(cluster_generator, cluster_options.max_clusters)
    cluster_dict: dict[str, set[str]] = dict()

    # Write the cluster metadata to file
    with open(Path(base_clusters_name + "_metadata.tsv"), "w") as file:
        print("name\tradius\tpeak valley ratio\tkind\tbp\tncontigs\tmedoid", file=file)
        for i, cluster in enumerate(clusters):
            cluster_dict[str(i + 1)] = {
                sequence_names[cast(int, i)] for i in cluster.members
            }
            print(
                str(i + 1),
                None if cluster.radius is None else round(cluster.radius, 3),
                (
                    None
                    if cluster.observed_pvr is None
                    else round(cluster.observed_pvr, 2)
                ),
                cluster.kind_str,
                sum(sequence_lens[i] for i in cluster.members),
                len(cluster.members),
                sequence_names[cluster.medoid],
                file=file,
                sep="\t",
            )

    elapsed = round(time.time() - begintime, 2)

    write_clusters_and_bins(
        fasta_output,
        bin_prefix,
        binsplitter,
        base_clusters_name,
        cluster_dict,
        sequence_names,
        cast(Sequence[int], sequence_lens),
    )
    logger.info(f"\tClustered contigs in {elapsed} seconds.\n")

def cluster_and_write_clusters(
    vamb_options: GeneralOptions,
    cluster_options: ClusterOptions,
    binsplitter: vamb.vambtools.BinSplitter,
    base_clusters_name: str,  # e.g. /foo/bar/vae -> /foo/bar/vae_unsplit.tsv
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
    logger.info(f"\tBinsplitter: {binsplitter.splitter}")

    cluster_generator = vamb.cluster.ClusterGenerator( #vamb.cluster_contr_vamb.ClusterGenerator(
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

    write_clusters(
        binsplitter,
        base_clusters_name,
        cluster_dict,
    )
    logger.info(f"\tClustered contigs in {elapsed} seconds.\n")

    return cluster_dict


def write_clusters_and_bins(
    fasta_output: Optional[FastaOutput],
    # If `x` and not None, clusters will be renamed `x` + old_name.
    # This is necessary since for the AAE, we may need to write bins
    # from three latent spaces into the same directory, and the names
    # must not clash.
    bin_prefix: Optional[str],
    binsplitter: vamb.vambtools.BinSplitter,
    base_clusters_name: str,  # e.g. /foo/bar/vae -> /foo/bar/vae_unsplit.tsv
    clusters: dict[str, set[str]],
    sequence_names: Sequence[str],
    sequence_lens: Sequence[int],
):
    # Write unsplit clusters to file
    begintime = time.time()
    unsplit_path = Path(base_clusters_name + "_unsplit.tsv")
    with open(unsplit_path, "w") as file:
        #file.write("clustername\tcontigname\n")
        (n_unsplit_clusters, n_contigs) = vamb.vambtools.write_clusters(
            file, clusters.items()
        )
    #Open unsplit clusters and split them
    if binsplitter.splitter is not None:
        split_path = Path(base_clusters_name + "_split.tsv")
        clusters = dict(binsplitter.binsplit(clusters.items()))
        # Add prefix before writing the clusters to file
        clusters = add_bin_prefix(clusters, bin_prefix)
        with open(split_path, "w") as file:
            (n_split_clusters, _) = vamb.vambtools.write_clusters(
                file, clusters.items()
            )
        msg = f"\tClustered {n_contigs} contigs in {n_split_clusters} split bins ({n_unsplit_clusters} clusters)"
    else:
        msg = f"\tClustered {n_contigs} contigs in {n_unsplit_clusters} unsplit bins"
        clusters = add_bin_prefix(clusters, bin_prefix)

    logger.info(msg)
    elapsed = round(time.time() - begintime, 2)
    logger.info(f"\tWrote cluster file(s) in {elapsed} seconds.")

    # Write bins, if necessary
    if fasta_output is not None:
        starttime = time.time()
        filtered_clusters: dict[str, set[str]] = dict()
        assert len(sequence_lens) == len(sequence_names)
        sizeof = dict(zip(sequence_names, sequence_lens))
        for binname, contigs in clusters.items():
            if sum(sizeof[c] for c in contigs) >= fasta_output.min_fasta_size:
                filtered_clusters[binname] = contigs

        with vamb.vambtools.Reader(fasta_output.existing_fasta_path.path) as file:
            vamb.vambtools.write_bins(
                fasta_output.bins_dir_to_populate,
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

def add_bin_prefix(
    clusters: dict[str, set[str]], prefix: Optional[str]
) -> dict[str, set[str]]:
    if prefix is None:
        return clusters
    else:
        return {prefix + b: c for (b, c) in clusters.items()}


def write_clusters(
    binsplitter: vamb.vambtools.BinSplitter,
    base_clusters_name: str,  # e.g. /foo/bar/vae -> /foo/bar/vae_unsplit.tsv
    clusters: dict[str, set[str]],
):
    # Write unsplit clusters to file
    begintime = time.time()
    unsplit_path = Path(base_clusters_name + "_unsplit.tsv")
    with open(unsplit_path, "w") as file:
        #file.write("clustername\tcontigname\n")
        (n_unsplit_clusters, n_contigs) = vamb.vambtools.write_clusters(
            file, clusters.items()
        )
    #Open unsplit clusters and split them
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


def find_neighbours(
    cosine_distances,
    idxs_without_neighs,
    radius=0.02,
    sort=True,
):
    logger.info("Neighbour if within %s cosine distance" % str(radius))

    # Set the diagonal to a value greater than the threshold to avoid self-selection
    np.fill_diagonal(cosine_distances, 1)
    logger.info("diagonal_filled")

    # Use NumPy operations for faster neighbor finding
    mask = cosine_distances < radius
    
    logger.info("mask generated with shape %i, %i"%(mask.shape[0],mask.shape[1]))

    idxs_with_neighs = np.any(mask, axis=1)
    
    logger.info("idxs_with_neighs generated")
    
    neighs = [
        []
        if i in set(idxs_without_neighs)
        else list(np.where(mask[i])[0])
        if idxs_with_neighs[i]
        else []
        for i in range(mask.shape[0])
    ]
    logger.info("neighs found")
    # new part
    if sort == True:
        neighs_sorted = []
        for i, nn_idxs in enumerate(neighs):
            if len(nn_idxs) == 0:
                neighs_sorted.append([])
                continue

            distances_neighs_i = cosine_distances[i, mask[i, :]]
            sorted_distances_i = sorted(
                range(len(distances_neighs_i)), key=lambda i: distances_neighs_i[i]
            )
            # nn_idxs_sorted = [nn_idxs[i] for i in sorted_distances_i]
            nn_idxs_sorted = [
                nn_idxs[i]
                for i in sorted_distances_i
                if nn_idxs[i] not in idxs_without_neighs
            ]
            neighs_sorted.append(nn_idxs_sorted)

        return neighs_sorted, np.where(idxs_with_neighs)[0]
    else:
        return neighs, idxs_with_neighs


def find_looners_and_expand_neighs(
    cosine_distances,
    contig_idxs_with_neighs,
    neighs_g,
    contignames,
    radius_looner=0.05,
    radius_neigh = 0.01,
    
):
    logger.info("Neighbour if within %.3f cosine distance" % radius_neigh)
    logger.info("Looner if no contigs within %.3f cosine distance" % radius_looner)

    c2idx = { c:i for i,c in enumerate(contignames)}
    # Set the diagonal to a value greater than the threshold to avoid self-selection
    np.fill_diagonal(cosine_distances, 1)
    mask_looners  = np.zeros(len(cosine_distances),dtype=bool)
    looner_contigs = set()
    for c_idx in range(len(cosine_distances)):
        if np.min(cosine_distances[c_idx]) > radius_looner and c_idx not in contig_idxs_with_neighs:
            looner_contigs.add(contignames[c_idx])
            mask_looners[c_idx]=True
            
            continue 
        idxs_neighs = np.where(cosine_distances[c_idx] < 0.01)[0]
        for idx_neigh in idxs_neighs:
            neighs_g.add_edge(contignames[c_idx],contignames[idx_neigh])
            
        
    hoods_with_looners_cs_d = {
        "hood_%i" % i: cs
        for i, cs in enumerate(nx.connected_components(neighs_g))
    }
    for i,c in enumerate(looner_contigs):
        hoods_with_looners_cs_d["looner_%i"%(i)]=set([c])
    
    mask_neighs  = np.zeros(len(cosine_distances),dtype=bool)
    mask_neighs[[c2idx[c]  for c in neighs_g ]]=True
    
    


    return looner_contigs,neighs_g,hoods_with_looners_cs_d,mask_looners, mask_neighs


def find_neighbourhoods(contignames,neighs):
    
    neighbourhoods_g = nx.Graph()
    for i,neigh_idxs in enumerate(neighs): 
        c = contignames[i]
        for neigh_idx in neigh_idxs:
            c_neigh = contignames[neigh_idx]
            if  c_neigh == c:
                continue 
            #neighbourhoods_g.add_edge(c_neigh,c)
            neighbourhoods_g.add_edge(c,c_neigh)
    
    neighbourhoods_cs_d = { i:cc for i,cc in enumerate(nx.connected_components(neighbourhoods_g))}
    
    return neighbourhoods_cs_d



def split_neighbourhoods_by_sample(neighbourhoods):
    nbhds_split = dict()
    for i,(nbhd_id, nbhd_cs) in enumerate(neighbourhoods.items()):
        samples_in_nbhd = set([ c.split("C")[0] for c in nbhd_cs ])
        
        nbhd_S_d = { S:set() for S in samples_in_nbhd}

        for c in nbhd_cs:
            nbhd_S_d[c.split("C")[0]].add(c)

        # remove 1 contig neighbourhoods
        nbhd_S_clean_d = { S:cs for S,cs in nbhd_S_d.items() if len(cs) > 1 }
        


        for S,cs in nbhd_S_clean_d.items():
            nbhds_split["%i_%s"%(nbhd_id,S)] = cs
        
    return nbhds_split



def load_composition_and_abundance_and_neighs(
    vamb_options: GeneralOptions,
    comp_options: CompositionOptions,
    abundance_options: AbundanceOptions,
    neighs_options: NeighsOptions,
    binsplitter: vamb.vambtools.BinSplitter,
) -> Tuple[vamb.parsecontigs.Composition, vamb.parsebam.Abundance, np.ndarray]:
    logger.info(
        "Starting contrastive-Vamb version "
        + ".".join(map(str, vamb.__version_str__))
    )

    logger.info("Date and time is " + str(datetime.datetime.now()))
    begintime = time.time()

    composition = calc_tnf(comp_options, vamb_options.min_contig_length ,vamb_options.out_dir, binsplitter)
    abundance = calc_abundance(
        abundance_options,
        vamb_options.out_dir,
        vamb_options.refcheck,
        composition.metadata,
        vamb_options.n_threads,
    )

    time_generating_input = round(time.time() - begintime, 2)
    logger.info(f"TNF and coabundances generated in {time_generating_input} seconds.")    
    
    logger.info(
        f"Loading contig neighs from {neighs_options.neighs_path}."
    )

    neighs = np.load(
        neighs_options.neighs_path,allow_pickle=True
    )["arr_0"]

    neighs_mask = np.array([ len(lst) > 0 for lst in neighs])
    
    contigs_with_neighs_n = np.sum(neighs_mask)
    total_neighs = np.sum([len(ns) for ns in neighs])
    mean_neighs_per_contig = np.mean([len(ns) for ns in neighs])
    std_neighs_per_contig = np.std([len(ns) for ns in neighs])
    logger.info(f"Contigs with neighs {contigs_with_neighs_n}.")
    logger.info("Mean(std) neighbours per contig: %.2f (%.2f)."%(mean_neighs_per_contig,std_neighs_per_contig))
    logger.info(f"Total neigh connections {total_neighs}.")
        
    return (
        composition,
        abundance,
        neighs,
        neighs_mask
    )

def run_bin_default(opt: BinDefaultOptions):

    composition, abundance = load_composition_and_abundance(
        vamb_options=opt.common.general,
        comp_options=opt.common.comp,
        abundance_options=opt.common.abundance,
        binsplitter=opt.common.output.binsplitter,
    )
    data_loader = vamb.encode.make_dataloader(
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        256,  # dummy value - we change this before using the actual loader
        destroy=True,
        cuda=opt.common.general.cuda,
    )
    latent = trainvae(
        vae_options=opt.vae,
        vamb_options=opt.common.general,
        data_loader=data_loader,
    )
    # Free up memory
    comp_metadata = composition.metadata
    del composition, abundance
    assert comp_metadata.nseqs == len(latent)

    cluster_and_write_files(
        opt.common.clustering,
        opt.common.output.binsplitter,
        latent,
        cast(Sequence[str], comp_metadata.identifiers),
        comp_metadata.lengths,
        opt.common.general.seed,
        opt.common.general.cuda,
        str(opt.common.general.out_dir.joinpath("vae_clusters")),
        FastaOutput.try_from_common(opt.common),
        None,
    )
    del latent


def run_bin_aae(opt: BinAvambOptions):
    composition, abundance = load_composition_and_abundance(
        vamb_options=opt.common.general,
        comp_options=opt.common.comp,
        abundance_options=opt.common.abundance,
        binsplitter=opt.common.output.binsplitter,
    )
    data_loader = vamb.encode.make_dataloader(
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        256,  # dummy value - we change this before using the actual loader
        destroy=True,
        cuda=opt.common.general.cuda,
    )
    latent_z, clusters_y_dict = trainaae(
        data_loader=data_loader,
        aae_options=opt.aae,
        vamb_options=opt.common.general,
        alpha=opt.vae.alpha,
        contignames=cast(Sequence[str], composition.metadata.identifiers),
    )
    # Free up memory
    comp_metadata = composition.metadata
    del composition, abundance
    assert comp_metadata.nseqs == len(latent_z)
    # Cluster and output the Z clusters
    # This function calls write_clusters_and_bins,
    # but also does the actual clustering and writes cluster metadata.
    # This does not apply to the aae_y clusters, since their cluster label
    # can be extracted directly from the latent space without clustering,
    # and hence below, `write_clusters_and_bins` is called directly instead.
    cluster_and_write_files(
        opt.common.clustering,
        opt.common.output.binsplitter,
        latent_z,
        cast(Sequence[str], comp_metadata.identifiers),
        comp_metadata.lengths,
        opt.common.general.seed,
        opt.common.general.cuda,
        str(opt.common.general.out_dir.joinpath("aae_z_clusters")),
        FastaOutput.try_from_common(opt.common),
        "z_",
    )
    del latent_z

    # We enforce this in the VAEAAEOptions constructor, see comment there
    # Cluster and output the Y clusters
    assert opt.common.clustering.max_clusters is None
    write_clusters_and_bins(
        FastaOutput.try_from_common(opt.common),
        "y_",
        binsplitter=opt.common.output.binsplitter,
        base_clusters_name=str(opt.common.general.out_dir.joinpath("aae_y_clusters")),
        clusters=clusters_y_dict,
        sequence_names=cast(Sequence[str], comp_metadata.identifiers),
        sequence_lens=cast(Sequence[int], comp_metadata.lengths),
    )


def predict_taxonomy(
    comp_metadata: vamb.parsecontigs.CompositionMetaData,
    abundance: np.ndarray,
    tnfs: np.ndarray,
    lengths: np.ndarray,
    out_dir: Path,
    taxonomy_options: TaxometerOptions,
    cuda: bool,
) -> vamb.taxonomy.PredictedTaxonomy:
    begintime = time.time()
    logger.info("Predicting taxonomy with Taxometer")

    taxonomies = vamb.taxonomy.Taxonomy.from_file(
        taxonomy_options.taxonomy.path, comp_metadata, False
    )
    nodes, ind_nodes, table_parent = vamb.taxvamb_encode.make_graph(
        taxonomies.contig_taxonomies
    )
    logger.info(f"\t{len(nodes)} nodes in the graph")
    classes_order: list[str] = []
    for i in taxonomies.contig_taxonomies:
        if i is None or len(i.ranks) == 0:
            classes_order.append("root")
        else:
            classes_order.append(i.ranks[-1])
    targets = np.array([ind_nodes[i] for i in classes_order])

    model = vamb.taxvamb_encode.VAMB2Label(
        abundance.shape[1],
        len(nodes),
        nodes,
        table_parent,
        nhiddens=[512, 512, 512, 512],
        hier_loss=taxonomy_options.ploss,
        cuda=cuda,
    )

    dataloader_vamb = vamb.encode.make_dataloader(
        abundance,
        tnfs,
        lengths,
        batchsize=taxonomy_options.basic.starting_batch_size,
    )
    dataloader_joint = vamb.taxvamb_encode.make_dataloader_concat_hloss(
        abundance,
        tnfs,
        lengths,
        targets,
        len(nodes),
        table_parent,
        batchsize=taxonomy_options.basic.starting_batch_size,
    )

    logger.info("\tCreated dataloader")

    predictortime = time.time()
    logger.info("Starting training the taxonomy predictor")
    logger.info(f"Using threshold {taxonomy_options.softmax_threshold}")

    model_path = out_dir.joinpath("predictor_model.pt")
    with open(model_path, "wb") as modelfile:
        model.trainmodel(
            dataloader_joint,
            nepochs=taxonomy_options.basic.num_epochs,
            modelfile=modelfile,
            batchsteps=taxonomy_options.basic.batch_steps,
        )
    logger.info(
        f"Finished training the taxonomy predictor in {round(time.time() - predictortime, 2)} seconds."
    )

    logger.info("Writing the taxonomy predictions")
    predicted_path = out_dir.joinpath("results_taxometer.tsv")
    nodes_ar = np.array(nodes)

    logger.info(f"Using threshold {taxonomy_options.softmax_threshold}")
    contig_taxonomies: list[vamb.taxonomy.PredictedContigTaxonomy] = []
    for predicted_vector, predicted_labels in model.predict(dataloader_vamb):
        for i in range(predicted_vector.shape[0]):
            label = predicted_labels[i]
            while table_parent[label] != -1:
                label = table_parent[label]
            threshold_mask: Sequence[bool] = (
                predicted_vector[i] > taxonomy_options.softmax_threshold
            )
            ranks = list(nodes_ar[threshold_mask][1:])
            probs = predicted_vector[i][threshold_mask][1:]
            tax = vamb.taxonomy.PredictedContigTaxonomy(
                vamb.taxonomy.ContigTaxonomy(ranks), probs
            )
            contig_taxonomies.append(tax)

    taxonomy = vamb.taxonomy.PredictedTaxonomy(contig_taxonomies, comp_metadata, False)

    with open(predicted_path, "w") as file:
        taxonomy.write_as_tsv(file, comp_metadata)

    logger.info(
        f"Completed taxonomy predictions in {round(time.time() - begintime, 2)} seconds."
    )
    return taxonomy


def run_taxonomy_predictor(opt: TaxometerOptions):
    composition, abundance = load_composition_and_abundance(
        opt.general,
        opt.composition,
        opt.abundance,
        vamb.vambtools.BinSplitter.inert_splitter(),
    )
    abundance, tnfs, lengths, _ = (
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        composition.metadata.identifiers,
    )
    predict_taxonomy(
        comp_metadata=composition.metadata,
        abundance=abundance,
        tnfs=tnfs,
        lengths=lengths,
        out_dir=opt.general.out_dir,
        taxonomy_options=opt,
        cuda=opt.general.cuda,
    )


def run_vaevae(opt: BinTaxVambOptions):
    vae_options = opt.vae
    composition, abundance = load_composition_and_abundance(
        vamb_options=opt.common.general,
        comp_options=opt.common.comp,
        abundance_options=opt.common.abundance,
        binsplitter=opt.common.output.binsplitter,
    )
    abundance, tnfs, lengths, contignames = (
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        composition.metadata.identifiers,
    )
    if isinstance(opt.taxonomy, TaxometerOptions):
        predicted_contig_taxonomies = predict_taxonomy(
            comp_metadata=composition.metadata,
            abundance=abundance,
            tnfs=tnfs,
            lengths=lengths,
            out_dir=opt.common.general.out_dir,
            taxonomy_options=opt.taxonomy,
            cuda=opt.common.general.cuda,
        )
        contig_taxonomies = predicted_contig_taxonomies.to_taxonomy()
    elif isinstance(opt.taxonomy, RefinedTaxonomy):
        logger.info(f'Loading already-refined taxonomy from file "{opt.taxonomy.path}"')
        contig_taxonomies = vamb.taxonomy.Taxonomy.from_refined_file(
            opt.taxonomy.path, composition.metadata, False
        )
    else:
        logger.info(f'Loading unrefined taxonomy from file "{opt.taxonomy.path}"')
        contig_taxonomies = vamb.taxonomy.Taxonomy.from_file(
            opt.taxonomy.path, composition.metadata, False
        )

    nodes, ind_nodes, table_parent = vamb.taxvamb_encode.make_graph(
        contig_taxonomies.contig_taxonomies
    )
    classes_order: list[str] = []
    for i in contig_taxonomies.contig_taxonomies:
        if i is None or len(i.ranks) == 0:
            classes_order.append("root")
        else:
            classes_order.append(i.ranks[-1])
    targets = np.array([ind_nodes[i] for i in classes_order])

    vae = vamb.taxvamb_encode.VAEVAEHLoss(
        abundance.shape[1],
        len(nodes),
        nodes,
        table_parent,
        nhiddens=vae_options.nhiddens,
        nlatent=vae_options.nlatent,
        alpha=vae_options.alpha,
        beta=vae_options.beta,
        dropout=vae_options.dropout,
        cuda=opt.common.general.cuda,
    )

    dataloader_vamb = vamb.encode.make_dataloader(
        abundance,
        tnfs,
        lengths,
        batchsize=vae_options.basic_options.starting_batch_size,
        cuda=opt.common.general.cuda,
    )
    dataloader_joint = vamb.taxvamb_encode.make_dataloader_concat_hloss(
        abundance,
        tnfs,
        lengths,
        targets,
        len(nodes),
        table_parent,
        batchsize=vae_options.basic_options.starting_batch_size,
        cuda=opt.common.general.cuda,
    )
    dataloader_labels = vamb.taxvamb_encode.make_dataloader_labels_hloss(
        abundance,
        tnfs,
        lengths,
        targets,
        len(nodes),
        table_parent,
        batchsize=vae_options.basic_options.starting_batch_size,
        cuda=opt.common.general.cuda,
    )

    shapes = (abundance.shape[1], 103, 1, len(nodes))
    dataloader = vamb.taxvamb_encode.make_dataloader_semisupervised_hloss(
        dataloader_joint,
        dataloader_vamb,
        dataloader_labels,
        len(nodes),
        table_parent,
        shapes,
        opt.common.general.seed,
        batchsize=vae_options.basic_options.starting_batch_size,
        cuda=opt.common.general.cuda,
    )
    model_path = opt.common.general.out_dir.joinpath("vaevae_model.pt")
    with open(model_path, "wb") as modelfile:
        vae.trainmodel(
            dataloader,
            nepochs=vae_options.basic_options.num_epochs,
            modelfile=modelfile,
            batchsteps=vae_options.basic_options.batch_steps,
        )

    latent_both = vae.VAEJoint.encode(dataloader_joint)
    logger.info(f"{latent_both.shape} embedding shape")

    latent_path = opt.common.general.out_dir.joinpath("vaevae_latent.npz")
    vamb.vambtools.write_npz(latent_path, latent_both)

    # Cluster, save tsv file
    cluster_and_write_files(
        opt.common.clustering,
        opt.common.output.binsplitter,
        latent_both,
        cast(Sequence[str], contignames),
        lengths,
        opt.common.general.seed,
        opt.common.general.cuda,
        str(opt.common.general.out_dir.joinpath("vaevae_clusters")),
        FastaOutput.try_from_common(opt.common),
        None,
    )

def run_reclustering(opt: ReclusteringOptions):
    composition = calc_tnf(
        opt.composition,
        opt.general.min_contig_length,
        opt.general.out_dir,
        opt.output.binsplitter,
    )    
    markers = load_markers(
        opt.markers, composition.metadata, opt.general.out_dir, opt.general.n_threads
    )
    latent = vamb.vambtools.read_npz(opt.latent_path)
    alg = opt.algorithm

    if isinstance(alg, DBScanOptions):
        # If we should refine or not.
        if isinstance(alg.taxonomy, TaxometerOptions):
            taxopt = alg.taxonomy
            abundance = calc_abundance(
                taxopt.abundance,
                taxopt.general.out_dir,
                taxopt.general.refcheck,
                composition.metadata,
                taxopt.general.n_threads,
            )
            predicted_tax = predict_taxonomy(
                composition.metadata,
                abundance.matrix,
                composition.matrix,
                composition.metadata.lengths,
                taxopt.general.out_dir,
                taxopt,
                taxopt.general.cuda,
            )
            taxonomy = predicted_tax.to_taxonomy()
        else:
            if isinstance(alg.taxonomy, UnrefinedTaxonomy):
                logger.info(
                    f'Loading unrefined taxonomy from file "{alg.taxonomy.path}"'
                )
                taxonomy = vamb.taxonomy.Taxonomy.from_file(
                    alg.taxonomy.path, composition.metadata, True
                )
            else:
                logger.info(f'Loading refined taxonomy from file "{alg.taxonomy.path}"')
                taxonomy = vamb.taxonomy.Taxonomy.from_refined_file(
                    alg.taxonomy.path, composition.metadata, True
                )

        instantiated_alg = vamb.reclustering.DBScanAlgorithm(
            composition.metadata, taxonomy, opt.general.n_threads
        )
        logger.info("Reclustering")
        logger.info("\tAlgorithm: DBSCAN")
        reclustered_contigs = vamb.reclustering.recluster_bins(
            markers, latent, instantiated_alg
        )
    else:
        with open(alg.clusters) as file:
            clusters = vamb.vambtools.read_clusters(file)
        contig_to_id: dict[str, int] = {
            c: i for (i, c) in enumerate(composition.metadata.identifiers)
        }
        clusters_as_ids: list[set[vamb.reclustering.ContigId]] = []
        for cluster in clusters.values():
            s = set()
            for contig in cluster:
                i = contig_to_id.get(contig, None)
                if i is None:
                    raise ValueError(
                        f'Contig "{contig}" found in the provided clusters file is not found in the '
                        "provided composition."
                    )
                s.add(vamb.reclustering.ContigId(i))
            clusters_as_ids.append(s)
        instantiated_alg = vamb.reclustering.KmeansAlgorithm(
            clusters_as_ids,
            abs(opt.general.seed) % 4294967295,  # Note: sklearn seeds must be uint32
            composition.metadata.lengths,
        )
        logger.info("Reclustering")
        logger.info("\tAlgorithm: KMeans")
        reclustered_contigs = vamb.reclustering.recluster_bins(
            markers, latent, instantiated_alg
        )

    logger.info("\tReclustering complete")
    identifiers = composition.metadata.identifiers
    clusters_dict: dict[str, set[str]] = dict()
    for i, cluster in enumerate(reclustered_contigs):
        clusters_dict[str(i)] = {identifiers[c] for c in cluster}

    if opt.output.min_fasta_output_size is None:
        fasta_output = None
    else:
        assert isinstance(opt.composition.path, FASTAPath)
        fasta_output = FastaOutput(
            opt.composition.path,
            opt.general.out_dir.joinpath("bins"),
            opt.output.min_fasta_output_size,
        )

    write_clusters_and_bins(
        fasta_output,
        None,
        opt.output.binsplitter,
        str(opt.general.out_dir.joinpath("clusters_reclustered")),
        clusters_dict,
        cast(Sequence[str], composition.metadata.identifiers),
        cast(Sequence[int], composition.metadata.lengths),
    )


def run_bin_contr_vamb(
    opt: BinContrVambOptions,
    # vamb_options: VambOptions,
    # comp_options: CompositionOptions,
    # abundance_options: AbundanceOptions,
    # neighs_options: NeighsOptions,
    # encoder_options: Encodern2vOptions,
    # training_options: TrainingOptions,
    # cluster_options: ClusterOptions,
):
    # # I HAVE TO MAKE CHANGES ACCORDING TO THE CHANGES IN THE ENCODE_N2V_ASIMETRIC
    # # I ALSO HAVE TO IMPLEMENT A FUNCTION THAT GENERATES AN OBJECT INDICATING THE NEIGHBOURS OF EACH CONTIG
    # # BAED ON EMBEDDING SPACE AND radius DISTANCE
    # vae_options = encoder_options.vae_options
    # vae_training_options = training_options.vae_options

    (
        composition,
        abundance,
        neighs_object,
        neighs_mask,
    ) = load_composition_and_abundance_and_neighs(
        vamb_options=opt.common.general,
        comp_options=opt.common.comp,
        abundance_options=opt.common.abundance,
        neighs_options=opt.neighs,
        binsplitter=opt.common.output.binsplitter,
    )

    # Write contignames and contiglengths needed for dereplication purposes
    np.savetxt(
        opt.common.general.out_dir.joinpath("contignames"),
        composition.metadata.identifiers,
        fmt="%s",
    )
    np.savez(opt.common.general.out_dir.joinpath("lengths.npz"), composition.metadata.lengths)


    
    logger.info(
        "Creating dataloader"
    )
    #logger.info("LONG INPUT ADDED WITH ABUNDANCES NO PROBABILITIES")

    (
        data_loader,
        neighs_object_after_dataloader,
    ) = vamb.encode_n2v_asimetric.make_dataloader_n2v(
        abundance.matrix.astype(np.float32),
        composition.matrix.astype(np.float32),
        neighs_mask.astype(np.float32),
        neighs_object,
        composition.metadata.lengths,
        256,  # dummy value - we change this before using the actual loader
        destroy=True,
        cuda=opt.common.general.cuda,
    )
    

    latent = None
    # Train, save model
    latent = train_contr_vae(
        vae_options=opt.vae,
        vamb_options=opt.common.general,
        # training_options=vae_training_options,
        # vamb_options=opt.common.general,
        neighs_options=opt.neighs,
        # #lrate=training_options.lrate,
        # alpha=encoder_options.alpha,
        data_loader=data_loader,
        neighs_object=neighs_object_after_dataloader,
    )

    # Free up memory
    comp_metadata = composition.metadata
    del composition, abundance

    assert latent is not None
    assert comp_metadata.nseqs == len(latent)
    
    ## First cluster based on hoods, cluster the rest with vamb default clustering
    # clean neighbours using latent space information
    neighs_object = clean_neighs_with_latents(
        latent,
        neighs_object_after_dataloader,
        opt.neighs.max_neighs_r
        )
    
    logger.info("Neighbourhoods before aggregation: %i"%len(find_neighbourhoods(comp_metadata.identifiers,neighs_object).keys()))

    ## optimized clustering
    logger.info("Clustering based on neighbourhoods communities")
    
    latent_communities_cs_d,looner_mask,contigs_in_latent_communities_mask = cluster_hoods_based(
        latent,
        neighs_object,
        comp_metadata.identifiers,
        opt.common.output.binsplitter.splitter,
        opt.neighs.radius_clustering
        )
    logger.info("Clustering baased on the neighbourhoods communities finished")

    
    if not os.path.isdir(os.path.join(opt.common.general.out_dir,"tmp")):
        os.mkdir(os.path.join(opt.common.general.out_dir,"tmp"))
    
    write_clusters(
        opt.common.output.binsplitter,
        str(opt.common.general.out_dir.joinpath("tmp/vae_clusters_community_based")),
        latent_communities_cs_d
    )

    logger.info(
        "\tClustering remaining contigs not within a radius distance from within_radius_merged_hoods_0.01 members"
    )

    cls_outside_coms_density_cs_d = cluster_and_write_clusters(
        opt.common.general,
        opt.common.clustering,
        opt.common.output.binsplitter,
        str(opt.common.general.out_dir.joinpath("tmp/vae_clusters_outside_communities_density")),
        latent.copy()[
            ~(looner_mask | contigs_in_latent_communities_mask)
        ],
        comp_metadata.identifiers[
            ~(looner_mask | contigs_in_latent_communities_mask)
        ],  # type:ignore
        comp_metadata.lengths[
            ~(looner_mask | contigs_in_latent_communities_mask)
        ],  # type:ignore
    )
    
    # merge the within and outside clusters and ensure tehre are not repeats and missing contigs
    merged_cl_cs_d = latent_communities_cs_d.copy()


    for cl,cs in cls_outside_coms_density_cs_d.items():
        merged_cl_cs_d[cl] = cs

    assert len(np.sum(set([c for cs in merged_cl_cs_d.values() for c in cs]))) == len(
        comp_metadata.identifiers
    )
    # save clusters within margins and outside margins

    write_clusters(
        opt.common.output.binsplitter,
        str(opt.common.general.out_dir.joinpath("vae_clusters_community_based_complete")),
        merged_cl_cs_d,
    )


    ## Now generate a new set of clusters based on vamb default clustering
    # Cluster, save tsv file
    logger.info("Clustering latents with vamb density algorithm")
    cluster_and_write_clusters(
        opt.common.general,
        opt.common.clustering,
        opt.common.output.binsplitter,
        str(opt.common.general.out_dir.joinpath("vae_clusters_density")),
        latent,
        comp_metadata.identifiers,  # type:ignore
        comp_metadata.lengths,  # type:ignore
    )
    
    del latent



def clean_neighs_with_latents(
    latent,
    neighs,
    max_radius=0.2
):
    """
    Given the contig latents, and the neighs, break neighbour connections if neighbours are furhter than a given cosine distance
    """
    neighs_clean=[]
    latent_nz = vamb.cluster._normalize(latent)
    total_neighs_removed=0
    total_neighs_original=0
    logger.info("Breaking neighbour connections if further than %s normalized cosine distance"%str(max_radius))
    for c_idx,neigh_idxs in enumerate(neighs):
        if len(neigh_idxs)==0:
            neighs_clean.append([])
            continue
        distances = vamb.cluster._calc_distances(latent_nz,c_idx)
        mask_neighs = torch.zeros(len(latent_nz),dtype=bool)
        mask_neighs[neigh_idxs] = True
        #print("Neighs before cleaning",neigh_idxs)
        within_max_radius = (distances <= max_radius) & mask_neighs # keep contigs that are neighbours and are within the max radius on latent space
        neighs_clean_i = list(np.arange(len(neighs))[within_max_radius])
        #logger("Neighs after cleaning",neighs_clean_i))
        neighs_clean.append(neighs_clean_i)
        total_neighs_removed += len(neigh_idxs)-len(neighs_clean_i)
        total_neighs_original+= len(neigh_idxs)
    logger.info("%i neighs before cleaining with latents"%total_neighs_original)
    logger.info("%i neighs after cleaining with latents"%(total_neighs_original-total_neighs_removed))
    return neighs_clean                
        
        
def cluster_hoods_based(
    latent,
    neighs,
    contignames,
    binsplit_separator="C",
    radius_clustering=0.01,
    radius_looner=0.5,
    
):
    """
    Given the latent space after training, and the neighs extracted from the n2v alignmnet-assembly graph,
    expand the communities based on the latent space.
    """
    radius = radius_clustering/2
    looner_radius=radius_looner/2
    
    looner_mask = np.zeros(len(latent),dtype=bool)
    latent_communities_g = nx.Graph()
    latent_nz = vamb.cluster._normalize(latent)

    ## For each contig latent, find the contigs that are within a distance, and link them
    for i in range(len(latent)):
        distances = vamb.cluster._calc_distances(latent_nz,i)
        mask_ = torch.ones(len(latent), dtype=torch.bool)
        mask_[i] = False
        within_radius = (distances <= radius) & mask_
        neighs_graph = neighs[i]
        if torch.sum(within_radius) == len(neighs_graph) == 0: # potentially a loner
            within_looner_radius = (distances <= looner_radius) & mask_        
            if torch.sum(within_looner_radius) == 0:
                looner_mask[i]=True
                latent_communities_g.add_node(contignames[i])
        else:
            for lat_neigh in np.where(within_radius)[0]:
                latent_communities_g.add_edge(contignames[i],contignames[lat_neigh])
            
            for neigh_idx in neighs_graph:
                latent_communities_g.add_edge(contignames[i],contignames[neigh_idx])
            #contigs_in_latent_communities_mask[i]=True
        
    
    
    # latent_communities_cs_d = {
    #     "nneighs_%i" % i if len(cs) > 1 else "looner_%i"%i: cs
    #     for i, cs in enumerate(nx.connected_components(latent_communities_g))
        
    # }
    ## For each cluster only composed by contigs from different samples, i.e. there is only one contig from each sample present in the cluster,
    ## remove those clusters since no graph information is used.
    latent_communities_cs_d_NOT_CLEAN = {
        "nneighs_%i" % i if len(cs) > 1 else "looner_%i"%i: cs
        for i, cs in enumerate(nx.connected_components(latent_communities_g))
    }
    logger.info(
        "Neighbourhoods before removing only intersample hoods: %i" % (len([cl for cl in latent_communities_cs_d_NOT_CLEAN.keys() if cl.startswith("nneighs_")]))
    )


    latent_communities_cs_d = {}
    neighs_to_split = []
    for i,cs in enumerate(nx.connected_components(latent_communities_g)):
        if len(cs) == 1:
            latent_communities_cs_d["looner_%i"%i]=cs
            continue
        samples_hood = set( [c.split(binsplit_separator)[0] for c in cs])
        if len(samples_hood) == len(cs):
            neighs_to_split.append(cs)
            continue
        latent_communities_cs_d["nneighs_%i"%i]=cs
    
    edges_to_remove=[]
    
    for cs in neighs_to_split:
        c_seen = set()
        for c_i in cs:
            c_seen.add(c_i)
            for c_j in cs-c_seen:
                if (c_i,c_j) in latent_communities_g.edges():
                    edges_to_remove.append((c_i,c_j))
    
    nodes_to_remove=set()
    latent_communities_g.remove_edges_from(edges_to_remove)
    for c in latent_communities_g.nodes():
        if len(latent_communities_g.edges(c)) == 0:
            nodes_to_remove.add(c)
    
    
    latent_communities_g.remove_nodes_from(nodes_to_remove)

    logger.info("%i (%i) edges (nodes) have been removed since they were part of clusters containing only intersample contigs"%(len(edges_to_remove),len(nodes_to_remove)))
    


    contigs_in_withinradiusclusters_n = np.sum([len(cs) for cs in latent_communities_cs_d.values() if len(cs) > 1 ])
    logger.info(
        "Contigs in within neighbourhoods radius clusters: %i"
        % contigs_in_withinradiusclusters_n
    )
    cluster_with_more_contigs = max(latent_communities_cs_d, key=lambda k : len(latent_communities_cs_d[k]) )
    
    n_contigs_max_cluster = len(latent_communities_cs_d[cluster_with_more_contigs])
    
#    logger.info("Removing within neighbourhoods radius cluster with more aggregated contigs %i"%n_contigs_max_cluster)
#    latent_communities_g.remove_nodes_from([c  for c in latent_communities_cs_d[cluster_with_more_contigs]])
#    del latent_communities_cs_d[cluster_with_more_contigs]
    contigs_in_latent_communities_mask= np.array([ 1 if c in latent_communities_g.nodes() else 0  for c in contignames ],dtype=bool)
    logger.info(
        "Looner contigs: %i"
        % np.sum(looner_mask)
    )

    logger.info(
        "Aggregated neighbourhoods: %i" % (len([cl for cl in latent_communities_cs_d.keys() if cl.startswith("nneighs_")]))
    )

    return latent_communities_cs_d,looner_mask,contigs_in_latent_communities_mask


    
def neighs_within_radius(
    neighs_object,
    contignames,
    latent_neighs,
):
    """
    There are two kind of neighbours, the neighs from the n2v embeddings space, and the ones from vamb's 
    latent space. Here we merge them.
    Input:
        neighs_object: list of lists of len == N_contigs where each element indicates the neighbouring contig indices in n2v embedding space
        contignames: contig names
        latent_neighs: list of lists of len == N_contigs where each element indicates the neighbouring contig indices in vamb's latent space
    """
    
    c2idx = {contigname: i for i, contigname in enumerate(contignames)}
    mask_contigs_withinR_neighbourhoods = np.zeros(len(contignames), dtype=bool)    
    mask_contigs_in_neighbourhoods =  np.zeros(len(contignames), dtype=bool)    
    ## First create the neighbourhoods objects only based upon 
    neighbourhoods_graph = nx.Graph()
    contigs_in_neighbourhoods = set()
    for c_i, neigh_idxs in enumerate(neighs_object):
        for n_i in neigh_idxs:
            neighbourhoods_graph.add_edge(contignames[c_i], contignames[n_i])
            contigs_in_neighbourhoods.add(contignames[n_i])
            contigs_in_neighbourhoods.add(contignames[c_i])
            mask_contigs_in_neighbourhoods[c_i] = True
            mask_contigs_in_neighbourhoods[n_i] = True
            
    logger.info("Contigs in neighbourhoods: %i" % len(contigs_in_neighbourhoods))

    n_neighbourhoods = np.sum(
        [1 for cs in nx.connected_components(neighbourhoods_graph) if len(cs) > 1]
    )
    logger.info("Initial neighbourhoods: %i" % n_neighbourhoods)

    # withinradiusclusters_g = nx.Graph()
    hood_cs_d = {
        "hood_%i" % i: cs
        for i, cs in enumerate(nx.connected_components(neighbourhoods_graph))
    }

    withinradiusclusters_g = neighbourhoods_graph
    logger.info("Graph object copied")
    singleton_contigs_aggregated = set()
    edges_to_add = []
    for cs in nx.connected_components(neighbourhoods_graph):
        # print(cs)
        assert len(cs) > 1, "neighbourhoods should be composed of at least 2 contigs"
        # define contig indices already part of the neighbourhood        
        idxs_cs_already_in_hood = [c2idx[c] for c in cs] 
        
        for idx_c_already_in_hood in idxs_cs_already_in_hood:
            idxs_latent_neighs_not_in_hood = set(latent_neighs[idx_c_already_in_hood]) - set(idxs_cs_already_in_hood)
            if len(idxs_latent_neighs_not_in_hood) > 0:
                for idx_latent_neighs_not_in_hood in idxs_latent_neighs_not_in_hood:
                    edges_to_add.append((contignames[idx_c_already_in_hood],contignames[idx_latent_neighs_not_in_hood]))
                    mask_contigs_withinR_neighbourhoods[idx_latent_neighs_not_in_hood] = True
                    if contignames[idx_latent_neighs_not_in_hood] not in contigs_in_neighbourhoods:
                        singleton_contigs_aggregated.add(contignames[idx_latent_neighs_not_in_hood])
    logger.info("merging finished")
    logger.info("%i (%i) edges (neighbourhoods) before merging"%(len(withinradiusclusters_g.edges()),len(list(nx.connected_components(withinradiusclusters_g)))))
    withinradiusclusters_g.add_edges_from(edges_to_add)
    logger.info("%i (%i) edges (neighbourhoods) after merging"%(len(withinradiusclusters_g.edges()),len(list(nx.connected_components(withinradiusclusters_g)))))
        # # define contig indices not part of the neighbourhood
        # idxs_cs_not_in_hood = set(np.arange(len(contignames))) - set(idxs_cs_already_in_hood) 
        
        # for idx_c_not_in_hood in idxs_cs_not_in_hood:
        #     # get the neighbouring contigs of the contigs not in hoods and that are in the hood (so the neighbours that are part of the original hood)
        #     latent_neighs_neighbourhood = set(latent_neighs[idx_c_not_in_hood]).intersection(
        #         set(idxs_cs_already_in_hood)
        #     )
        #     if len(latent_neighs_neighbourhood) > 0:
        #         for latent_neigh_idx in latent_neighs_neighbourhood:
        #             withinradiusclusters_g.add_edge(contignames[latent_neigh_idx], contignames[idx_c_not_in_hood])
        #             mask_contigs_withinR_neighbourhoods[idx_c_not_in_hood] = True
        #             if contignames[idx_c_not_in_hood] not in contigs_in_neighbourhoods:
        #                 singleton_contigs_aggregated.add(contignames[idx_c_not_in_hood])
                        
    withinradiusclusters_cs_d = {
        "nneighs_%i" % i: cs
        for i, cs in enumerate(nx.connected_components(withinradiusclusters_g))
    }
    logger.info(
        "Collected singleton contigs: %i"
        % len(singleton_contigs_aggregated)
    )
    
    contigs_in_withinradiusclusters_n = np.sum([len(cs) for cs in withinradiusclusters_cs_d.values() ])
    logger.info(
        "Contigs in within neighbourhoods radius clusters: %i"
        % contigs_in_withinradiusclusters_n
    )

    logger.info(
        "Aggregated neighbourhoods: %i" % (len(withinradiusclusters_cs_d.keys()))
    )
    return (
        hood_cs_d,
        withinradiusclusters_cs_d,
        withinradiusclusters_g,
        mask_contigs_in_neighbourhoods,
        mask_contigs_withinR_neighbourhoods,
    )


def add_help_arguments(parser: argparse.ArgumentParser):
    helpos = parser.add_argument_group(title="Help and version", description=None)
    helpos.add_argument("-h", "--help", help="Print help and exit", action="help")
    helpos.add_argument(
        "--version",
        action="version",
        version=f"Vamb {vamb.__version_str__}",
    )


def add_general_arguments(subparser: argparse.ArgumentParser):
    add_help_arguments(subparser)
    reqos = subparser.add_argument_group(title="Output", description=None)
    reqos.add_argument(
        "--outdir",
        metavar="",
        type=Path,
        # required=True, # Setting this flag breaks the argparse with positional arguments. The existence of the folder is checked elsewhere
        help="Output directory to create",
        required=True,
    )

    general = subparser.add_argument_group(
        title="General optional arguments", description=None
    )
    general.add_argument(
        "-m",
        dest="minlength",
        metavar="",
        type=int,
        default=2000,
        help="Ignore contigs shorter than this [2000]",
    )

    general.add_argument(
        "-p",
        dest="nthreads",
        metavar="",
        type=int,
        default=DEFAULT_THREADS,
        help=f"number of threads to use where customizable [{DEFAULT_THREADS}]",
    )
    general.add_argument(
        "--norefcheck",
        help="Skip reference name hashing check [False]",
        action="store_true",
    )
    general.add_argument(
        "--cuda", help="Use GPU to train & cluster [False]", action="store_true"
    )
    general.add_argument(
        "--seed",
        metavar="",
        type=int,
        default=int.from_bytes(os.urandom(7), "little"),
        help="Random seed (determinism not guaranteed)",
    )
    return subparser


def add_composition_arguments(subparser: argparse.ArgumentParser):
    tnfos = subparser.add_argument_group(title="Composition input")
    tnfos.add_argument("--fasta", metavar="", type=Path, help="Path to fasta file")
    tnfos.add_argument(
        "--composition", metavar="", type=Path, help="Path to .npz of composition"
    )
    return subparser


def add_abundance_arguments(subparser: argparse.ArgumentParser):
    abundanceos = subparser.add_argument_group(title="Abundance input")
    # Note: This argument is deprecated, but we'll keep supporting it for now.
    # Instead, use --bamdir.
    abundanceos.add_argument(
        "--bamfiles",
        dest="bampaths",
        metavar="",
        type=Path,
        help=argparse.SUPPRESS,
        nargs="+",
    )
    abundanceos.add_argument(
        "--bamdir",
        metavar="",
        type=Path,
        help="Dir with .bam files to use",
    )
    abundanceos.add_argument(
        "--abundance_tsv",
        metavar="",
        type=Path,
        help='Path to TSV file of precomputed abundances with header being "contigname(\\t<samplename>)*"',
    )
    abundanceos.add_argument(
        "--abundance",
        metavar="",
        dest="abundancepath",
        type=Path,
        help="Path to .npz of abundances",
    )
    abundanceos.add_argument(
        "-z",
        dest="min_alignment_id",
        metavar="",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    return subparser


def add_taxonomy_arguments(subparser: argparse.ArgumentParser, taxonomy_only=False):
    taxonomys = subparser.add_argument_group(title="Taxonomy input")
    taxonomys.add_argument(
        "--taxonomy",
        metavar="",
        type=Path,
        help="Path to the taxonomy file",
    )
    if not taxonomy_only:
        taxonomys.add_argument(
            "--no_predictor",
            help="Do not complete input taxonomy with Taxometer [False]",
            action="store_true",
        )
    return subparser


def add_marker_arguments(subparser: argparse.ArgumentParser):
    marker_s = subparser.add_argument_group(title="Marker gene input")
    marker_s.add_argument(
        "--markers",
        metavar="",
        type=Path,
        help="Path to the marker .npz file",
    )
    marker_s.add_argument(
        "--hmm_path",
        metavar="",
        type=Path,
        help="Path to the .hmm file of marker gene profiles",
    )
    return subparser


def add_bin_output_arguments(subparser: argparse.ArgumentParser):
    bin_os = subparser.add_argument_group(title="Bin output options", description=None)
    bin_os.add_argument(
        "--minfasta",
        dest="min_fasta_output_size",
        metavar="",
        type=int,
        default=None,
        help="Minimum bin size to output as fasta [None = no files]",
    )
    bin_os.add_argument(
        "-o",
        dest="binsplit_separator",
        metavar="",
        type=str,
        # This means: None is not passed, "" if passed but empty, e.g. "-o -c 5"
        # otherwise a string.
        default=None,
        const="",
        nargs="?",
        help="Binsplit separator [C if present] (pass empty string to disable)",
    )
    return subparser


def add_vae_arguments(subparser: argparse.ArgumentParser):
    # VAE arguments
    vaeos = subparser.add_argument_group(title="VAE options", description=None)

    vaeos.add_argument(
        "-n",
        dest="nhiddens",
        metavar="",
        type=int,
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    vaeos.add_argument(
        "-l",
        dest="nlatent",
        metavar="",
        type=int,
        default=32,
        help=argparse.SUPPRESS,
    )
    vaeos.add_argument(
        "-a",
        dest="alpha",
        metavar="",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    vaeos.add_argument(
        "-b",
        dest="beta",
        metavar="",
        type=float,
        default=200.0,
        help=argparse.SUPPRESS,
    )
    vaeos.add_argument(
        "-d",
        dest="dropout",
        metavar="",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )

    trainos = subparser.add_argument_group(title="Training options", description=None)

    trainos.add_argument(
        "-e", dest="nepochs", metavar="", type=int, default=300, help=argparse.SUPPRESS
    )
    trainos.add_argument(
        "-t",
        dest="batchsize",
        metavar="",
        type=int,
        default=256,
        help=argparse.SUPPRESS,
    )
    trainos.add_argument(
        "-q",
        dest="batchsteps",
        metavar="",
        type=int,
        nargs="*",
        default=[25, 75, 150, 225],
        help=argparse.SUPPRESS,
    )
    trainos.add_argument(
        "-r",
        dest="lrate",
        metavar="",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    return subparser


def add_predictor_arguments(subparser: argparse.ArgumentParser):
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
        help=argparse.SUPPRESS,
    )
    pred_trainos.add_argument(
        "-pt",
        dest="pred_batchsize",
        metavar="",
        type=int,
        default=1024,
        help=argparse.SUPPRESS,
    )
    pred_trainos.add_argument(
        "-pq",
        dest="pred_batchsteps",
        metavar="",
        type=int,
        nargs="*",
        default=[25, 75],
        help=argparse.SUPPRESS,
    )
    pred_trainos.add_argument(
        "-pthr",
        dest="pred_softmax_threshold",
        metavar="",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    pred_trainos.add_argument(
        "-ploss",
        dest="ploss",
        metavar="",
        type=str,
        choices=["flat_softmax", "cond_softmax", "soft_margin"],
        default="flat_softmax",
        help=argparse.SUPPRESS,
    )
    return subparser


def add_clustering_arguments(subparser: argparse.ArgumentParser):
    # Clustering arguments
    clusto = subparser.add_argument_group(title="Clustering options", description=None)
    clusto.add_argument(
        "-w",
        dest="window_size",
        metavar="",
        type=int,
        default=300,
        help=argparse.SUPPRESS,
    )
    clusto.add_argument(
        "-u",
        dest="min_successes",
        metavar="",
        type=int,
        default=15,
        help=argparse.SUPPRESS,
    )
    clusto.add_argument(
        "-c",
        dest="max_clusters",
        metavar="",
        type=int,
        default=None,  # meaning: do not stop
        help=argparse.SUPPRESS,
    )

    return subparser


def add_model_selection_arguments(subparser: argparse.ArgumentParser):
    # Model selection argument
    model_selection = subparser.add_argument_group(
        title="Model selection", description=None
    )

    model_selection.add_argument(
        "--model",
        dest="model",
        metavar="",
        type=str,
        choices=["vae", "aae", "vae-aae", "vaevae", "vae_asy", "vae_sy"],
        default="vae",
        help="Choose which model to run; only vae (vae); only aae (aae); the combination of vae and aae (vae-aae); taxonomy_predictor; vaevae; reclustering [vae]",
    )
    return subparser


def add_aae_arguments(subparser: argparse.ArgumentParser):
    # AAE arguments
    aaeos = subparser.add_argument_group(title="AAE options", description=None)

    aaeos.add_argument(
        "--n_aae",
        dest="nhiddens_aae",
        metavar="",
        type=int,
        default=547,
        help=argparse.SUPPRESS,
    )
    aaeos.add_argument(
        "--z_aae",
        dest="nlatent_aae_z",
        metavar="",
        type=int,
        default=283,
        help=argparse.SUPPRESS,
    )
    aaeos.add_argument(
        "--y_aae",
        dest="nlatent_aae_y",
        metavar="",
        type=int,
        default=700,
        help=argparse.SUPPRESS,
    )
    aaeos.add_argument(
        "--sl_aae",
        dest="sl",
        metavar="",
        type=float,
        default=0.00964,
        help=argparse.SUPPRESS,
    )
    aaeos.add_argument(
        "--slr_aae",
        dest="slr",
        metavar="",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    aaeos.add_argument(
        "--aae_temp",
        dest="temp",
        metavar="",
        type=float,
        default=0.1596,
        help=argparse.SUPPRESS,
    )

    aaeos.add_argument(
        "--e_aae",
        dest="nepochs_aae",
        metavar="",
        type=int,
        default=70,
        help=argparse.SUPPRESS,
    )
    aaeos.add_argument(
        "--t_aae",
        dest="batchsize_aae",
        metavar="",
        type=int,
        default=256,
        help=argparse.SUPPRESS,
    )
    aaeos.add_argument(
        "--q_aae",
        dest="batchsteps_aae",
        metavar="",
        type=int,
        nargs="*",
        default=[25, 50],
        help=argparse.SUPPRESS,
    )
    return subparser


def add_reclustering_arguments(subparser: argparse.ArgumentParser):
    # Reclustering arguments
    reclusters = subparser.add_argument_group(title="K-means reclustering arguments")
    reclusters.add_argument(
        "--latent_path",
        metavar="",
        type=Path,
        help="Path to latent space .npz file",
    )
    reclusters.add_argument(
        "--clusters_path",
        metavar="",
        type=Path,
        help="Path to TSV file with clusters",
    )
    reclusters.add_argument(
        "--algorithm",
        metavar="",
        type=str,
        default="kmeans",
        choices=["kmeans", "dbscan"],
        help="Which reclustering algorithm to use ('kmeans', 'dbscan') [kmeans]",
    )
    return subparser


def add_contr_vae_arguments(subparser: argparse.ArgumentParser):
    # Reclustering arguments
    contr_vaeos = subparser.add_argument_group(title="Contrastive-VAE arguments")
    contr_vaeos.add_argument(
        "--Rc",
        dest="radius_clustering",
        help="radius clustering within radius  [0.01]",
        type=float,
        default=0.01,
        
    )
    contr_vaeos.add_argument(
        "--top_neighbours",
        dest="top_neighbours",
        help="only consider n closest contigs in embedding space as neighbours [50]",
        type=int,
        default=50,
    )    
    contr_vaeos.add_argument(
        "--neighs",
        dest="neighs_path",
        metavar="",
        type=Path,
        help="paths to graph neighs obj, where each row (contig) indicates the neibouring contig indices",
    )
    contr_vaeos.add_argument(
        "--gamma",
        dest="gamma",
        help="Weight for the embeddings contrastive losss [0.1]",
        type=float,
        default=0.1,
    )
    contr_vaeos.add_argument(
        "--margin",
        dest="margin",
        help="Margin where cosine distances stop being penalized [0.01]",
        type=float,
        default=0.01,
    )
    contr_vaeos.add_argument(
        "--max_neighs_r",
        dest="max_neighs_r",
        help="Neighbours further than this on latent space will be removed [0.2]",
        type=float,
        default=0.2,
    )
    
    return subparser


def main():
    doc = f"""
    Version: {vamb.__version_str__}

    Default use, good for most datasets:
    vamb bin default --outdir out --fasta my_contigs.fna --bamdir bam_dir

    Find the latest updates and documentation at https://github.com/RasmussenLab/vamb"""
    logger.add(sys.stderr, format=format_log)

    parser = argparse.ArgumentParser(
        prog="vamb",
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    add_help_arguments(parser)

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
    CONTR_VAMB = "contr_vamb"

    vaevae_parserbin_parser = subparsers.add_parser(
        BIN,
        help="""
        VAMB, TaxVAMB, AVAMB,CONTR_VAMB binners
        """,
        add_help=False,
    )
    add_help_arguments(vaevae_parserbin_parser)
    subparsers_model = vaevae_parserbin_parser.add_subparsers(dest="model_subcommand")

    vae_parser = subparsers_model.add_parser(
        VAMB,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="""
        default binner based on a variational autoencoder. 
        See the paper "Improved metagenome binning and assembly using deep variational autoencoders" 
        (https://www.nature.com/articles/s41587-020-00777-4)
        """,
        add_help=False,
        usage="%(prog)s [options]",
        description="""Bin using a VAE that merges composition and abundance information.

Required arguments: Outdir, at least one composition input and at least one abundance input""",
    )
    add_general_arguments(vae_parser)
    add_composition_arguments(vae_parser)
    add_abundance_arguments(vae_parser)
    add_bin_output_arguments(vae_parser)
    add_vae_arguments(vae_parser)
    add_clustering_arguments(vae_parser)

    vaevae_parser = subparsers_model.add_parser(
        TAXVAMB,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="""
        taxonomy informed binner based on a bi-modal variational autoencoder. 
        See the paper "TaxVAMB: taxonomic annotations improve metagenome binning" 
        (link)
        """,
        add_help=False,
        usage="%(prog)s [options]",
        description="""Bin using a semi-supervised VAEVAE model that merges composition, abundance and taxonomic information.

Required arguments: Outdir, taxonomy, at least one composition input and at least one abundance input""",
    )
    add_general_arguments(vaevae_parser)
    add_composition_arguments(vaevae_parser)
    add_abundance_arguments(vaevae_parser)
    add_taxonomy_arguments(vaevae_parser)
    add_bin_output_arguments(vaevae_parser)
    add_vae_arguments(vaevae_parser)
    add_clustering_arguments(vaevae_parser)
    add_predictor_arguments(vaevae_parser)

    vaeaae_parser = subparsers_model.add_parser(
        AVAMB,
        help="""
        Ensemble model of an adversarial autoencoder and a variational autoencoder. See the paper "Adversarial and variational autoencoders improve metagenomic binning
                        (https://www.nature.com/articles/s42003-023-05452-3). WARNING: recommended use is through the Snakemake pipeline
        """,
        #argparse.SUPPRESS,
        add_help=False,
        usage="%(prog)s [options]",
    )
    add_general_arguments(vaeaae_parser)
    add_composition_arguments(vaeaae_parser)
    add_abundance_arguments(vaeaae_parser)
    add_bin_output_arguments(vaeaae_parser)
    add_vae_arguments(vaeaae_parser)
    add_aae_arguments(vaeaae_parser)
    add_clustering_arguments(vaeaae_parser)


    contr_vae_parser = subparsers_model.add_parser(
        CONTR_VAMB,
        help="""
        Variational Autoencoder binner that besides coabundances and TNF also applies 
        a contrastive effect to integrate non-uniform genome features into the latent representations,
        at the moment is applied to integrate communities extracted from the assembly-alignment graph.
        """,
        add_help=False,
    )
    add_general_arguments(contr_vae_parser)
    add_composition_arguments(contr_vae_parser)
    add_abundance_arguments(contr_vae_parser)
    add_bin_output_arguments(contr_vae_parser)
    add_vae_arguments(contr_vae_parser)
    add_contr_vae_arguments(contr_vae_parser)
    add_clustering_arguments(contr_vae_parser)
    

    predict_parser = subparsers.add_parser(
        TAXOMETER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="""
        refines taxonomic annotations of any metagenome classifier. 
        See the paper "Taxometer: deep learning improves taxonomic classification of contigs using binning features and a hierarchical loss" 
        (link)
        """,
        add_help=False,
        usage="%(prog)s [options]",
        description="""Refine taxonomy using composition and abundance information.

Required arguments: Outdir, unrefined taxonomy, at least one composition input and at least one abundance input""",
    )
    add_general_arguments(predict_parser)
    add_composition_arguments(predict_parser)
    add_abundance_arguments(predict_parser)
    add_taxonomy_arguments(predict_parser, taxonomy_only=True)
    add_predictor_arguments(predict_parser)

    recluster_parser = subparsers.add_parser(
        RECLUSTER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="""
        reclustering using single-copy genes for the binning results of VAMB, TaxVAMB or AVAMB
        """,
        add_help=False,
        usage="%(prog)s [options]",
        description="""Use marker genes to re-cluster (DBScan) or refine (K-means) clusters.
        
Required arguments:
  K-means algorithm: Outdir, at least one composition input, at least one marker gene input,
    latent path and clusters path
  DBScan algorithm with --no_predictor: Outdir, at least one composition input,
    at least one marker gene input, latent path and taxonomy
  DBScan algorithm without --no_predictor: Outdir, at least one composition input,
    at least one abundance input, at least one marker gene input, latent path and taxonomy
""",
    )
    add_general_arguments(recluster_parser)
    add_composition_arguments(recluster_parser)
    add_abundance_arguments(recluster_parser)
    add_marker_arguments(recluster_parser)
    add_bin_output_arguments(recluster_parser)
    add_reclustering_arguments(recluster_parser)
    add_predictor_arguments(recluster_parser)
    add_taxonomy_arguments(recluster_parser)


    args = parser.parse_args()

    if args.subcommand == TAXOMETER:
        opt = TaxometerOptions.from_args(args)
        runner = partial(run_taxonomy_predictor, opt)
        run(runner, opt.general)
    elif args.subcommand == BIN:
        model = args.model_subcommand
        if model is None:
            vaevae_parserbin_parser.print_help()
            sys.exit(1)
        if model == VAMB:
            opt = BinDefaultOptions.from_args(args)
            runner = partial(run_bin_default, opt)
            run(runner, opt.common.general)
        elif model == TAXVAMB:
            opt = BinTaxVambOptions.from_args(args)
            runner = partial(run_vaevae, opt)
            run(runner, opt.common.general)
        elif model == AVAMB:
            opt = BinAvambOptions.from_args(args)
            runner = partial(run_bin_aae, opt)
            run(runner, opt.common.general)
        elif model == CONTR_VAMB:
            opt = BinContrVambOptions.from_args(args)
            runner = partial(run_bin_contr_vamb, opt)
            run(runner, opt.common.general)
    
    elif args.subcommand == RECLUSTER:
        opt = ReclusteringOptions.from_args(args)
        runner = partial(run_reclustering, opt)
        run(runner, opt.general)
    else:
        # There are no more subcommands
        assert False


if __name__ == "__main__":
    main()

