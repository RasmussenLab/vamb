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
from collections import defaultdict
from torch.utils.data import DataLoader
from functools import partial
from loguru import logger

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
    def __init__(self, abundance: Path):
        self.abundance = check_existing_file(abundance)


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


class AEMBPaths(list):
    __slots__ = ["paths"]

    def __init__(self, dir: Path):
        if not dir.is_dir():
            raise NotADirectoryError(dir)
        paths = [b for b in dir.iterdir() if b.is_file() and b.suffix == ".tsv"]
        if len(paths) == 0:
            raise ValueError(
                f'No `.tsv` files found in --aemb argument "{dir}". '
                "Make sure all TSV files in the directory ends with `.tsv`."
            )
        self.paths = paths


class AbundanceOptions:
    __slots__ = ["paths"]

    @staticmethod
    def are_args_present(args: argparse.Namespace) -> bool:
        return (
            isinstance(args.bampaths, list)
            or isinstance(args.bamdir, Path)
            or isinstance(args.aemb, Path)
            or isinstance(args.abundancepath, Path)
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(
            typeasserted(args.bampaths, (list, type(None))),
            typeasserted(args.bamdir, (Path, type(None))),
            typeasserted(args.aemb, (Path, type(None))),
            typeasserted(args.abundancepath, (Path, type(None))),
            typeasserted(args.min_alignment_id, (float, type(None))),
        )

    def __init__(
        self,
        bampaths: Optional[list[Path]],
        bamdir: Optional[Path],
        aemb: Optional[Path],
        abundancepath: Optional[Path],
        min_alignment_id: Optional[float],
    ):
        assert isinstance(bampaths, (list, type(None)))
        assert isinstance(bamdir, (Path, type(None)))
        assert isinstance(aemb, (Path, type(None)))
        assert isinstance(abundancepath, (Path, type(None)))
        assert isinstance(min_alignment_id, (float, type(None)))

        # Make sure only one abundance input is there
        if (
            (bampaths is not None)
            + (abundancepath is not None)
            + (bamdir is not None)
            + (aemb is not None)
        ) != 1:
            raise argparse.ArgumentTypeError(
                "Must specify exactly one of BAM files, BAM dir, AEMB or abundance NPZ file input"
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
        elif aemb is not None:
            self.paths = AEMBPaths(aemb)


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
            typeasserted(args.pred_batchsteps, list),
        )

    def __init__(
        self, num_epochs: int, starting_batch_size: int, batch_steps: list[int]
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


class VAEOptions:
    __slots__ = ["nhiddens", "nlatent", "alpha", "beta", "dropout", "basic_options"]

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


class TaxonomyPath:
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(typeasserted(args.taxonomy, Path))

    def __init__(self, path: Path):
        self.path = check_existing_file(path)


class TaxometerOptions:
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(
            GeneralOptions.from_args(args),
            CompositionOptions.from_args(args),
            AbundanceOptions.from_args(args),
            TaxonomyPath.from_args(args),
            BasicTrainingOptions.from_args_taxometer(args),
            typeasserted(args.pred_softmax_threshold, float),
            args.ploss,
        )

    def __init__(
        self,
        general: GeneralOptions,
        composition: CompositionOptions,
        abundance: AbundanceOptions,
        taxonomy: TaxonomyPath,
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


class RefinableTaxonomyOptions:
    __slots__ = ["path_or_tax_options"]

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        predict = not typeasserted(args.no_predictor, bool)

        # TaxometerOptions have more options, but only the composition and the abundance
        # can be omitted, so we only check those here.
        if predict:
            if not CompositionOptions.are_args_present(
                args
            ) or not AbundanceOptions.are_args_present(args):
                raise ValueError(
                    "If `--no_predictor` is not passed, Taxometer is run to refine taxonomy, "
                    "and this requires composition input and abundance input to be passed in"
                )
            return cls(TaxometerOptions.from_args(args))
        else:
            return cls(TaxonomyPath.from_args(args))

    def __init__(self, path_or_tax_options: Union[TaxonomyPath, TaxometerOptions]):
        self.path_or_tax_options = path_or_tax_options


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
        hmm = typeasserted(args.hmmout_path, (Path, type(None)))
        if not (markers is None) ^ (hmm is None):
            raise ValueError(
                "Exactly one of --markers or --hmmout_path must be set to input taxonomic information"
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
    def __init__(self, taxonomy_options: RefinableTaxonomyOptions, n_processes: int):
        self.taxonomy_options = taxonomy_options
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
                    "If minfasta is not None, "
                    "input fasta file must be given explicitly"
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
    logger.info(f"Invoked with CLI args: 'f{' '.join(sys.argv)}'")
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
        taxonomy = RefinableTaxonomyOptions.from_args(args)
        return cls(common, vae, taxonomy)

    # The VAEVAE models share the same settings as the VAE model, so we just use VAEOptions
    def __init__(
        self,
        common: BinnerCommonOptions,
        vae: VAEOptions,
        taxonomy: RefinableTaxonomyOptions,
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
            tax = RefinableTaxonomyOptions.from_args(args)
            algorithm = DBScanOptions(tax, general.n_threads)
        else:
            assert False  # no more algorithms

        if isinstance(algorithm, DBScanOptions) and isinstance(
            algorithm.taxonomy_options.path_or_tax_options, TaxometerOptions
        ):
            comp = algorithm.taxonomy_options.path_or_tax_options.composition
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
        logger.info(f"\tLoading composition from npz at: {path.path}")
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
        f'\tReference hash: {comp_metadata.refhash.hex() if refcheck else "None"}'
    )

    paths = abundance_options.paths
    if isinstance(paths, AbundancePath):
        logger.info(f"\tLoading depths from npz at: {str(paths)}")

        abundance = vamb.parsebam.Abundance.load(
            paths.abundance,
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
    elif isinstance(paths, AEMBPaths):
        paths = list(paths.paths)
        logger.info(f"\tParsing {len(paths)} AEMB TSV files")
        abundance = vamb.parsebam.Abundance.from_aemb(
            paths,
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
        print("name\tradius\tpeak valley ratio\tkind\tbp\tncontigs", file=file)
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
                file=file,
                sep="\t",
            )

    elapsed = round(time.time() - begintime, 2)

    write_clusters_and_bins(
        fasta_output,
        binsplitter,
        base_clusters_name,
        cluster_dict,
        sequence_names,
        cast(Sequence[int], sequence_lens),
    )
    logger.info(f"\tClustered contigs in {elapsed} seconds.\n")


def write_clusters_and_bins(
    fasta_output: Optional[FastaOutput],
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
    )
    del latent_z

    # We enforce this in the VAEAAEOptions constructor, see comment there
    assert opt.common.clustering.max_clusters is None
    write_clusters_and_bins(
        FastaOutput.try_from_common(opt.common),
        binsplitter=opt.common.output.binsplitter,
        base_clusters_name=str(opt.common.general.out_dir.joinpath("aae_y_clusters")),
        clusters=clusters_y_dict,
        sequence_names=cast(Sequence[str], comp_metadata.identifiers),
        sequence_lens=cast(Sequence[int], comp_metadata.lengths),
    )


def predict_taxonomy(
    abundance: np.ndarray,
    tnfs: np.ndarray,
    lengths: np.ndarray,
    contignames: np.ndarray,
    taxonomy_path: Path,
    out_dir: Path,
    taxonomy_options: TaxometerOptions,
    cuda: bool,
):
    begintime = time.time()

    taxonomies = vamb.vambtools.parse_taxonomy(
        input_file=taxonomy_path,
        contignames=cast(Sequence[str], contignames),
        is_canonical=False,
    )
    nodes, ind_nodes, table_parent = vamb.taxvamb_encode.make_graph(taxonomies)
    logger.info(f"{len(nodes)} nodes in the graph")
    classes_order: list[str] = []
    for i in taxonomies:
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
            threshold_mask = predicted_vector[i] > taxonomy_options.softmax_threshold
            pred_line = ";".join(nodes_ar[threshold_mask][1:])
            predictions.append(pred_line)
            absolute_probs = predicted_vector[i][threshold_mask]
            absolute_prob = ";".join(map(str, absolute_probs[1:]))
            probs.append(absolute_prob)

        with open(predicted_path, "a") as file:
            if not output_exists:
                print("contigs\tpredictions\tlengths\tscores", file=file)
            for contig, length, prediction, prob in zip(
                contignames[row : row + N], predictions, lengths[row : row + N], probs
            ):
                print(contig, length, prediction, prob, file=file, sep="\t")
        output_exists = True
        row += N

    logger.info(
        f"Completed taxonomy predictions in {round(time.time() - begintime, 2)} seconds."
    )


def run_taxonomy_predictor(opt: TaxometerOptions):
    composition, abundance = load_composition_and_abundance(
        opt.general,
        opt.composition,
        opt.abundance,
        vamb.vambtools.BinSplitter.inert_splitter(),
    )
    abundance, tnfs, lengths, contignames = (
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        composition.metadata.identifiers,
    )
    predict_taxonomy(
        abundance=abundance,
        tnfs=tnfs,
        lengths=lengths,
        contignames=contignames,
        taxonomy_path=opt.taxonomy.path,
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
        binsplitter=vamb.vambtools.BinSplitter.inert_splitter(),
    )
    abundance, tnfs, lengths, contignames = (
        abundance.matrix,
        composition.matrix,
        composition.metadata.lengths,
        composition.metadata.identifiers,
    )
    if isinstance(opt.taxonomy.path_or_tax_options, TaxometerOptions):
        logger.info("Predicting missing values from taxonomy")
        predict_taxonomy(
            abundance=abundance,
            tnfs=tnfs,
            lengths=lengths,
            contignames=contignames,
            taxonomy_path=opt.taxonomy.path_or_tax_options.taxonomy.path,
            out_dir=opt.common.general.out_dir,
            taxonomy_options=opt.taxonomy.path_or_tax_options,
            cuda=opt.common.general.cuda,
        )
        contig_taxonomies = vamb.vambtools.parse_taxonomy(
            input_file=opt.common.general.out_dir.joinpath("results_taxometer.tsv"),
            contignames=cast(Sequence[str], contignames),
            is_canonical=False,
        )
    else:
        logger.info("Not predicting the taxonomy")
        contig_taxonomies = vamb.vambtools.parse_taxonomy(
            input_file=opt.taxonomy.path_or_tax_options.path,
            contignames=cast(Sequence[str], contignames),
            is_canonical=False,
        )

    nodes, ind_nodes, table_parent = vamb.taxvamb_encode.make_graph(contig_taxonomies)
    classes_order: list[str] = []
    for i in contig_taxonomies:
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
    )


def run_reclustering(opt: ReclusteringOptions):
    composition = calc_tnf(
        opt.composition,
        opt.general.min_contig_length,
        opt.general.out_dir,
        opt.output.binsplitter,
    )
    logger.info(f"{len(composition.metadata.identifiers)} contig names after filtering")
    logger.info("Taxonomy predictions are provided")
    markers = opt.markers.content
    assert isinstance(markers, PredictMarkers)
    reclustered = vamb.reclustering.recluster_bins(
        None if isinstance(opt.algorithm, DBScanOptions) else opt.algorithm.clusters,
        opt.latent_path,
        str(opt.composition.path.path),
        composition.metadata.identifiers,
        out_dir=opt.general.out_dir,
        minfasta=0,
        binned_length=1000,
        num_process=40,
        random_seed=opt.general.seed,
        hmmout_path=markers.hmm,
        algorithm="kmeans" if isinstance(opt.algorithm, KmeansOptions) else "dbscan",
        predictions_path=None
        if isinstance(opt.algorithm, KmeansOptions)
        else (
            opt.algorithm.taxonomy_options.path_or_tax_options.path
            if isinstance(
                opt.algorithm.taxonomy_options.path_or_tax_options, TaxonomyPath
            )
            else opt.algorithm.taxonomy_options.path_or_tax_options.taxonomy.path
        ),
    )
    cluster_dict: defaultdict[str, set[str]] = defaultdict(set)
    for k, v in zip(reclustered, composition.metadata.identifiers):
        cluster_dict[cast(str, k)].add(v)

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
        opt.output.binsplitter,
        str(opt.general.out_dir.joinpath("clusters_reclustered")),
        cluster_dict,
        cast(Sequence[str], composition.metadata.identifiers),
        cast(Sequence[int], composition.metadata.lengths),
    )


def add_help_arguments(parser: argparse.ArgumentParser):
    helpos = parser.add_argument_group(title="Help and version", description=None)
    helpos.add_argument("-h", "--help", help="print help and exit", action="help")
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
        help="output directory to create",
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
        help="ignore contigs shorter than this [2000]",
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
        help="skip reference name hashing check [False]",
        action="store_true",
    )
    general.add_argument(
        "--cuda", help="use GPU to train & cluster [False]", action="store_true"
    )
    general.add_argument(
        "--seed",
        metavar="",
        type=int,
        default=int.from_bytes(os.urandom(7), "little"),
        help="Random seed (random determinism not guaranteed)",
    )
    return subparser


def add_composition_arguments(subparser: argparse.ArgumentParser):
    tnfos = subparser.add_argument_group(title="Composition input")
    tnfos.add_argument("--fasta", metavar="", type=Path, help="path to fasta file")
    tnfos.add_argument(
        "--composition", metavar="", type=Path, help="path to .npz of composition"
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
        "--aemb",
        metavar="",
        type=Path,
        help="Dir with .tsv files from strobealign --aemb",
    )
    abundanceos.add_argument(
        "--abundance",
        metavar="",
        dest="abundancepath",
        type=Path,
        help="path to .npz of abundances",
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
        help="path to the taxonomy file",
    )
    if not taxonomy_only:
        taxonomys.add_argument(
            "--no_predictor",
            help="do not complete input taxonomy with Taxometer [False]",
            action="store_true",
        )
    return subparser


def add_marker_arguments(subparser: argparse.ArgumentParser):
    marker_s = subparser.add_argument_group(title="Marker gene input")
    marker_s.add_argument(
        "--markers",
        metavar="",
        type=Path,
        help="path to the marker .npz file",
    )
    marker_s.add_argument(
        "--hmmout_path",
        metavar="",
        type=Path,
        help="path to the .hmm file of marker gene profiles",
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
        help="minimum bin size to output as fasta [None = no files]",
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
        help="binsplit separator [C if present] (pass empty string to disable)",
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
        type=Optional[float],
        default=None,
        help=argparse.SUPPRESS,
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
        default=[25, 75, 150, 225],
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


def add_clustering_arguments(subparser):
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
        "--algorithm",
        metavar="",
        type=str,
        default="kmeans",
        choices=["kmeans", "dbscan"],
        help="which reclustering algorithm to use ('kmeans', 'dbscan'). DBSCAN requires a taxonomy predictions file [kmeans]",
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

    vaevae_parserbin_parser = subparsers.add_parser(
        BIN,
        help="""
        VAMB, TaxVAMB, AVAMB binners
        """,
        add_help=False,
    )
    add_help_arguments(vaevae_parserbin_parser)
    subparsers_model = vaevae_parserbin_parser.add_subparsers(dest="model_subcommand")

    vae_parser = subparsers_model.add_parser(
        VAMB,
        help="""
        default binner based on a variational autoencoder. 
        See the paper "Improved metagenome binning and assembly using deep variational autoencoders" 
        (https://www.nature.com/articles/s41587-020-00777-4)
        """,
        add_help=False,
        usage="%(prog)s [options]",
    )
    add_general_arguments(vae_parser)
    add_composition_arguments(vae_parser)
    add_abundance_arguments(vae_parser)
    add_bin_output_arguments(vae_parser)
    add_vae_arguments(vae_parser)
    add_clustering_arguments(vae_parser)

    vaevae_parser = subparsers_model.add_parser(
        TAXVAMB,
        help="""
        taxonomy informed binner based on a bi-modal variational autoencoder. 
        See the paper "TaxVAMB: taxonomic annotations improve metagenome binning" 
        (link)
        """,
        add_help=False,
        usage="%(prog)s [options]",
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
        ensemble model of an adversarial autoencoder and a variational autoencoder. 
        See the paper "Adversarial and variational autoencoders improve metagenomic binning" 
        (https://www.nature.com/articles/s42003-023-05452-3). 
        WARNING: recommended use is through the Snakemake pipeline
        """,
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

    predict_parser = subparsers.add_parser(
        TAXOMETER,
        help="""
        refines taxonomic annotations of any metagenome classifier. 
        See the paper "Taxometer: deep learning improves taxonomic classification of contigs using binning features and a hierarchical loss" 
        (link)
        """,
        add_help=False,
        usage="%(prog)s [options]",
    )
    add_general_arguments(predict_parser)
    add_composition_arguments(predict_parser)
    add_abundance_arguments(predict_parser)
    add_taxonomy_arguments(predict_parser, taxonomy_only=True)
    add_predictor_arguments(predict_parser)

    recluster_parser = subparsers.add_parser(
        RECLUSTER,
        help="""
        reclustering using single-copy genes for the binning results of VAMB, TaxVAMB or AVAMB
        """,
        add_help=False,
        usage="%(prog)s [options]",
    )
    add_general_arguments(recluster_parser)
    add_composition_arguments(recluster_parser)
    add_abundance_arguments(recluster_parser)
    add_marker_arguments(recluster_parser)
    add_bin_output_arguments(recluster_parser)
    add_reclustering_arguments(recluster_parser)
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
    elif args.subcommand == RECLUSTER:
        opt = ReclusteringOptions.from_args(args)
        runner = partial(run_reclustering, opt)
        run(runner, opt.general)
    else:
        # There are no more subcommands
        assert False


if __name__ == "__main__":
    main()
