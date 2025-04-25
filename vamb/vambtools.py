__doc__ = "Various classes and functions Vamb uses internally."

import gzip as _gzip
import bz2 as _bz2
import lzma as _lzma
import numpy as _np
import re as _re
from vambcore import kmercounts, overwrite_matrix
import collections as _collections
from itertools import zip_longest
from hashlib import md5 as _md5
from collections.abc import Iterable, Iterator
from typing import Optional, IO, Union
from pathlib import Path
from loguru import logger

CLUSTERS_HEADER = "clustername\tcontigname"


def log_and_error(typ: type, msg: str):
    logger.opt(raw=True).info("\n")
    logger.error(msg)
    raise typ(msg)


class BinSplitter:
    """
    The binsplitter can be either
    * Instantiated with an explicit option, in which case `is_default` is False,
      and `splitter` is None if the user explicitly opted out of binsplitting by
      passing in an empty string, else a `str`, the string explicitly asked for
    * Instantiated by default, in which case `is_default` is `True`, and `splitter`
      is `_DEFAULT_SPLITTER`

    The `initialize` function checks the validity of the binsplitter on the set of
    identifiers:
    * If the binsplitter sep is explicitly `None`, do nothing
    * If the binsplitter is default and the separator is not found, warn the user,
      then set the separator to `None` (disabling binsplitting)
    * If the binsplitter is explicitly a string, check this string occurs in all
      identifiers, else error early.
    """

    _DEFAULT_SPLITTER = "C"
    __slots__ = ["is_default", "splitter", "is_initialized"]

    def __init__(self, binsplitter: Optional[str]):
        if binsplitter is None:
            self.is_default = True
            self.splitter = self._DEFAULT_SPLITTER

        else:
            self.is_default = False
            if len(binsplitter) == 0:
                self.splitter = None
            else:
                self.splitter = binsplitter

        self.is_initialized = False

    @classmethod
    def inert_splitter(cls):
        return cls("")

    def initialize(self, identifiers: Iterable[str]):
        if self.is_initialized:
            return None
        self.is_initialized = True
        separator = self.splitter
        if separator is None:
            return None
        message = (
            'Binsplit separator (option `-o`) {imexplicit} passed as "{separator}", '
            'but sequence identifier "{identifier}" does not contain this separator, '
            "or contains it at the very start or end.\n"
            "A binsplit separator X implies that every sequence identifier is formatted as\n"
            "[sample identifier][X][sequence identifier], e.g. a binsplit separator of 'C' "
            "means that 'S1C19' and '7C11' are valid identifiers.\n"
        )

        if not self.is_default:
            for identifier in identifiers:
                (front, _, rest) = identifier.partition(separator)
                if not front or not rest:
                    log_and_error(
                        ValueError,
                        message.format(
                            imexplicit="explicitly",
                            separator=separator,
                            identifier=identifier,
                        ),
                    )
        else:
            for identifier in identifiers:
                (front, _, rest) = identifier.partition(separator)
                if not front or not rest:
                    message += "\nSkipping binsplitting."
                    logger.opt(raw=True).info("\n")
                    logger.warning(
                        message.format(
                            imexplicit="implicitly",
                            separator=separator,
                            identifier=identifier,
                        )
                    )
                    self.splitter = None
                    break

    def split_bin(
        self,
        binname: str,
        identifiers: Iterable[str],
    ) -> Iterable[tuple[str, set[str]]]:
        "Split a single bin by the prefix of the headers"
        if self.splitter is None:
            yield (binname, set(identifiers))
            return None
        else:
            by_sample: dict[str, set[str]] = _collections.defaultdict(set)
            for identifier in identifiers:
                sample, _, rest = identifier.partition(self.splitter)

                if not rest or not sample:
                    raise KeyError(
                        f"Separator '{self.splitter}' not in sequence identifier, or is at the very start or end of identifier: '{identifier}'"
                    )

                by_sample[sample].add(identifier)

            for sample, splitheaders in by_sample.items():
                newbinname = f"{sample}{self.splitter}{binname}"
                yield newbinname, splitheaders

    def binsplit(
        self,
        clusters: Iterable[tuple[str, Iterable[str]]],
    ) -> Iterable[tuple[str, set[str]]]:
        """Splits a set of clusters by the prefix of their names.
        The separator is a string which separated prefix from postfix of contignames. The
        resulting split clusters have the prefix and separator prepended to them.

        clusters can be an iterator, in which case this function returns an iterator, or a dict
        with contignames: set_of_contignames pair, in which case a dict is returned.

        Example:
        >>> clusters = {"bin1": {"s1-c1", "s1-c5", "s2-c1", "s2-c3", "s5-c8"}}
        >>> binsplit(clusters, "-")
        {'s2-bin1': {'s1-c1', 's1-c3'}, 's1-bin1': {'s1-c1', 's1-c5'}, 's5-bin1': {'s1-c8'}}
        """
        for binname, headers in clusters:
            for newbinname, splitheaders in self.split_bin(binname, headers):
                yield newbinname, splitheaders

    def log_string(self) -> str:
        if not self.is_default:
            if self.splitter is None:
                return "Explicitly passed as empty (no binsplitting)"
            else:
                return f'"{self.splitter}"'
        else:
            if self.splitter is None:
                return "Defaulting to 'C', but disabled due to incompatible identifiers"
            else:
                return "Defaulting to 'C'"


class PushArray:
    """Data structure that allows efficient appending and extending a 1D Numpy array.
    Intended to strike a balance between not resizing too often (which is slow), and
    not allocating too much at a time (which is memory inefficient).

    Usage:
    >>> arr = PushArray(numpy.float64)
    >>> arr.append(5.0)
    >>> arr.extend(numpy.linspace(4, 3, 3))
    >>> arr.take() # return underlying Numpy array
    array([5. , 4. , 3.5, 3. ])
    """

    __slots__ = ["data", "capacity", "length"]

    def __init__(self, dtype, start_capacity: int = 1 << 16):
        self.capacity: int = start_capacity
        self.data: _np.ndarray = _np.empty(self.capacity, dtype=dtype)
        self.length = 0

    def __len__(self) -> int:
        return self.length

    def _setcapacity(self, n: int) -> None:
        self.data.resize(n, refcheck=False)
        self.capacity = n

    def _grow(self, mingrowth: int) -> None:
        """Grow capacity by power of two between 1/8 and 1/4 of current capacity, though at
        least mingrowth"""
        growth = max(int(self.capacity * 0.125), mingrowth)
        nextpow2 = 1 << (growth - 1).bit_length()
        self._setcapacity(self.capacity + nextpow2)

    def append(self, value) -> None:
        if self.length == self.capacity:
            self._grow(64)

        self.data[self.length] = value
        self.length += 1

    def extend(self, values) -> None:
        lenv = len(values)
        if self.length + lenv > self.capacity:
            self._grow(lenv)

        self.data[self.length : self.length + lenv] = values
        self.length += lenv

    def take(self) -> _np.ndarray:
        "Return the underlying array"
        self._setcapacity(self.length)
        return self.data

    def clear(self) -> None:
        "Empties the PushArray. Does not clear the underlying memory"
        self.length = 0


def zscore(
    array: _np.ndarray, axis: Optional[int] = None, inplace: bool = False
) -> _np.ndarray:
    """Calculates zscore for an array. A cheap copy of scipy.stats.zscore.

    Inputs:
        array: Numpy array to be normalized
        axis: Axis to operate across [None = entrie array]
        inplace: Do not create new array, change input array [False]

    Output:
        If inplace is True: Input numpy array
        else: New normalized Numpy-array
    """

    if axis is not None and (axis >= array.ndim or axis < 0):
        raise _np.exceptions.AxisError(str(axis))

    if inplace and not _np.issubdtype(array.dtype, _np.floating):
        raise TypeError("Cannot convert a non-float array to zscores")

    mean = array.mean(axis=axis)
    std = array.std(axis=axis)

    if axis is None:
        if std == 0:
            std = 1  # prevent divide by zero

    else:
        std[std == 0.0] = 1  # prevent divide by zero
        shape = tuple(dim if ax != axis else 1 for ax, dim in enumerate(array.shape))
        mean.shape, std.shape = shape, shape

    if inplace:
        array -= mean
        array /= std
        return array
    else:
        return (array - mean) / std


def numpy_inplace_maskarray(array: _np.ndarray, mask: _np.ndarray) -> _np.ndarray:
    """In-place masking of a Numpy array, i.e. if `mask` is a boolean mask of same
    length as `array`, then array[mask] == numpy_inplace_maskarray(array, mask),
    but does not allocate a new array.
    """

    if len(mask) != len(array):
        raise ValueError("Lengths of array and mask must match")
    elif len(array.shape) != 2:
        raise ValueError("Can only take a 2 dimensional-array.")

    index = overwrite_matrix(array, mask)
    array.resize((index, array.shape[1]), refcheck=False)
    return array


def torch_inplace_maskarray(array, mask):
    """In-place masking of a Tensor, i.e. if `mask` is a boolean mask of same
    length as `array`, then array[mask] == torch_inplace_maskarray(array, mask),
    but does not allocate a new tensor.
    """

    if len(mask) != len(array):
        raise ValueError("Lengths of array and mask must match")
    elif array.dim() != 2:
        raise ValueError("Can only take a 2 dimensional-array.")

    np_array = array.numpy()
    index = overwrite_matrix(np_array, _np.frombuffer(mask.numpy(), dtype=bool))
    array.resize_((index, array.shape[1]))
    return array


def mask_lower_bits(floats: _np.ndarray, bits: int) -> None:
    if bits < 0 or bits > 23:
        raise ValueError("Must mask between 0 and 23 bits")

    mask = ~_np.uint32(2**bits - 1)
    u = floats.view(_np.uint32)
    u &= mask


class Reader:
    """Use this instead of `open` to open files which are either plain text,
    gzipped, bzip2'd or zipped with LZMA.

    Usage:
    >>> with Reader(file, readmode) as file: # by default textmode
    >>>     print(next(file))
    TEST LINE
    """

    def __init__(self, filename: Union[str, Path]):
        self.filename = filename

        with open(self.filename, "rb") as f:
            signature = f.peek(8)[:8]

        # Gzipped files begin with the two bytes 0x1F8B
        if tuple(signature[:2]) == (0x1F, 0x8B):
            self.filehandle = _gzip.open(self.filename, "rb")

        # bzip2 files begin with the signature BZ
        elif signature[:2] == b"BZ":
            self.filehandle = _bz2.open(self.filename, "rb")

        # .XZ files begins with 0xFD377A585A0000
        elif tuple(signature[:7]) == (0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00):
            self.filehandle = _lzma.open(self.filename, "rb")

        # Else we assume it's a text file.
        else:
            self.filehandle = open(self.filename, "rb")

    def close(self):
        self.filehandle.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def __iter__(self):
        return self.filehandle


class FastaEntry:
    """One single FASTA entry. Instantiate with byte header and bytearray
    sequence."""

    # IUPAC ambiguous DNA letters + u
    allowed = b"acgtuswkmyrbdhvn"
    allowed += allowed.upper()
    # Allow only the same identifier chars that can be in the BAM file, otherwise
    # users will be frustrated with FASTA and BAM headers do not match.
    # BAM only includes identifier, e.g. stuff before whitespace. So we accept anything
    # after whitespace, but do not use it as header.
    # This brutal regex is derived from the SAM specs, valid identifiers,
    # but disallow leading # sign, and allow trailing whitespace + ignored description
    regex = _re.compile(
        b"([0-9A-Za-z!$%&+./:;?@^_|~-][0-9A-Za-z!#$%&*+./:;=?@^_|~-]*)([^\\S\r\n][^\r\n]*)?$"
    )
    __slots__ = ["identifier", "description", "sequence"]

    def _verify_header(self, header: bytes) -> tuple[str, str]:
        m = self.regex.match(header)
        if m is None:
            raise ValueError(
                f'Invalid header in FASTA: "{header.decode()}". '
                '\nMust conform to identifier regex pattern of SAM specification: "'
                '>([0-9A-Za-z!$%&+./:;?@^_|~-][0-9A-Za-z!#$%&*+./:;=?@^_|~-]*)([^\\S\\r\\n][^\\r\\n]*)?$".\n'
                "If the header does not fit this pattern, the header cannot be represented in BAM files, "
                "which means Vamb cannot compare sequences in BAM and FASTA files."
            )
        identifier, description = m.groups()
        description = "" if description is None else description.decode()
        return (identifier.decode(), description)

    def __init__(self, header: bytes, sequence: bytearray):
        identifier, description = self._verify_header(header)
        self.identifier: str = identifier
        self.description: str = description
        masked = sequence.translate(None, b" \t\n\r")
        stripped = masked.translate(None, self.allowed)
        if len(stripped) > 0:
            codeunit = stripped[0]
            bad_character = chr(codeunit)
            raise ValueError(
                f"Non-IUPAC DNA/RNA byte in sequence '{identifier}': '{bad_character}', byte value {codeunit}"
            )

        self.sequence: bytearray = masked

    @property
    def header(self) -> str:
        return self.identifier + self.description

    def rename(self, header: bytes) -> None:
        identifier, description = self._verify_header(header)
        self.identifier = identifier
        self.description = description

    def __len__(self) -> int:
        return len(self.sequence)

    def format(self, width: int = 60) -> str:
        sixtymers = range(0, len(self.sequence), width)
        spacedseq = "\n".join(
            [self.sequence[i : i + width].decode() for i in sixtymers]
        )
        return f">{self.header}\n{spacedseq}"

    def kmercounts(self) -> _np.ndarray:
        counts = _np.zeros(256, dtype=_np.uint32)
        kmercounts(counts, self.sequence)
        return counts


def strip_newline(s: bytes) -> bytes:
    "Remove trailing \r\n or \n from the bytestring, if present"
    if len(s) > 0 and s[-1] == 10:
        if len(s) > 1 and s[-2] == 13:
            return s[:-2]
        else:
            return s[:-1]
    else:
        return s


def strip_string_newline(s: str) -> str:
    if len(s) > 0 and s[-1] == "\n":
        if len(s) > 1 and s[-2] == "\r":
            return s[:-2]
        else:
            return s[:-1]
    else:
        return s


def byte_iterfasta(
    filehandle: Iterable[bytes], filename: Optional[str]
) -> Iterator[FastaEntry]:
    """Yields FastaEntries from a binary opened fasta file.

    Usage:
    >>> with Reader('/dir/fasta.fna') as filehandle:
    ...     entries = byte_iterfasta(filehandle, '/dir/fasta/fna') # a generator

    Inputs:
        filehandle: Any iterator of binary lines of a FASTA file
        comment: Ignore lines beginning with any whitespace + comment

    Output: Generator of FastaEntry-objects from file
    """

    # Make it work for persistent iterators, e.g. lists
    line_iterator = iter(filehandle)
    prefix = "" if filename is None else f"In file '{filename}', "
    header = next(line_iterator, None)

    # Empty file is valid - we return from the iterator
    if header is None:
        return None
    elif not isinstance(header, bytes):
        raise TypeError(
            f"{prefix}first line is not binary. Are you sure you are reading the file in binary mode?"
        )
    elif not header.startswith(b">"):
        raise ValueError(
            f"{prefix}FASTA file is invalid, first line does not begin with '>'"
        )

    # 13 is the byte value of \r, meaning we remove either \r\n or \n
    header = strip_newline(header[1:])
    buffer: list[bytes] = list()

    # Iterate over lines
    for line in line_iterator:
        if line.startswith(b">"):
            yield FastaEntry(header, bytearray().join(buffer))
            buffer.clear()
            header = strip_newline(line[1:])

        else:
            buffer.append(line)

    yield FastaEntry(header, bytearray().join(buffer))


class RefHasher:
    __slots__ = ["hasher"]

    def __init__(self):
        self.hasher = _md5()

    def add_refname(self, ref: str) -> None:
        self.hasher.update(ref.encode().rstrip())

    def add_refnames(self, refs: Iterable[str]):
        for ref in refs:
            self.add_refname(ref)
        return self

    @classmethod
    def hash_refnames(cls, refs: Iterable[str]) -> bytes:
        return cls().add_refnames(refs).digest()

    def digest(self) -> bytes:
        return self.hasher.digest()

    @staticmethod
    def verify_refhash(
        refhash: bytes,
        target_refhash: bytes,
        observed_name: Optional[str],
        target_name: Optional[str],
        identifiers: Optional[tuple[Iterable[str], Iterable[str]]],
    ) -> None:
        if refhash == target_refhash:
            return None

        obs_name = "observed" if observed_name is None else observed_name
        tgt_name = "target" if target_name is None else target_name

        message = (
            f"Mismatch between sequence identifiers (names) in {obs_name} and {tgt_name}.\n"
            f"Observed {obs_name} identifier hash: {refhash.hex()}.\n"
            f"Expected {tgt_name} identifier hash: {target_refhash.hex()}\n"
            f"Make sure all identifiers in {obs_name} and {tgt_name} are identical "
            "and in the same order. "
            "Note that the identifier is the header before any whitespace."
        )

        if identifiers is not None:
            (observed_ids, target_ids) = identifiers
            for i, (observed_id, target_id) in enumerate(
                zip_longest(observed_ids, target_ids)
            ):
                if observed_id is None:
                    message += (
                        f"\nIdentifier mismatch: {obs_name} has only "
                        f"{i} identifier(s), which is fewer than {tgt_name}"
                    )
                    log_and_error(ValueError, message)
                elif target_id is None:
                    message += (
                        f"\nIdentifier mismatch: {tgt_name} has only "
                        f"{i} identifier(s), which is fewer than {obs_name}"
                    )
                    log_and_error(ValueError, message)
                elif observed_id != target_id:
                    message += (
                        f"\nIdentifier mismatch: Identifier number {i + 1} does not match "
                        f"between {obs_name} and {tgt_name}:"
                        f'{obs_name}: "{observed_id}"'
                        f'{tgt_name}: "{target_id}"'
                    )
                    log_and_error(ValueError, message)
            assert False
        else:
            log_and_error(ValueError, message)


def write_clusters(
    io: IO[str],
    clusters: Iterable[tuple[str, set[str]]],
) -> tuple[int, int]:
    n_clusters = 0
    n_contigs = 0
    print(CLUSTERS_HEADER, file=io)
    for cluster_name, contig_names in clusters:
        n_clusters += 1
        n_contigs += len(contig_names)
        for contig_name in contig_names:
            print(cluster_name, contig_name, sep="\t", file=io)

    return (n_clusters, n_contigs)


def read_clusters(filehandle: Iterable[str], min_size: int = 1) -> dict[str, set[str]]:
    """Read clusters from a file as created by function `writeclusters`.

    Inputs:
        filehandle: An open filehandle that can be read from
        min_size: Minimum number of contigs in cluster to be kept

    Output: A {clustername: set(contigs)} dict"""

    contigsof: _collections.defaultdict[str, set[str]] = _collections.defaultdict(set)
    lines = iter(filehandle)
    header = next(lines)
    if header.rstrip(" \n") != CLUSTERS_HEADER:
        raise ValueError(
            f'Expected cluster TSV file to start with header: "{CLUSTERS_HEADER}"'
        )

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped[0] == "#":
            continue

        clustername, contigname = stripped.split("\t")
        contigsof[clustername].add(contigname)

    contigsof_dict = {cl: co for cl, co in contigsof.items() if len(co) >= min_size}

    return contigsof_dict


def check_is_creatable_file_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise NotADirectoryError(path.parent)


def create_dir_if_not_existing(path: Path) -> None:
    if path.is_dir():
        return None
    if path.is_file():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise NotADirectoryError(path.parent)
    path.mkdir(exist_ok=True)


def write_bins(
    directory: Path,
    bins: dict[str, set[str]],
    fastaio: Iterable[bytes],
    maxbins: Optional[int] = 1000,
):
    """Writes bins as FASTA files in a directory, one file per bin.

    Inputs:
        directory: Directory to create or put files in
        bins: dict[str: set[str]] (can be loaded from clusters.tsv using vamb.cluster.read_clusters)
        fastaio: bytes iterator containing FASTA file with all sequences
        maxbins: None or else raise an error if trying to make more bins than this [1000]
    Output: None
    """

    # Safety measure so someone doesn't accidentally make 50000 tiny bins
    # If you do this on a compute cluster it can grind the entire cluster to
    # a halt and piss people off like you wouldn't believe.
    if maxbins is not None and len(bins) > maxbins:
        raise ValueError(f"{len(bins)} bins exceed maxbins of {maxbins}")

    create_dir_if_not_existing(directory)

    keep: set[str] = set()
    for i in bins.values():
        keep.update(i)

    bytes_by_id: dict[str, bytes] = dict()
    for entry in byte_iterfasta(fastaio, None):
        if entry.identifier in keep:
            bytes_by_id[entry.identifier] = _gzip.compress(
                entry.format().encode(), compresslevel=1
            )

    # Now actually print all the contigs to files
    for binname, contigs in bins.items():
        for contig in contigs:
            byts = bytes_by_id.get(contig)
            if byts is None:
                raise IndexError(
                    f'Contig "{contig}" in bin missing from input FASTA file'
                )

        # Print bin to file
        with open(directory.joinpath(binname + ".fna"), "wb") as file:
            for contig in contigs:
                file.write(_gzip.decompress(bytes_by_id[contig]))
                file.write(b"\n")


def validate_input_array(array: _np.ndarray) -> _np.ndarray:
    "Returns array similar to input array but C-contiguous and with own data."
    if not array.flags["C_CONTIGUOUS"]:
        array = _np.ascontiguousarray(array)
    if not array.flags["OWNDATA"]:
        array = array.copy()

    assert array.flags["C_CONTIGUOUS"] and array.flags["OWNDATA"]
    return array


def read_npz(file) -> _np.ndarray:
    """Loads array in .npz-format

    Input: Open file or path to file with npz-formatted array

    Output: A Numpy array
    """

    npz = _np.load(file)
    array = validate_input_array(npz["arr_0"])
    npz.close()

    return array


def write_npz(file, array: _np.ndarray):
    """Writes a Numpy array to an open file or path in .npz format

    Inputs:
        file: Open file or path to file
        array: Numpy array

    Output: None
    """
    _np.savez_compressed(file, array)


def concatenate_fasta_ios(
    outfile: IO[str],
    readers: Iterable[Iterable[bytes]],
    minlength: int = 2000,
    rename: bool = True,
):
    """Creates a new FASTA file from input paths, and optionally rename contig headers
    to the pattern "S{sample number}C{contig identifier}".

    Inputs:
        outpath: Open filehandle for output file
        readers: Iterable of iterable of bytes to read from, representing FASTA sequences
        minlength: Minimum contig length to keep [2000]
        rename: Rename headers

    Output: None
    """

    identifiers: set[str] = set()
    for reader_no, reader in enumerate(readers):
        # If we rename, seq identifiers only have to be unique for each sample
        if rename:
            identifiers.clear()

        for entry in byte_iterfasta(reader, None):
            if len(entry) < minlength:
                continue

            if rename:
                entry.rename(f"S{reader_no + 1}C{entry.identifier}".encode())

            if entry.identifier in identifiers:
                raise ValueError(
                    "Multiple sequences would be given "
                    f'identifier "{entry.identifier}".'
                )
            identifiers.add(entry.identifier)
            print(entry.format(), file=outfile)


def concatenate_fasta(
    outfile: IO[str],
    inpaths: Iterable[Path],
    minlength: int = 2000,
    rename: bool = True,
):
    concatenate_fasta_ios(
        outfile, open_file_iterator(inpaths), minlength=minlength, rename=rename
    )


def open_file_iterator(paths: Iterable[Path]) -> Iterable[Reader]:
    for path in paths:
        with Reader(path) as io:
            yield io
