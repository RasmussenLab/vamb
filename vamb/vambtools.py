__doc__ = "Various classes and functions Vamb uses internally."

import sys
import os as _os
import gzip as _gzip
import bz2 as _bz2
import lzma as _lzma
import numpy as _np
import re as _re
from vamb._vambtools import _kmercounts, _overwrite_matrix
import collections as _collections
from hashlib import md5 as _md5
from collections.abc import Iterable, Iterator, Generator
from typing import Optional, IO, Union
from pathlib import PurePath as _PurePath


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
        raise _np.AxisError(str(axis))

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

    uints = _np.frombuffer(mask, dtype=_np.uint8)
    index = _overwrite_matrix(array, uints)
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
    np_mask = _np.frombuffer(mask.numpy(), dtype=_np.uint8)
    index = _overwrite_matrix(np_array, np_mask)
    array.resize_((index, array.shape[1]))
    return array


class Reader:
    """Use this instead of `open` to open files which are either plain text,
    gzipped, bzip2'd or zipped with LZMA.

    Usage:
    >>> with Reader(file, readmode) as file: # by default textmode
    >>>     print(next(file))
    TEST LINE
    """

    def __init__(self, filename: str):
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
                '>([0-9A-Za-z!$%&+./:;?@^_|~-][0-9A-Za-z!#$%&*+./:;=?@^_|~-]*)([^\\S\\r\\n][^\\r\\n]*)?$"'
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
                f"Non-IUPAC DNA/RNA byte in sequence {identifier}: '{bad_character}', byte value {codeunit}"
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

    def kmercounts(self, k: int) -> _np.ndarray:
        if k < 1 or k > 10:
            raise ValueError("k must be between 1 and 10 inclusive")

        counts = _np.zeros(1 << (2 * k), dtype=_np.int32)
        _kmercounts(self.sequence, k, counts)
        return counts


def byte_iterfasta(
    filehandle: Iterable[bytes], comment: bytes = b"#"
) -> Iterator[FastaEntry]:
    """Yields FastaEntries from a binary opened fasta file.

    Usage:
    >>> with Reader('/dir/fasta.fna') as filehandle:
    ...     entries = byte_iterfasta(filehandle) # a generator

    Inputs:
        filehandle: Any iterator of binary lines of a FASTA file
        comment: Ignore lines beginning with any whitespace + comment

    Output: Generator of FastaEntry-objects from file
    """

    # Make it work for persistent iterators, e.g. lists
    line_iterator = iter(filehandle)
    # Skip to first header
    try:
        for probeline in line_iterator:
            stripped = probeline.lstrip()
            if stripped.startswith(comment):
                pass

            elif probeline[0:1] == b">":
                break

            else:
                raise ValueError("First non-comment line is not a Fasta header")

        else:  # no break
            return None
            # raise ValueError('Empty or outcommented file')

    except TypeError:
        errormsg = (
            "First line does not contain bytes. Are you reading file in binary mode?"
        )
        raise TypeError(errormsg) from None

    header = probeline[1:]
    buffer: list[bytes] = list()

    # Iterate over lines
    for line in line_iterator:
        if line.startswith(comment):
            pass

        elif line.startswith(b">"):
            yield FastaEntry(header, bytearray().join(buffer))
            buffer.clear()
            header = line[1:]

        else:
            buffer.append(line)

    yield FastaEntry(header, bytearray().join(buffer))


def write_clusters(
    filehandle: IO[str],
    clusters: Iterable[tuple[str, set[str]]],
    max_clusters: Optional[int] = None,
    min_size: int = 1,
    header: Optional[str] = None,
    rename: bool = True,
    cluster_prefix= str,
) -> tuple[int, int]:
    """Writes clusters to an open filehandle.
    Inputs:
        filehandle: An open filehandle that can be written to
        clusters: An iterator generated by function `clusters` or a dict
        max_clusters: Stop printing after this many clusters [None]
        min_size: Don't output clusters smaller than N contigs
        header: Commented one-line header to add
        rename: Rename clusters to "cluster_1", "cluster_2" etc.
        cluster_prefix: prepend a tag to identify which model produced the clusters (vae,aae_l, aae_y)

    Outputs:
        clusternumber: Number of clusters written
        ncontigs: Number of contigs written
    """

    if not hasattr(filehandle, "writable") or not filehandle.writable():
        raise ValueError("Filehandle must be a writable file")

    if max_clusters is not None and max_clusters < 1:
        raise ValueError("max_clusters must None or at least 1, not {max_clusters}")

    if header is not None and len(header) > 0:
        if "\n" in header:
            raise ValueError("Header cannot contain newline")

        if header[0] != "#":
            header = "# " + header

        print(header, file=filehandle)

    clusternumber = 0
    ncontigs = 0

    for clustername, contigs in clusters:
        if len(contigs) < min_size:
            continue

        if rename:
            clustername = cluster_prefix + "cluster_" + str(clusternumber + 1)
        else:
            clustername = cluster_prefix +  str(clusternumber + 1)

        for contig in contigs:
            print(clustername, contig, sep="\t", file=filehandle)
        filehandle.flush()

        clusternumber += 1
        ncontigs += len(contigs)

        if clusternumber == max_clusters:
            break

    return clusternumber, ncontigs


def read_clusters(filehandle: Iterable[str], min_size: int =1) -> dict[str, set[str]]:
    """Read clusters from a file as created by function `writeclusters`.

    Inputs:
        filehandle: An open filehandle that can be read from
        min_size: Minimum number of contigs in cluster to be kept

    Output: A {clustername: set(contigs)} dict"""

    contigsof: _collections.defaultdict[str, set[str]] = _collections.defaultdict(set)

    for line in filehandle:
        stripped = line.strip()

        if not stripped or stripped[0] == "#":
            continue

        clustername, contigname = stripped.split("\t")
        contigsof[clustername].add(contigname)

    contigsof_dict = {cl: co for cl, co in contigsof.items() if len(co) >= min_size}

    return contigsof_dict


def write_bins(
    directory: Union[str, _PurePath],
    bins: dict[str, set[str]],
    fastaio: Iterable[bytes],
    maxbins: Optional[int] = 250,
    minsize: int = 0,
    separator: str = None,
):
    """Writes bins as FASTA files in a directory, one file per bin.

    Inputs:
        directory: Directory to create or put files in
        bins: dict[str: set[str]] (can be loaded from
        clusters.tsv using vamb.cluster.read_clusters)
        fastaio: bytes iterator containing FASTA file with all sequences
        maxbins: None or else raise an error if trying to make more bins than this [250]
        minsize: Minimum number of nucleotides in cluster to be output [0]
        separator: string that separates the contig/cluster name from the sample ; i.e. sample_id_separator_contig_name/cluster_name 
    Output: None
    """

    # Safety measure so someone doesn't accidentally make 50000 tiny bins
    # If you do this on a compute cluster it can grind the entire cluster to
    # a halt and piss people off like you wouldn't believe.
    if maxbins is not None and len(bins) > maxbins:
        raise ValueError(f"{len(bins)} bins exceed maxbins of {maxbins}")

    # Check that the directory is not a non-directory file,
    # and that its parent directory indeed exists
    abspath = _os.path.abspath(directory)
    parentdir = _os.path.dirname(abspath)

    if parentdir != "" and not _os.path.isdir(parentdir):
        raise NotADirectoryError(parentdir)

    if _os.path.isfile(abspath):
        raise FileExistsError(abspath)

    if minsize < 0:
        raise ValueError("Minsize must be nonnegative")

    byteslen_by_id: dict[str, tuple[bytes, int]] = dict()
    for entry in byte_iterfasta(fastaio):
        byteslen_by_id[entry.identifier] = (
            _gzip.compress(entry.format().encode(), compresslevel=1),
            len(entry),
        )

    # Make the directory if it does not exist - if it does, do nothing
    try:
        _os.mkdir(directory)
    except FileExistsError:
        pass
    except:
        raise

    # Now actually print all the contigs to files
    for binname, contigs in bins.items():
        size = 0
        if separator is not None:
            binsample=next(iter(contigs)).split(separator)[0]
        for contig in contigs:
            byteslen = byteslen_by_id.get(contig)
            if byteslen is None:
                raise IndexError(
                    f'Contig "{contig}" in bin missing from input FASTA file'
                )
            size += byteslen[1]

        if size < minsize:
            continue
        # added by pau to split bin files into sample dirs
        if separator is not None:
       
            bin_dir=_os.path.join(directory,binsample)
            try:
                _os.mkdir(bin_dir)
            except FileExistsError:
                pass
            except:
                raise
        else:
            bin_dir=directory
        # Print bin to file
        filename = _os.path.join(bin_dir, binname + ".fna")

        #filename = _os.path.join(directory, binname + ".fna")
        with open(filename, "wb") as file:
            for contig in contigs:
                file.write(_gzip.decompress(byteslen_by_id[contig][0]))
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


def concatenate_fasta(
    outfile: IO[str], inpaths: Iterable[str], minlength: int = 2000, rename: bool = True
):
    """Creates a new FASTA file from input paths, and optionally rename contig headers
    to the pattern "S{sample number}C{contig identifier}".

    Inputs:
        outpath: Open filehandle for output file
        inpaths: Iterable of paths to FASTA files to read from
        minlength: Minimum contig length to keep [2000]
        rename: Rename headers

    Output: None
    """

    identifiers: set[str] = set()
    for (inpathno, inpath) in enumerate(inpaths):
        try:
            with Reader(inpath) as infile:

                # If we rename, seq identifiers only have to be unique for each sample
                if rename:
                    identifiers.clear()

                for entry in byte_iterfasta(infile):
                    if len(entry) < minlength:
                        continue

                    if rename:
                        entry.rename(f"S{inpathno + 1}C{entry.identifier}".encode())

                    if entry.identifier in identifiers:
                        raise ValueError(
                            "Multiple sequences would be given "
                            f'identifier "{entry.identifier}".'
                        )
                    identifiers.add(entry.identifier)
                    print(entry.format(), file=outfile)
        except Exception as e:
            print(f"Exception occured when parsing file {inpath}", file=sys.stderr)
            raise e from None


def hash_refnames(refnames: Iterable[str]) -> bytes:
    "Hashes an iterable of strings of reference names using MD5."
    hasher = _md5()
    for refname in refnames:
        hasher.update(refname.encode().rstrip())

    return hasher.digest()


def _split_bin(
    binname: str,
    headers: Iterable[str],
    separator: str,
    bysample: _collections.defaultdict[str, set[str]] = _collections.defaultdict(set),
) -> Generator[tuple[str, set[str]], None, None]:
    "Split a single bin by the prefix of the headers"

    bysample.clear()
    for header in headers:
        if not isinstance(header, str):  # type: ignore
            raise TypeError(
                f"Can only split named sequences, not of type {type(header)}"
            )

        sample, _, identifier = header.partition(separator)

        if not identifier:
            raise KeyError(f"Separator '{separator}' not in sequence label: '{header}'")

        bysample[sample].add(header)

    for sample, splitheaders in bysample.items():
        newbinname = f"{sample}{separator}{binname}"
        yield newbinname, splitheaders


def binsplit(
    clusters: Iterable[tuple[str, Iterable[str]]], separator: str
) -> Generator[tuple[str, set[str]], None, None]:
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
    if isinstance(clusters, dict): # added by Pau since y clusters are in a dictionary object
        clusters = clusters.items()
 
    for binname, headers in clusters:
        for newbinname, splitheaders in _split_bin(binname, headers, separator):
            yield newbinname, splitheaders
