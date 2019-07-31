import os as _os
import gzip as _gzip
import bz2 as _bz2
import lzma as _lzma
import numpy as _np
from vamb._vambtools import _kmercounts, _fourmerfreq, zeros, _overwrite_matrix
import collections as _collections

TNF_HEADER = '#contigheader\t' + '\t'.join([
'AAAA/TTTT', 'AAAC/GTTT', 'AAAG/CTTT', 'AAAT/ATTT',
'AACA/TGTT', 'AACC/GGTT', 'AACG/CGTT', 'AACT/AGTT',
'AAGA/TCTT', 'AAGC/GCTT', 'AAGG/CCTT', 'AAGT/ACTT',
'AATA/TATT', 'AATC/GATT', 'AATG/CATT', 'AATT',
'ACAA/TTGT', 'ACAC/GTGT', 'ACAG/CTGT', 'ACAT/ATGT',
'ACCA/TGGT', 'ACCC/GGGT', 'ACCG/CGGT', 'ACCT/AGGT',
'ACGA/TCGT', 'ACGC/GCGT', 'ACGG/CCGT', 'ACGT',
'ACTA/TAGT', 'ACTC/GAGT', 'ACTG/CAGT', 'AGAA/TTCT',
'AGAC/GTCT', 'AGAG/CTCT', 'AGAT/ATCT', 'AGCA/TGCT',
'AGCC/GGCT', 'AGCG/CGCT', 'AGCT',      'AGGA/TCCT',
'AGGC/GCCT', 'AGGG/CCCT', 'AGTA/TACT', 'AGTC/GACT',
'AGTG/CACT', 'ATAA/TTAT', 'ATAC/GTAT', 'ATAG/CTAT',
'ATAT',      'ATCA/TGAT', 'ATCC/GGAT', 'ATCG/CGAT',
'ATGA/TCAT', 'ATGC/GCAT', 'ATGG/CCAT', 'ATTA/TAAT',
'ATTC/GAAT', 'ATTG/CAAT', 'CAAA/TTTG', 'CAAC/GTTG',
'CAAG/CTTG', 'CACA/TGTG', 'CACC/GGTG', 'CACG/CGTG',
'CAGA/TCTG', 'CAGC/GCTG', 'CAGG/CCTG', 'CATA/TATG',
'CATC/GATG', 'CATG',      'CCAA/TTGG', 'CCAC/GTGG',
'CCAG/CTGG', 'CCCA/TGGG', 'CCCC/GGGG', 'CCCG/CGGG',
'CCGA/TCGG', 'CCGC/GCGG', 'CCGG',      'CCTA/TAGG',
'CCTC/GAGG', 'CGAA/TTCG', 'CGAC/GTCG', 'CGAG/CTCG',
'CGCA/TGCG', 'CGCC/GGCG', 'CGCG',      'CGGA/TCCG',
'CGGC/GCCG', 'CGTA/TACG', 'CGTC/GACG', 'CTAA/TTAG',
'CTAC/GTAG', 'CTAG',      'CTCA/TGAG', 'CTCC/GGAG',
'CTGA/TCAG', 'CTGC/GCAG', 'CTTA/TAAG', 'CTTC/GAAG',
'GAAA/TTTC', 'GAAC/GTTC', 'GACA/TGTC', 'GACC/GGTC',
'GAGA/TCTC', 'GAGC/GCTC', 'GATA/TATC', 'GATC',
'GCAA/TTGC', 'GCAC/GTGC', 'GCCA/TGGC', 'GCCC/GGGC',
'GCGA/TCGC', 'GCGC',      'GCTA/TAGC', 'GGAA/TTCC',
'GGAC/GTCC', 'GGCA/TGCC', 'GGCC', 'GGGA/TCCC',
'GGTA/TACC', 'GTAA/TTAC', 'GTAC', 'GTCA/TGAC',
'GTGA/TCAC', 'GTTA/TAAC', 'TAAA/TTTA', 'TACA/TGTA',
'TAGA/TCTA', 'TATA',      'TCAA/TTGA', 'TCCA/TGGA',
'TCGA',      'TGAA/TTCA', 'TGCA', 'TTAA']) + '\n'

def zscore(array, axis=None, inplace=False):
    """Calculates zscore for an array. A cheap copy of scipy.stats.zscore.

    Inputs:
        array: Numpy array to be normalized
        axis: Axis to operate across [None = entrie array]
        inplace: Do not create new array, change input array [False]

    Output:
        If inplace is True: None
        else: New normalized Numpy-array"""

    if axis is not None and axis >= array.ndim:
        raise _np.AxisError('array only has {} axes'.format(array.ndim))

    if inplace and array.dtype not in (_np.float, _np.float16, _np.float32, _np.float64, _np.float128):
        raise TypeError('Cannot convert a non-float array to zscores')

    mean = array.mean(axis=axis)
    std = array.std(axis=axis)

    if axis is None:
        if std == 0:
            std = 1 # prevent divide by zero

    else:
        std[std == 0.0] = 1 # prevent divide by zero
        shape = tuple(dim if ax != axis else 1 for ax, dim in enumerate(array.shape))
        mean.shape, std.shape = shape, shape

    if inplace:
        array -= mean
        array /= std
        return None
    else:
        return (array - mean) / std

def inplace_maskarray(array, mask):
    """In-place masking of an array, i.e. if `mask` is a boolean mask of same
    length as `array`, then array[mask] == inplace_maskarray(array, mask),
    but does not allocate a new array.
    """

    if len(mask) != len(array):
        raise ValueError('Lengths of array and mask must match')
    elif array.ndim != 2:
        raise ValueError('Can only take a 2 dimensional-array.')

    # Cython doesn't support bool arrays, so this does a no-copy type casting.
    uints = _np.frombuffer(mask, dtype=_np.uint8)
    index = _overwrite_matrix(array, uints)
    array.resize((index, array.shape[1]), refcheck=False)
    return array

class Reader:
    """Use this instead of `open` to open files which are either plain text,
    gzipped, bzip2'd or zipped with LZMA.

    Usage:
    >>> with Reader(file, readmode) as file: # by default textmode
    >>>     print(next(file))
    TEST LINE
    """

    def __init__(self, filename, readmode='r'):
        if readmode not in ('r', 'rb'):
            raise ValueError("the Reader cannot write, set mode to 'r' or 'rb'")

        self.filename = filename
        self.readmode = readmode

    def __enter__(self):
        with open(self.filename, 'rb') as f:
            signature = f.peek(8)[:8]

        # Gzipped files begin with the two bytes 0x1F8B
        if tuple(signature[:2]) == (0x1F, 0x8B):
            if self.readmode == 'r':
                self.filehandle = _gzip.open(self.filename, 'rt')
            else:
                self.filehandle = _gzip.open(self.filename, self.readmode)
        # bzip2 files begin with the signature BZ
        elif signature[:2] == b'BZ':
            if self.readmode == 'r':
                self.filehandle = _bz2.open(self.filename, 'rt')
            else:
                self.filehandle = _bz2.open(self.filename, self.readmode)
        # .XZ files begins with 0xFD377A585A0000
        elif tuple(signature[:7]) == (0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00):
            if self.readmode == 'r':
                self.filehandle = _lzma.open(self.filename, 'rt')
            else:
                self.filehandle = _lzma.open(self.filename, self.readmode)
        # Else we assume it's a text file.
        else:
            self.filehandle = open(self.filename, self.readmode)

        return self.filehandle

    def __exit__(self, type, value, traceback):
        self.filehandle.close()

class FastaEntry:
    """One single FASTA entry"""

    __slots__ = ['header', 'sequence']

    def __init__(self, header, sequence):
        if header[0] in ('>', '#') or header[0].isspace():
            raise ValueError('Header cannot begin with #, > or whitespace')
        if '\t' in header:
            raise ValueError('Header cannot contain a tab')

        self.header = header
        self.sequence = bytearray(sequence)

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return '>{}\n{}'.format(self.header, self.sequence.decode())

    def format(self, width=60):
        sixtymers = range(0, len(self.sequence), width)
        spacedseq = '\n'.join([self.sequence[i: i+width].decode() for i in sixtymers])
        return '>{}\n{}'.format(self.header, spacedseq)

    def __getitem__(self, index):
        return self.sequence[index]

    def __repr__(self):
        return '<FastaEntry {}>'.format(self.header)

    def kmercounts(self, k):
        if k < 1 or k > 10:
            raise ValueError('k must be between 1 and 10 inclusive')
        return _kmercounts(self.sequence, k)

    def fourmer_freq(self):
        return _fourmerfreq(self.sequence)

def byte_iterfasta(filehandle, comment=b'#'):
    """Yields FastaEntries from a binary opened fasta file.

    Usage:
    >>> with Reader('/dir/fasta.fna', 'rb') as filehandle:
    ...     entries = byte_iterfasta(filehandle) # a generator

    Inputs:
        filehandle: Any iterator of binary lines of a FASTA file
        comment: Ignore lines beginning with any whitespace + comment

    Output: Generator of FastaEntry-objects from file
    """

    linemask = bytes.maketrans(b'acgtuUswkmyrbdhvnSWKMYRBDHV',
                               b'ACGTTTNNNNNNNNNNNNNNNNNNNNN')

    # Skip to first header
    try:
        for linenumber, probeline in enumerate(filehandle):
            stripped = probeline.lstrip()
            if stripped.startswith(comment):
                pass

            elif probeline[0:1] == b'>':
                break

            else:
                raise ValueError('First non-comment line is not a Fasta header')

        else: # no break
            raise ValueError('Empty or outcommented file')

    except TypeError:
        errormsg = 'First line does not contain bytes. Are you reading file in binary mode?'
        raise TypeError(errormsg) from None

    header = probeline.strip(b'>\n').decode()
    buffer = list()

    # Iterate over lines
    for line in filehandle:
        linenumber += 1

        if line.startswith(comment):
            continue

        if line.startswith(b'>'):
            yield FastaEntry(header, b''.join(buffer))
            buffer.clear()
            header = line[1:-1].decode()

        else:
            # Check for un-parsable characters in the sequence
            stripped = line.translate(None, b'acgtuACGTUswkmyrbdhvnSWKMYRBDHVN \t\n')
            if len(stripped) > 0:
                bad_character = chr(stripped[0])
                raise ValueError("Non-IUPAC DNA in line {}: '{}'".format(linenumber + 1,
                                                                         bad_character))
            masked = line.translate(linemask, b' \t\n')
            buffer.append(masked)

    yield FastaEntry(header, b''.join(buffer))

def loadfasta(byte_iterator, keep=None, comment=b'#'):
    """Loads a FASTA file into a dictionary.

    Usage:
    >>> with Reader('/dir/fasta.fna', 'rb') as filehandle:
    ...     fastadict = loadfasta(filehandle)

    Input:
        byte_iterator: Iterator of binary lines of FASTA file
        keep: Keep entries with headers in `keep`. If None, keep all entries
        comment: Ignore lines beginning with any whitespace + comment

    Output: {header: FastaEntry} dict
    """

    entries = dict()

    for entry in byte_iterfasta(byte_iterator, comment=comment):
        if keep is None or entry.header in keep:
            entries[entry.header] = entry

    return entries

def write_bins(directory, bins, fastadict, maxbins=250):
    """Writes bins as FASTA files in a directory, one file per bin.

    Inputs:
        directory: Directory to create or put files in
        bins: {'name': {set of contignames}} dictionary (can be loaded from
        clusters.tsv using vamb.cluster.read_clusters)
        fastadict: {contigname: FastaEntry} dict as made by `loadfasta`
        maxbins: None or else raise an error if trying to make more bins than this

    Output: None
    """

    # Safety measure so someone doesn't accidentally make 50000 tiny bins
    # If you do this on a compute cluster it can grind the entire cluster to
    # a halt and piss people off like you wouldn't believe.
    if maxbins is not None and len(bins) > maxbins:
        raise ValueError('{} bins exceed maxbins of {}'.format(len(bins), maxbins))

    # Check that the directory is not a non-directory file,
    # and that its parent directory indeed exists
    abspath = _os.path.abspath(directory)
    parentdir = _os.path.dirname(abspath)

    if parentdir != '' and not _os.path.isdir(parentdir):
        raise NotADirectoryError(parentdir)

    if _os.path.isfile(abspath):
        raise NotADirectoryError(abspath)

    # Check that all contigs in all bins are in the fastadict
    allcontigs = set()

    for contigs in bins.values():
        allcontigs.update(set(contigs))

    allcontigs -= fastadict.keys()
    if allcontigs:
        nmissing = len(allcontigs)
        raise IndexError('{} contigs in bins missing from fastadict'.format(nmissing))

    # Make the directory if it does not exist - if it does, do nothing
    try:
        _os.mkdir(directory)
    except FileExistsError:
        pass
    except:
        raise

    # Now actually print all the contigs to files
    for binname, contigs in bins.items():
        filename = _os.path.join(directory, binname + '.fna')

        with open(filename, 'w') as file:
            for contig in contigs:
                entry = fastadict[contig]
                print(entry.format(), file=file)

def read_npz(file):
    """Loads array in .npz-format

    Input: Open file or path to file with npz-formatted array

    Output: A Numpy array
    """

    npz = _np.load(file)
    array = npz['arr_0']
    npz.close()

    return array

def write_npz(file, array):
    """Writes a Numpy array to an open file or path in .npz format

    Inputs:
        file: Open file or path to file
        array: Numpy array

    Output: None
    """
    _np.savez_compressed(file, array)

def filtercontigs(infile, outfile, minlength=2000):
    """Creates new FASTA file with filtered contigs

    Inputs:
        infile: Binary opened input FASTA file
        outfile: Write-opened output FASTA file
        minlength: Minimum contig length to keep [2000]

    Output: None
    """

    fasta_entries = _vambtools.byte_iterfasta(infile)

    for entry in fasta_entries:
        if len(entry) > minlength:
            print(entry.format(), file=outfile)

def load_jgi(filehandle):
    """Load depths from the --outputDepth of jgi_summarize_bam_contig_depths.
    See https://bitbucket.org/berkeleylab/metabat for more info on that program.

    Usage:
        with open('/path/to/jgi_depths.tsv') as file:
            depths = load_jgi(file)
    Input:
        File handle of open output depth file
    Output:
        N_contigs x N_samples Numpy matrix of dtype float32
    """

    header = next(filehandle)
    fields = header.split('\t')
    if not fields[4].endswith('-var'):
        raise ValueError('Input file format error: 5th column should contain variances.')

    columns = tuple(range(3, len(fields), 2))
    return _np.loadtxt(filehandle, dtype=_np.float32, usecols=columns)

def _split_bin(binname, headers, separator, bysample=_collections.defaultdict(set)):
    "Split a single bin by the prefix of the headers"

    bysample.clear()
    for header in headers:
        if not isinstance(header, str):
            raise TypeError('Can only split named sequences, not of type {}'.format(type(header)))

        sample, sep, identifier = header.partition(separator)

        if not identifier:
            raise KeyError("Separator '{}' not in sequence label: '{}'".format(separator, header))

        bysample[sample].add(header)

    for sample, splitheaders in bysample.items():
        newbinname = "{}{}{}".format(sample, separator, binname)
        yield newbinname, splitheaders

def _binsplit_generator(cluster_iterator, separator):
    "Return a generator over split bins with the function above."
    for binname, headers in cluster_iterator:
        for newbinname, splitheaders in _split_bin(binname, headers, separator):
            yield newbinname, splitheaders

def binsplit(clusters, separator):
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
    if iter(clusters) is clusters: # clusters is an iterator
        return _binsplit_generator(clusters, separator)

    elif isinstance(clusters, dict):
        return dict(_binsplit_generator(clusters.items(), separator))

    else:
        raise TypeError("clusters must be iterator of pairs or dict")
