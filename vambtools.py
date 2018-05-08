
import gzip as _gzip
import numpy as _np

if __package__ is None or __package__ == '':
    from vambtools_c import reverse_complement_kmer, _kmercounts, _fourmerfreq, _freq_432mers
    
else:
    from vamb.vambtools_c import reverse_complement_kmer, _kmercounts, _fourmerfreq, _freq_432mers



def zscore(array, axis=None, inplace=False):
    "Calculates zscore for an array. A cheap copy of scipy.stats.zscore."
    
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



class Reader:
    "Use this instead of `open` for files which may be gzipped or not."
    
    def __init__(self, filename, readmode='r'):
        if readmode not in ('r', 'rb'):
            raise ValueError("the reader cannot write, set mode to 'r' or 'rb'")
        
        self.filename = filename
        self.readmode = readmode
    
    def __enter__(self):
        with open(self.filename, 'rb') as f:
            signature = f.peek(2)[:2]
        
        # Gzipped files begin with the two bytes 1F8B
        if tuple(signature) == (31, 139):
            if self.readmode == 'r':
                self.filehandle = _gzip.open(self.filename, 'rt')
                
            else:
                self.filehandle = _gzip.open(self.filename, self.readmode)
                
        else:
            self.filehandle = open(self.filename, self.readmode)
            
        return self.filehandle
    
    def __exit__(self, type, value, traceback):
        self.filehandle.close()



class FastaEntry:
    """One single FASTA entry"""
    
    __slots__ = ['header', 'sequence']
    
    def __init__(self, header, sequence):
        if not header or not sequence:
            raise ValueError('Header and sequence must be nonempty')
            
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
    
    # Two entries with same header cannot co-exist in same set/dict!
    def __hash__(self):
        return hash(self.header)
    
    def __contains__(self, other):
        if isinstance(other, str):
            return other.encode() in self.sequence
        
        elif isinstance(other, bytes) or isinstance(other, bytearray):
            return other in self.sequence
        
        else:
            raise TypeError('Can only compare to str, bytes or bytearray')
    
    # Entries are compared equal by their sequence.
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.sequence == other.sequence
        else:
            raise TypeError('Cannot compare to object of other class')
        
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
    
    def freq_432mers(self):
        return _freq_432mers(self.sequence)



def byte_iterfasta(filehandle):
    "Yields FastaEntries from a binary opened fasta file."

    # Skip to first header
    try:
        for probeline in filehandle:
            if probeline.startswith(b'>'):
                break
                
        else: # nobreak
            raise TypeError('No headers in this file.')
            
    except TypeError:
        errormsg = 'First line does not contain bytes. Are you reading file in binary mode?'
        raise TypeError(errormsg) from None

    header = probeline.strip(b'>\n')
    buffer = list()

    # Iterate over lines
    for linenumber, line in enumerate(filehandle):
        if line.startswith(b'>'):
            yield FastaEntry(header, b''.join(buffer))
            buffer.clear()
            header = line[1:-1].decode()

        else:
            upper = line.upper()[:-1]
            stripped = upper.translate(None, delete=b'ACGTN')
            
            if len(stripped) > 0:
                raise ValueError('Non-ACGTN in line {}: {}'.format(linenumber, stripped[0]))
            
            buffer.append(upper)

    yield FastaEntry(header, b''.join(buffer))

