
__doc__ = """Run the entire Vamb pipiline.

Usage:
Given a path to contigs and a list/tuple of paths to bamfiles
>>> run(outdir, contigpath, bampaths, **kwargs)

Creates a new directory and runs each module of the Vamb pipeline in the
new directory. Does not yet support resuming stopped runs - in order to do so,
you must run each module independently from commandline or from Python.

See the tutorial at https://github.com/jakobnissen/vamb
"""

__cmd_doc__ = """Run the Vamb pipeline.

Creates a new direcotry and runs each module of the Vamb pipeline in the
new directory. Does not yet support resuming stopped runs - in order to do so,
you must run each module independently from commandline or from Python
"""



# Import all Vamb modules
import sys as _sys
import os as _os
import numpy as _np
import torch as _torch

if __package__ is None or __package__ == '':
    import vambtools as _vambtools
    import parsecontigs as _parsecontigs
    import parsebam as _parsebam
    import encode as _encode
    import cluster as _cluster
    import threshold as _threshold
    
else:
    import vamb.vambtools as _vambtools
    import vamb.parsecontigs as _parsecontigs
    import vamb.parsebam as _parsebam
    import vamb.encode as _encode
    import vamb.cluster as _cluster
    import vamb.threshold as _threshold
    
if __name__ == '__main__':
    import argparse



DEFAULT_PROCESSES = min(8, _os.cpu_count())



def _checkpaths(outdir, contigspath, bampaths):
    # Check presence of outdir and its parent directory
    if _os.path.exists(outdir):
        raise FileExistsError(outdir)
    
    parentdir = _os.path.dirname(outdir)
    if parentdir and not _os.path.isdir(parentdir):
        raise NotADirectoryError(parentdir)
        
    # Check presence of contigs
    if not _os.path.isfile(contigspath):
        raise FileNotFoundError(contigspath)
        
    # Check presence of bamfiles
    for bampath in bampaths:
        if not _os.path.isfile(bampath):
            raise FileNotFoundError(bampath)



def run(outdir, contigspath, bampaths, minlength=100, minascore=50,
        readprocesses=DEFAULT_PROCESSES, nhiddens=[325, 325], nlatent=40,
       nepochs=300, batchsize=100, cuda=False, tnfweight=1.0, lrate=1e-4,
       verbose=True, tandemcluster=False, max_steps=15, nsamples=1000,
       min_clustersize=5):
    "placeholder docstring"
    
    ############## NOTE #############################
    #
    # Things to add/change:
    # - Needed to dump RPKM and reload it? What is mem requirement?
    # - Maybe dump trained weights?
    #
    #################################################
    
    _checkpaths(outdir, contigspath, bampaths)
    
    if verbose:
        print('Creating output directory', end='\n\n')
    _os.mkdir(outdir)
    
    # Get contignames, contiglengths and TNF
    if verbose:
        print('Computing TNF from contigs file', end='\n\n')
        
    with _vambtools.Reader(contigspath, 'rb') as contigsfile:
        tnfs, contignames, lengths = _parsecontigs.read_contigs(contigsfile, minlength=minlength)
    
    del lengths
    
    # Get RPKMS
    if verbose:
        print('Computing depths from BAM files', end='\n\n')
    
    sample_rpkms, contignames2 = _parsebam.read_bamfiles(bampaths, minscore=minascore,
                                                       minlength=minlength, processes=readprocesses)
    
    assert contignames == contignames2
    del contignames2
    contignames = _np.array(contignames)
    
    # Save RPKMs, delete and load into Numpy array to save memory (needed?)
    npzpath = _os.path.join(outdir, 'sample_rpkms.npz')
    
    _parsebam.write_samplerpkm(npzpath, sample_rpkms)
    del sample_rpkms
    
    rpkms = _parsebam.read_npz(npzpath, bampaths)
    
    # Construct VAE and train it
    if verbose:
        print('Making and training VAE on data.')
    
    vae, dataloader = _encode.trainvae(rpkms, tnfs, nhiddens=nhiddens, nlatent=nlatent,
                                      nepochs=nepochs, batchsize=batchsize, cuda=cuda,
                                      tnfweight=tnfweight, lrate=lrate, verbose=verbose)
    
    # Feed dataloader to VAE
    if verbose:
        print('\nEncoding data to latent representation', end='\n\n')
    
    latent = vae.encode(dataloader)
    
    del tnfs, rpkms
    
    _vambtools.zscore(latent, axis=1, inplace=True)
    
    # Cluster and write
    if verbose:
        print('Clustering latent representation', end='\n\n')
    
    if tandemcluster:
        clusters = _cluster.tandemcluster(latent, contignames,
                                          max_steps=max_steps, normalized=True)
        
    else:
        clusterit = _cluster.cluster(latent, contignames,
                                          max_steps=max_steps, normalized=True)
        clusters = dict()
        
        for medoid, cluster in clusterit:
            clusters[medoid] = cluster
    
    clusterpath = _os.path.join(outdir, 'clusters.tsv')
    
    with open(clusterpath, 'w') as clusterfile:
        _cluster.writeclusters(clusterfile, clusters, min_size=min_clustersize)
        
    if verbose:
        print('Done.')



run.__doc__ = """Run the entire Vamb pipeline.

Inputs:
    outdir: Directory to create and put outputs
    contigspath: Path to contigs FASTA file
    bampaths: Tuple or lists of paths to BAM files

    minlength: Ignore sequences with length below this [100]
    minascore: Ignore reads with alignment score below this [50]
    readprocesses: N processes to spawn reading BAMs [{}]
    nhiddens: List of number of neurons in hidden layers [325, 325]
    nlatent: Number of neurons in latent layer [40]
    nepochs: Number of training epochs [300]
    batchsize: Sequences per mini-batch when training [100]
    cuda: Use CUDA (GPU acceleration software) [False]
    tnfweight: Relative weight of TNF error when computing loss [1.0]
    lrate: Learning rate for the optimizer [1e-4]
    verbose: Print progress to STDOUT [True]
    tandemcluster: Use fast but inaccurate clustering [False]
    max_steps: Stop searching for optimal medoid after N futile attempts [15]
    nsamples: Number of samples to estimate clustering threshold [1000]
    min_clustersize: Ignore all clusters with fewer sequences than this [5]

Output: None, but creates directory with clusters.tsv file.
""".format(DEFAULT_PROCESSES)



if __name__ == '__main__':
    
    ################ Create parser ############################
    
    usage = "python runvamb.py OUTPATH FASTA BAMPATHS [OPTIONS ...]"
    parser = argparse.ArgumentParser(
        description=__cmd_doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=usage, add_help=False)
    
    # Help
    helpos = parser.add_argument_group(title='Help', description=None)
    helpos.add_argument('-h', '--help', help='print help and exit', action='help')
    
    # Positional arguments
    reqos = parser.add_argument_group(title='Required arguments', description=None)
    reqos.add_argument('outpath', help='output directory to create')
    reqos.add_argument('fasta', help='path to fasta file')
    reqos.add_argument('bamfiles', help='path to BAM files', nargs='+')
    
    # Optional arguments
    inputos = parser.add_argument_group(title='IO options options', description=None)
    
    inputos.add_argument('-m', dest='minlength', metavar='', type=int, default=[100],
                         help='ignore sequences shorter than this [100]')
    inputos.add_argument('-a', dest='minascore', metavar='', type=int, default=[50],
                         help='ignore reads with alignment score below this [50]')
    inputos.add_argument('-p', dest='processes', metavar='', type=int, default=DEFAULT_PROCESSES,
                         help=('reading processes to spawn '
                              '[min(' + str(DEFAULT_PROCESSES) + ', nbamfiles)]'))
    inputos.add_argument('--silent', help='Run silently [False]', action='store_false')
    
    vambos = parser.add_argument_group(title='Training options', description=None)
    
    vambos.add_argument('-n', dest='nhiddens', metavar='', type=int, nargs='+',
                        default=[325, 325], help='hidden neurons [325 325]')
    vambos.add_argument('-l', dest='nlatent', metavar='', type=int,
                        default=40, help='latent neurons [40]')
    vambos.add_argument('-e', dest='nepochs', metavar='', type=int,
                        default=300, help='epochs [300]')
    vambos.add_argument('-b', dest='batchsize', metavar='', type=int,
                        default=100, help='batch size [100]')
    vambos.add_argument('-t', dest='tnfweight',  metavar='',type=float,
                        default=1.0, help='TNF weight [1.0]')
    vambos.add_argument('-r', dest='lrate', metavar='', type=float,
                        default=0.0001, help='learning rate [1e-4]')
    vambos.add_argument('--cuda', help='use GPU [False]', action='store_true')
    
    clusto = parser.add_argument_group(title='Clustering options', description=None)
    
    clusto.add_argument('-x', dest='max_steps', metavar='', type=int,
                        default=15, help='maximum clustering steps [15]')
    clusto.add_argument('-s', dest='nsamples', metavar='', type=int, default=1000,
                        help='n_samples to determine cluster threshold [1000]')
    clusto.add_argument('--tandem', help='tandem cluster [False]', action='store_true')
    
    
    # If no arguments, print help
    if len(_sys.argv) == 1:
        parser.print_help()
        _sys.exit()
        
    args = parser.parse_args()
    
    # Check inputs
    
    _checkpaths(args.outpath, args.fasta, args.bamfiles)
    
    # Check IO options
    
    if args.minlength < 4:
        raise ValueError('Minlength must be at least 4')
    
    if args.processes < 1:
        raise argsparse.ArgumentTypeError('Zero or negative processes requested.')
        
    # Check training options
    
    if any(i < 1 for i in args.nhiddens):
        raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(args.hidden)))
        
    if args.nlatent < 1:
        raise ValueError('Minimum 1 latent neuron, not {}'.format(args.latent))
        
    if args.nepochs < 1:
        raise ValueError('Minimum 1 epoch, not {}'.format(args.nepochs))
        
    if args.batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(args.batchsize))
        
    if args.tnfweight < 0:
        raise ValueError('TNF weight cannot be negative')
        
    if args.lrate < 0:
        raise ValueError('Learning rate cannot be negative')
        
    if args.cuda and not _torch.cuda.is_available():
        raise ModuleNotFoundError('Cuda is not available for PyTorch')
        
    # Check clustering options
    if args.nsamples < 5:
        raise ValueError('Must use at least 5 samples to determine threshold')
    
    if args.max_steps < 1:
        raise ValueError('Max steps must be 1 or above.')
        
    # Run the program
    run(args.outpath, args.fasta, args.bamfiles, minlength=args.minlength,
        minascore=args.minascore,
        readprocesses=args.processes,
        nhiddens=args.nhiddens,
        nlatent=args.nlatent,
        nepochs=args.nepochs,
        batchsize=args.batchsize,
        cuda=args.cuda,
        tnfweight=args.tnfweight,
        lrate=args.lrate,
        verbose=True,
        tandemcluster=args.tandem,
        max_steps=args.max_steps,
        nsamples=args.nsamples,
        min_clustersize=1)

