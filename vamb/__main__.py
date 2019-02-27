#!/usr/bin/env python3

# More imports below, but the user's choice of processors must be parsed before
# numpy can be imported.
import sys
import os
import argparse
import torch
import datetime
import time
import shutil

DEFAULT_PROCESSES = min(os.cpu_count(), 8)

# These MUST be set before importing numpy
# I know this is a shitty hack, see https://github.com/numpy/numpy/issues/11826
os.environ["MKL_NUM_THREADS"] = str(DEFAULT_PROCESSES)
os.environ["NUMEXPR_NUM_THREADS"] = str(DEFAULT_PROCESSES)
os.environ["OMP_NUM_THREADS"] = str(DEFAULT_PROCESSES)

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import numpy as np
import vamb

################################# DEFINE FUNCTIONS ##########################
def log(string, logfile, indent=0):
    print(('\t' * indent) + string, file=logfile)
    logfile.flush()

def calc_tnf(outdir, fastapath, tnfpath, namespath, lengthspath, mincontiglength, logfile):
    begintime = time.time()
    log('\nLoading TNF', logfile, 0)
    # If no path to FASTA is given, we load TNF from .npz files
    if fastapath is None:
        tnfs = vamb.vambtools.read_npz(tnfpath)
        contignames = vamb.vambtools.read_npz(namespath)
        contiglengths = vamb.vambtools.read_npz(lengthspath)

        if not tnfs.dtype == np.float32:
            raise ValueError('TNFs .npz array must be of float32 dtype')

        if not np.issubdtype(contiglengths.dtype, np.integer):
            raise ValueError('TNFs .npz array must be of an integer dtype')

        if not (len(tnfs) == len(contignames) == len(contiglengths)):
            raise ValueError('Not all of TNFs, names and lengths are same length')

        # Discard any sequence with a length below mincontiglength
        mask = contiglengths >= mincontiglength
        tnfs = tnfs[mask]
        contignames = list(contignames[mask])
        contiglengths = contiglengths[mask]

    else:
        with open(fastapath, 'rb') as tnffile:
            ret = vamb.parsecontigs.read_contigs(tnffile,
                                                 minlength=mincontiglength,
                                                 preallocate=True)
        tnfs, contignames, contiglengths = ret
        vamb.vambtools.write_npz(os.path.join(outdir, 'tnf.npz'), tnfs)

    elapsed = round(time.time() - begintime, 2)
    ncontigs = len(contiglengths)
    nbases = contiglengths.sum()
    log('Kept {} bases in {} sequences.'.format(nbases, ncontigs), logfile, 1)
    log('Processed TNF in {} seconds.'.format(elapsed), logfile, 1)

    return tnfs, contignames

def calc_rpkm(outdir, bampaths, rpkmpath, mincontiglength, minalignscore, subprocesses,
              ncontigs, logfile):
    begintime = time.time()
    log('\nLoading RPKM', logfile)
    # If bampaths is None, we load RPKM directly from .npz file
    if bampaths is None:
        rpkms = vamb.vambtools.read_npz(rpkmpath)

    if not rpkms.dtype == np.float32:
        raise ValueError('RPKMs .npz array must be of float32 dtype')

    else:
        log('Order of columns are:', logfile, 1)
        log('\n\t'.join(bampaths), logfile, 1)
        print('', file=logfile)

        log('Parsing {} BAM files with {} subprocesses.'.format(len(bampaths), subprocesses),
           logfile, 1)

        dumpdirectory = os.path.join(outdir, 'tmp')
        rpkms = vamb.parsebam.read_bamfiles(bampaths,
                                            dumpdirectory=dumpdirectory,
                                            minscore=minalignscore,
                                            minlength=mincontiglength,
                                            subprocesses=subprocesses,
                                            logfile=logfile)
        print('', file=logfile)

    if len(rpkms) != ncontigs:
        raise ValueError('Number of TNF and RPKM sequences do not match. '
                         'Are you sure the BAM files originate from same FASTA file '
                         'and have headers?')

    if bampaths is not None:
        vamb.vambtools.write_npz(os.path.join(outdir, 'rpkm.npz'), rpkms)
        shutil.rmtree(dumpdirectory)

    elapsed = round(time.time() - begintime, 2)
    log('Processed RPKM in {} seconds.'.format(elapsed), logfile, 1)

    return rpkms

def trainvae(outdir, rpkms, tnfs, nhiddens, nlatent, alpha, beta, dropout, cuda,
            batchsize, nepochs, lrate, batchsteps, logfile):

    begintime = time.time()
    log('\nCreating and training VAE', logfile)

    nsamples = rpkms.shape[1]
    vae = vamb.encode.VAE(nsamples=nsamples, nhiddens=nhiddens, nlatent=nlatent,
                            alpha=alpha, beta=beta, dropout=dropout, cuda=cuda)

    log('Created VAE', logfile, 1)
    dataloader, mask = vamb.encode.make_dataloader(rpkms, tnfs, batchsize,
                                                   destroy=True, cuda=cuda)
    log('Created dataloader and mask', logfile, 1)
    vamb.vambtools.write_npz(os.path.join(outdir, 'mask.npz'), mask)
    n_discarded = len(mask) - mask.sum()
    log('Number of sequences unsuitable for encoding: {}'.format(n_discarded), logfile, 1)
    log('Number of sequences remaining: {}'.format(len(mask) - n_discarded), logfile, 1)
    print('', file=logfile)

    modelpath = os.path.join(outdir, 'model.pt')
    vae.trainmodel(dataloader, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps,
                  logfile=logfile, modelfile=modelpath)

    print('', file=logfile)
    log('Encoding to latent representation', logfile, 1)
    latent = vae.encode(dataloader)
    vamb.vambtools.write_npz(os.path.join(outdir, 'latent.npz'), latent)
    del vae # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    log('Trained VAE and encoded in {} seconds.'.format(elapsed), logfile, 1)

    return mask, latent

def cluster(outdir, latent, contignames, windowsize, minsuccesses, maxclusters,
            minclustersize, cuda, logfile):
    begintime = time.time()

    log('\nClustering', logfile)
    log('Min successful thresholds detected: {}'.format(minsuccesses), logfile, 1)
    log('Max clusters: {}'.format(maxclusters), logfile, 1)
    log('Min cluster size: {}'.format(minclustersize), logfile, 1)
    log('Use CUDA for clustering: {}'.format(cuda), logfile, 1)

    it = vamb.cluster.cluster(latent, destroy=True, windowsize=windowsize,
                              minsuccesses=minsuccesses, labels=contignames,
                              cuda=cuda, logfile=logfile)

    with open(os.path.join(outdir, 'clusters.tsv'), 'w') as clustersfile:
        clusternumber, ncontigs = vamb.cluster.write_clusters(clustersfile, it,
                                                              max_clusters=maxclusters,
                                                              min_size=minclustersize)

    print('', file=logfile)
    log('Clustered {} contigs in {} bins.'.format(ncontigs, clusternumber), logfile, 1)

    clusterdonetime = time.time()
    elapsed = round(time.time() - begintime, 2)
    log('Clustered contigs in {} seconds.'.format(elapsed), logfile, 1)

def run(outdir, fastapath, tnfpath, namespath, lengthspath, bampaths, rpkmpath, mincontiglength,
        minalignscore, subprocesses, nhiddens, nlatent, nepochs, batchsize,
        cuda, alpha, beta, dropout, lrate, batchsteps, windowsize, minsuccesses,
        minclustersize, maxclusters, logfile):

    log('Starting Vamb version ' + '.'.join(map(str, vamb.__version__)), logfile)
    log('Date and time is ' + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()

    # Get TNFs, save as npz
    tnfs, contignames = calc_tnf(outdir, fastapath, tnfpath, namespath, lengthspath,
                                 mincontiglength, logfile)

    # Parse BAMs, save as npz
    ncontigs = len(contignames)
    rpkms = calc_rpkm(outdir, bampaths, rpkmpath, mincontiglength, minalignscore,
                      subprocesses, ncontigs, logfile)

    # Train, save model
    mask, latent = trainvae(outdir, rpkms, tnfs, nhiddens, nlatent, alpha, beta,
                           dropout, cuda, batchsize, nepochs, lrate, batchsteps, logfile)

    del tnfs, rpkms
    contignames = [c for c, m in zip(contignames, mask) if m]

    # Cluster, save tsv file
    cluster(outdir, latent, contignames, windowsize, minsuccesses, maxclusters,
            minclustersize, cuda, logfile)

    elapsed = round(time.time() - begintime, 2)
    log('\nCompleted Vamb in {} seconds.'.format(elapsed), logfile)

def main():
    doc = """Vamb: Variational autoencoders for metagenomic binning.

    For advanced use and extensions of Vamb, check documentation of the package
    at https://github.com/jakobnissen/vamb."""
    usage = "python runvamb.py OUTPATH FASTA|.NPZs BAMPATHS|RPKM [OPTIONS ...]"
    parser = argparse.ArgumentParser(
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=usage, add_help=False)

    # Help
    helpos = parser.add_argument_group(title='Help', description=None)
    helpos.add_argument('-h', '--help', help='print help and exit', action='help')

    # Positional arguments
    reqos = parser.add_argument_group(title='Output (required)', description=None)
    reqos.add_argument('--outdir', metavar='', help='output directory to create')

    # TNF arguments
    tnfos = parser.add_argument_group(title='TNF input (either fasta or all .npz files required)')
    tnfos.add_argument('--fasta', metavar='', help='path to fasta file')
    tnfos.add_argument('--tnfs', metavar='', help='path to .npz of TNF')
    tnfos.add_argument('--names', metavar='', help='path to .npz of names of sequences')
    tnfos.add_argument('--lengths', metavar='', help='path to .npz of seq lengths')

    # RPKM arguments
    rpkmos = parser.add_argument_group(title='RPKM input (either BAMs or .npz required)')
    rpkmos.add_argument('--bamfiles', metavar='', help='paths to (multiple) BAM files', nargs='+')
    rpkmos.add_argument('--rpkm', metavar='', help='path to .npz of RPKM')

    # Optional arguments
    inputos = parser.add_argument_group(title='IO options', description=None)

    inputos.add_argument('-m', dest='minlength', metavar='', type=int, default=100,
                         help='ignore contigs shorter than this [100]')
    inputos.add_argument('-s', dest='minascore', metavar='', type=int,
                         help='ignore reads with alignment score below this [None]')
    inputos.add_argument('-p', dest='subprocesses', metavar='', type=int, default=DEFAULT_PROCESSES,
                         help=('number of subprocesses to spawn '
                              '[min(' + str(DEFAULT_PROCESSES) + ', nbamfiles)]'))

    # VAE arguments
    vaeos = parser.add_argument_group(title='VAE hyperparameters', description=None)

    vaeos.add_argument('-n', dest='nhiddens', metavar='', type=int, nargs='+',
                        default=[325, 325], help='hidden neurons [325 325]')
    vaeos.add_argument('-l', dest='nlatent', metavar='', type=int,
                        default=40, help='latent neurons [40]')
    vaeos.add_argument('-a', dest='alpha',  metavar='',type=float,
                        default=None, help='alpha, weight of TNF versus depth loss [Auto]')
    vaeos.add_argument('-b', dest='beta',  metavar='',type=float,
                        default=200.0, help='beta, capacity to learn [200.0]')
    vaeos.add_argument('-d', dest='dropout',  metavar='',type=float,
                        default=None, help='dropout [Auto]')
    vaeos.add_argument('--cuda', help='use GPU [False]', action='store_true')

    trainos = parser.add_argument_group(title='Training options', description=None)

    trainos.add_argument('-e', dest='nepochs', metavar='', type=int,
                        default=500, help='epochs [500]')
    trainos.add_argument('-t', dest='batchsize', metavar='', type=int,
                        default=64, help='starting batch size [64]')
    trainos.add_argument('-q', dest='batchsteps', metavar='', type=int, nargs='+',
                        default=[25, 75, 150, 300], help='double batch size at epochs [25 75 150 300]')
    trainos.add_argument('-r', dest='lrate',  metavar='',type=float,
                        default=1e-3, help='learning rate [0.001]')

    clusto = parser.add_argument_group(title='Clustering options', description=None)
    clusto.add_argument('-w', dest='windowsize', metavar='', type=int,
                        default=200, help='size of window to count successes [200]')
    clusto.add_argument('-u', dest='minsuccesses', metavar='', type=int,
                        default=15, help='minimum success in window [15]')
    clusto.add_argument('-i', dest='minsize', metavar='', type=int,
                        default=1, help='minimum cluster size [1]')
    clusto.add_argument('-c', dest='maxclusters', metavar='', type=int,
                        default=None, help='stop after c clusters [None = infinite]')

    ######################### PRINT HELP IF NO ARGUMENTS ###################
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
    ######################### CHECK INPUT/OUTPUT FILES #####################

    # Outdir does not exist
    args.outdir = os.path.abspath(args.outdir)
    if os.path.exists(args.outdir):
        raise FileExistsError(args.outdir)

    # Outdir is in an existing parent dir
    parentdir = os.path.dirname(args.outdir)
    if parentdir and not os.path.isdir(parentdir):
        raise NotADirectoryError(parentdir)

    # Make sure only one TNF input is there
    if args.fasta is None:
        for path in args.tnfs, args.names, args.lengths:
            if path is None:
                raise argparse.ArgumentTypeError('Must specify either FASTA or the three .npz inputs')
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
    else:
        for path in args.tnfs, args.names, args.lengths:
            if path is not None:
                raise argparse.ArgumentTypeError('Must specify either FASTA or the three .npz inputs')
        if not os.path.isfile(args.fasta):
            raise FileNotFoundError('Not an existing file: ' + args.fasta)

    # Make sure only one RPKM input is there
    if args.bamfiles is None:
        if args.rpkm is None:
            raise argparse.ArgumentTypeError('Must specify either BAM files or RPKM input')
        if not os.path.isfile(args.rpkm):
            raise FileNotFoundError('Not an existing file: ' + args.rpkm)
    else:
        if args.rpkm is not None:
            raise argparse.ArgumentTypeError('Must specify either BAM files or RPKM input')
        for bampath in args.bamfiles:
            if not os.path.isfile(bampath):
                raise FileNotFoundError('Not an existing file: ' + bampath)

    ####################### CHECK ARGUMENTS FOR TNF AND BAMFILES ###########
    if args.minlength < 100:
        raise argparse.ArgumentTypeError('Minimum contig length must be at least 100')

    if args.subprocesses < 1:
        raise argparse.ArgumentTypeError('Zero or negative subprocesses requested.')

    ####################### CHECK VAE OPTIONS ################################
    if any(i < 1 for i in args.nhiddens):
        raise argparse.ArgumentTypeError('Minimum 1 neuron per layer, not {}'.format(min(args.hidden)))

    if args.nlatent < 1:
        raise argparse.ArgumentTypeError('Minimum 1 latent neuron, not {}'.format(args.latent))

    if args.alpha is not None and (args.alpha <= 0 or args.alpha >= 1):
        raise argparse.ArgumentTypeError('alpha must be above 0 and below 1')

    if args.beta <= 0:
        raise argparse.ArgumentTypeError('beta cannot be negative or zero')

    if args.dropout is not None and (args.dropout < 0 or args.dropout >= 1):
        raise argparse.ArgumentTypeError('dropout must be in 0 <= d < 1.')

    if args.cuda and not torch.cuda.is_available():
        raise ModuleNotFoundError('Cuda is not available for PyTorch')

    ###################### CHECK TRAINING OPTIONS ####################
    if args.nepochs < 1:
        raise argparse.ArgumentTypeError('Minimum 1 epoch, not {}'.format(args.nepochs))

    if args.batchsize < 1:
        raise argparse.ArgumentTypeError('Minimum batchsize of 1, not {}'.format(args.batchsize))

    args.batchsteps = sorted(set(args.batchsteps))
    if max(args.batchsteps) >= args.nepochs:
        raise argparse.ArgumentTypeError('All batchsteps must be less than nepochs')

    if args.lrate <= 0:
        raise argparse.ArgumentTypeError('Learning rate must be positive')

    ###################### CHECK CLUSTERING OPTIONS ####################
    if args.minsize < 1:
        raise argparse.ArgumentTypeError('Minimum cluster size must be at least 0.')

    if args.windowsize < 1:
        raise argparse.ArgumentTypeError('Window size must be at least 1.')

    if args.minsuccesses < 1 or args.minsuccesses > args.windowsize:
        raise argparse.ArgumentTypeError('Minimum cluster size must be in 1:windowsize.')

    ###################### SET UP LAST PARAMS ############################

    # This doesn't actually work, but maybe the PyTorch folks will fix it sometime.
    torch.set_num_threads(args.subprocesses)
    subprocesses = DEFAULT_PROCESSES
    if args.bamfiles is not None:
        subprocesses = min(DEFAULT_PROCESSES, len(args.bamfiles))

    ################### RUN PROGRAM #########################
    os.mkdir(args.outdir)
    logpath = os.path.join(args.outdir, 'log.txt')

    with open(logpath, 'w') as logfile:
        run(args.outdir,
            args.fasta,
            args.tnfs,
            args.names,
            args.lengths,
            args.bamfiles,
            args.rpkm,
            mincontiglength=args.minlength,
            minalignscore=args.minascore,
            subprocesses=subprocesses,
            nhiddens=args.nhiddens,
            nlatent=args.nlatent,
            nepochs=args.nepochs,
            batchsize=args.batchsize,
            cuda=args.cuda,
            alpha=args.alpha,
            beta=args.beta,
            dropout=args.dropout,
            lrate=args.lrate,
            batchsteps=args.batchsteps,
            windowsize=args.windowsize,
            minsuccesses=args.minsuccesses,
            minclustersize=args.minsize,
            maxclusters=args.maxclusters,
            logfile=logfile)
