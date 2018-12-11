
import sys
import os
import torch
import argparse
import datetime
import time
import shutil
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vamb

DEFAULT_PROCESSES = min(os.cpu_count(), 8)

def log(string, logfile, indent=0):
    print(('\t' * indent) + string, file=logfile)
    logfile.flush()

def calc_tnf(outdir, fastapath, mincontiglength, logfile):
    begintime = time.time()

    log('\nCalculating TNF', logfile, 0)

    with open(fastapath, 'rb') as tnffile:
        ret = vamb.parsecontigs.read_contigs(tnffile,
                                             minlength=mincontiglength,
                                             preallocate=True)
    tnfs, contignames, contiglengths = ret

    vamb.vambtools.write_npz(os.path.join(outdir, 'tnf.npz'), tnfs)

    elapsed = round(time.time() - begintime, 2)
    ncontigs = len(contiglengths)
    nbases = contiglengths.sum()

    log('Processed {} bases in {} contigs.'.format(nbases, ncontigs), logfile, 1)
    log('Calculated TNF in {} seconds.'.format(elapsed), logfile, 1)

    return tnfs, contignames

def calc_rpkm(outdir, bampaths, mincontiglength, minalignscore, subprocesses,
              ncontigs, logfile):
    begintime = time.time()

    log('\nCalculating RPKM', logfile)
    log('Parsing {} BAM files with {} subprocesses.'.format(len(bampaths), subprocesses),
       logfile, 1)

    dumpdirectory = os.path.join(outdir, 'tmp')
    rpkms = vamb.parsebam.read_bamfiles(bampaths,
                                        dumpdirectory=dumpdirectory,
                                        minscore=minalignscore,
                                        minlength=mincontiglength,
                                        subprocesses=subprocesses,
                                        logfile=logfile)

    if len(rpkms) != ncontigs:
        raise ValueError('Number of FASTA vs BAM file headers do not match. '
                         'Are you sure the BAM files originate from same FASTA file '
                         'and have headers?')

    vamb.vambtools.write_npz(os.path.join(outdir, 'rpkm.npz'), rpkms)
    elapsed = round(time.time() - begintime, 2)
    log('Calculated RPKM in {} seconds.'.format(elapsed), logfile, 1)

    shutil.rmtree(dumpdirectory)
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
    n_discarded = len(mask) - mask.sum()
    log('Number of contigs unsuitable for encoding: {}'.format(n_discarded), logfile, 1)
    log('Number of contigs remaining: {}'.format(len(mask) - n_discarded), logfile, 1)

    modelpath = os.path.join(outdir, 'model.pt')
    log('\nTraining VAE', logfile, 1)
    vae.trainmodel(dataloader, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps,
                  logfile=logfile, modelfile=modelpath)

    latent = vae.encode(dataloader)
    vamb.vambtools.write_npz(os.path.join(outdir, 'latent.npz'), latent)
    del vae # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    print('', file=logfile)
    log('Trained VAE and encoded in {} seconds.'.format(elapsed), logfile, 1)

    return mask, latent

def cluster(outdir, latent, contignames, maxclusters, minclustersize, logfile):
    begintime = time.time()

    log('\nClustering', logfile)
    clusteriterator = vamb.cluster.cluster(latent, labels=contignames,
                                           logfile=logfile, destroy=True)

    with open(os.path.join(outdir, 'clusters.tsv'), 'w') as clustersfile:
        clusternumber, ncontigs = vamb.cluster.write_clusters(clustersfile,
                                                              clusteriterator,
                                                              max_clusters=maxclusters,
                                                              min_size=minclustersize)

    log('Clustered {} contigs in {} bins.'.format(ncontigs, clusternumber), logfile, 1)

    clusterdonetime = time.time()
    elapsed = round(time.time() - begintime, 2)
    log('Clustered contigs in {} seconds.'.format(elapsed), logfile, 1)

def main(outdir, fastapath, bampaths, mincontiglength, minalignscore, subprocesses,
         nhiddens, nlatent, nepochs, batchsize, cuda, alpha, beta, dropout, lrate,
         batchsteps, minclustersize, maxclusters, logfile):

    # Print starting vamb version ...
    log('Starting Vamb version ' + '.'.join(map(str, vamb.__version__)), logfile)
    log('Date and time is ' + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()

    # Get TNFs, save as npz
    tnfs, contignames = calc_tnf(outdir, fastapath, mincontiglength, logfile)

    # Parse BAMs, save as npz
    rpkms = calc_rpkm(outdir, bampaths, mincontiglength, minalignscore, subprocesses, len(contignames), logfile)


    # Train, save model
    mask, latent = trainvae(outdir, rpkms, tnfs, nhiddens, nlatent, alpha, beta,
                           dropout, cuda, batchsize, nepochs, lrate, batchsteps, logfile)

    del tnfs, rpkms
    contignames = [c for c, m in zip(contignames, mask) if m]

    # Cluster, save tsv file
    cluster(outdir, latent, contignames, maxclusters, minclustersize, logfile)

    elapsed = round(time.time() - begintime, 2)
    log('\nCompleted Vamb in {} seconds.'.format(elapsed), logfile)

__cmd_doc__ = """Run the Vamb pipeline.

For advanced use and extensions of Vamb, check documentation of the package
at https://github.com/jakobnissen/vamb.
"""
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
reqos.add_argument('outdir', help='output directory to create')
reqos.add_argument('fasta', help='path to fasta file')
reqos.add_argument('bamfiles', help='path to BAM files', nargs='+')

# Optional arguments
inputos = parser.add_argument_group(title='IO options', description=None)

inputos.add_argument('-m', dest='minlength', metavar='', type=int, default=100,
                     help='ignore contigs shorter than this [100]')
inputos.add_argument('-s', dest='minascore', metavar='', type=int, default=50,
                     help='ignore reads with alignment score below this [50]')
inputos.add_argument('-p', dest='subprocesses', metavar='', type=int, default=DEFAULT_PROCESSES,
                     help=('reading subprocesses to spawn '
                          '[min(' + str(DEFAULT_PROCESSES) + ', nbamfiles)]'))

# VAE arguments
vaeos = parser.add_argument_group(title='VAE hyperparameters', description=None)

vaeos.add_argument('-n', dest='nhiddens', metavar='', type=int, nargs='+',
                    default=[325, 325], help='hidden neurons [325 325]')
vaeos.add_argument('-l', dest='nlatent', metavar='', type=int,
                    default=40, help='latent neurons [40]')
vaeos.add_argument('-a', dest='alpha',  metavar='',type=float,
                    default=0.05, help='alpha, weight of TNF versus depth loss [0.05]')
vaeos.add_argument('-b', dest='beta',  metavar='',type=float,
                    default=200.0, help='beta, capacity to learn [200.0]')
vaeos.add_argument('-d', dest='dropout',  metavar='',type=float,
                    default=0.2, help='Dropout [0.2]')
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

# Contigs exists
if not os.path.isfile(args.fasta):
    raise FileNotFoundError(args.fasta)

# All bamfiles exists
for bampath in args.bamfiles:
    if not os.path.isfile(bampath):
        raise FileNotFoundError(bampath)

####################### CHECK ARGUMENTS FOR TNF AND BAMFILES ###########
if args.minlength < 100:
    raise argparse.ArgumentTypeError('Minimum contig length must be at least 100')

if args.subprocesses < 1:
    raise argparse.ArgumentTypeError('Zero or negative subprocesses requested.')

if args.minascore < 0:
    raise argparse.ArgumentTypeError('Minimum alignment score cannot be negative')

####################### CHECK VAE OPTIONS ################################
if any(i < 1 for i in args.nhiddens):
    raise argparse.ArgumentTypeError('Minimum 1 neuron per layer, not {}'.format(min(args.hidden)))

if args.nlatent < 1:
    raise argparse.ArgumentTypeError('Minimum 1 latent neuron, not {}'.format(args.latent))

if args.alpha <= 0 or args.alpha >= 1:
    raise argparse.ArgumentTypeError('alpha must be above 0 and below 1')

if args.beta <= 0:
    raise argparse.ArgumentTypeError('beta cannot be negative or zero')

if args.dropout <= 0 or args.dropout >= 1:
    raise argparse.ArgumentTypeError('dropout must be above 0 and below 1')

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

################### RUN PROGRAM #########################
os.mkdir(args.outdir)
logpath = os.path.join(args.outdir, 'log.txt')

with open(logpath, 'w') as logfile:
    main(args.outdir, args.fasta, args.bamfiles,
         mincontiglength=args.minlength,
         minalignscore=args.minascore,
         subprocesses=args.subprocesses,
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
         minclustersize=args.minsize,
         maxclusters=args.maxclusters,
         logfile=logfile)
