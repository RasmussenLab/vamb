
import sys
import os
import torch
import argparse
import datetime
import time



sys.path.append('../..')
import vamb



DEFAULT_PROCESSES = min(os.cpu_count(), 8)



def main(outdir, fastapath, bampaths, mincontiglength, minalignscore, subprocesses,
         nhiddens, nlatent, nepochs, batchsize, cuda, errorsum, mseratio,
         minclustersize, maxclusters, logfile):
    # Assume all files are as they should be
    
    # Print starting vamb version ...
    print('Starting Vamb version ', '.'.join(map(str, vamb.__version__)), file=logfile)
    print('\tDate and time is', datetime.datetime.now(), file=logfile)
    starttime = time.time()
    
    # Get TNFs, save as npz
    print('\nCalculating TNF', file=logfile)
    
    with open(fastapath, 'rb') as tnffile:
        tnfs, contignames, contiglengths = vamb.parsecontigs.read_contigs(tnffile, minlength=mincontiglength)
    del contiglengths
    
    vamb.vambtools.write_npz(os.path.join(outdir, 'tnf.npz'), tnfs)
    tnfdonetime = time.time()
    seconds = round(tnfdonetime - starttime, 2)
    print('\tCalculated TNF in {} seconds.'.format(seconds), file=logfile)
    
    # Parse BAMs, save as npz
    print('\nParsing {} BAM files with {} subprocesses.'.format(len(bampaths), subprocesses), file=logfile)
    rpkms = vamb.parsebam.read_bamfiles(bampaths, minalignscore, mincontiglength, subprocesses, logfile=logfile)
    
    if len(rpkms) != len(contignames):
        raise ValueError('Number of FASTA vs BAM file headers do not match. '
                         'Are you sure the BAM files originate from same FASTA file '
                         'and have headers?')
        
    vamb.vambtools.write_npz(os.path.join(outdir, 'rpkm.npz'), rpkms)
    bamdonetime = time.time()
    seconds = round(bamdonetime - tnfdonetime, 2)
    print('\tCalculated RPKM in {} seconds.'.format(seconds), file=logfile)
    
    # Train, save model
    print('\nTraining VAE', file=logfile)
    modelpath = os.path.join(outdir, 'model')
    vae, dataloader = vamb.encode.trainvae(rpkms, tnfs, nhiddens=nhiddens, nlatent=nlatent,
                                          nepochs=nepochs, batchsize=batchsize, cuda=cuda,
                                          errorsum=errorsum, mseratio=mseratio, verbose=True,
                                          logfile=logfile, modelfile=modelpath)
    
    latent = vae.encode(dataloader)
    vamb.vambtools.write_npz(os.path.join(outdir, 'latent.npz'), latent)
    del tnfs, rpkms, dataloader
    
    encodedonetime = time.time()
    seconds = round(encodedonetime - bamdonetime, 2)
    print('\tTrained VAE and encoded in {} seconds.'.format(seconds), file=logfile)
    
    # Cluster, save tsv file
    print('\nClustering', file=logfile)
    clusteriterator = vamb.cluster.cluster(latent, labels=contignames, logfile=logfile)
    
    with open(os.path.join(outdir, 'clusters.tsv'), 'w') as clustersfile:
        vamb.cluster.write_clusters(clustersfile, clusteriterator, max_clusters=maxclusters, min_size=minclustersize)
    
    clusterdonetime = time.time()
    seconds = round(clusterdonetime - encodedonetime, 2)
    print('\tClustered contigs in {} seconds.'.format(seconds), file=logfile)
    
    seconds = round(clusterdonetime - starttime, 2)
    print('\nDone with Vamb in {} seconds.'.format(seconds), file=logfile)



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
                     help='ignore sequences shorter than this [100]')
inputos.add_argument('-a', dest='minascore', metavar='', type=int, default=50,
                     help='ignore reads with alignment score below this [50]')
inputos.add_argument('-p', dest='subprocesses', metavar='', type=int, default=DEFAULT_PROCESSES,
                     help=('reading subprocesses to spawn '
                          '[min(' + str(DEFAULT_PROCESSES) + ', nbamfiles)]'))

vambos = parser.add_argument_group(title='Training options', description=None)

vambos.add_argument('-n', dest='nhiddens', metavar='', type=int, nargs='+',
                    default=[325, 325], help='hidden neurons [325 325]')
vambos.add_argument('-l', dest='nlatent', metavar='', type=int,
                    default=40, help='latent neurons [40]')
vambos.add_argument('-e', dest='nepochs', metavar='', type=int,
                    default=300, help='epochs [300]')
vambos.add_argument('-b', dest='batchsize', metavar='', type=int,
                    default=100, help='batch size [100]')
vambos.add_argument('-s', dest='errorsum',  metavar='',type=float,
                    default=1000.0, help='Amount to learn [1000]')
vambos.add_argument('-r', dest='mseratio',  metavar='',type=float,
                    default=0.2, help='Weight of TNF versus depth [0.2]')
vambos.add_argument('--cuda', help='use GPU [False]', action='store_true')

clusto = parser.add_argument_group(title='Clustering options', description=None)
clusto.add_argument('--tandem', help='use tandem clustering [False]', action='store_true')

clusto.add_argument('-i', dest='minsize', metavar='', type=int,
                    default=1, help='Minimum cluster size [1]')
clusto.add_argument('-c', dest='maxclusters', metavar='', type=int,
                    default=-1, help='Stop after c clusters [-1 = inf]')

######################### PRINT HELP IF NO ARGUMENTS ###################
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

######################### CHECK INPUT/OUTPUT FILES #####################

# Outdir does not exist
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

###################### CHECK TRAINING OPTIONS ####################

if any(i < 1 for i in args.nhiddens):
    raise argparse.ArgumentTypeError('Minimum 1 neuron per layer, not {}'.format(min(args.hidden)))

if args.nlatent < 1:
    raise argparse.ArgumentTypeError('Minimum 1 latent neuron, not {}'.format(args.latent))

if args.nepochs < 1:
    raise argparse.ArgumentTypeError('Minimum 1 epoch, not {}'.format(args.nepochs))

if args.batchsize < 1:
    raise argparse.ArgumentTypeError('Minimum batchsize of 1, not {}'.format(args.batchsize))

if args.errorsum < 0:
    raise argparse.ArgumentTypeError('Errorsum cannot be negative')
    
if args.mseratio <= 0 or args.mseratio >= 1:
    raise argparse.ArgumentTypeError('MSE ratio must be above 0 and below 1')

if args.cuda and not torch.cuda.is_available():
    raise ModuleNotFoundError('Cuda is not available for PyTorch')
    
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
         errorsum=args.errorsum,
         mseratio=args.mseratio,
         cuda=args.cuda,
         minclustersize=args.minsize,
         maxclusters=args.maxclusters,
         logfile=logfile)

