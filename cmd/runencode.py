import argparse

################ Create parser ############################

usage = "python encode.py DEPTHSFILE TNFFILE [OPTIONS ...]"
parser = argparse.ArgumentParser(
    description=__cmd_doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    usage=usage)

# Positional arguments
parser.add_argument('depthsfile', help='path to depths/RPKM file')
parser.add_argument('tnffile', help='path to TNF file')
parser.add_argument('latentfile', help='path to put latent representation')

# Optional arguments
parser.add_argument('-m', dest='modelfile', help='path to put model weights [None]')
parser.add_argument('-n', dest='nhiddens', type=int, nargs='+',
                    default=[325, 325], help='hidden neurons [325 325]')
parser.add_argument('-l', dest='nlatent', type=int,
                    default=40, help='latent neurons [40]')
parser.add_argument('-e', dest='nepochs', type=int,
                    default=300, help='epochs [300]')
parser.add_argument('-b', dest='batchsize', type=int,
                    default=100, help='batch size [100]') 
parser.add_argument('-r', dest='lrate', type=float,
                    default=0.0001, help='learning rate [1e-4]')
parser.add_argument('--cuda', help='Use GPU [False]', action='store_true')

# If no arguments, print help
if len(_sys.argv) == 1:
    parser.print_help()
    _sys.exit()

args = parser.parse_args()

################# Check inputs ###################

if args.cuda and not _torch.cuda.is_available():
    raise ModuleNotFoundError('Cuda is not available for PyTorch')

if any(i < 1 for i in args.nhiddens):
    raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(args.hidden)))

if args.nlatent < 1:
    raise ValueError('Minimum 1 latent neuron, not {}'.format(args.latent))

if args.nepochs < 1:
    raise ValueError('Minimum 1 epoch, not {}'.format(args.nepochs))

if args.batchsize < 1:
    raise ValueError('Minimum batchsize of 1, not {}'.format(args.batchsize))

if args.lrate < 0:
    raise ValueError('Learning rate cannot be negative')

for inputfile in (args.depthsfile, args.tnffile):
    if not _os.path.isfile(inputfile):
        raise FileNotFoundError(inputfile)

for outputfile in (args.outfile, args.modelfile):
    if _os.path.exists(outputfile):
        raise FileExistsError(outputfile)

    directory = _os.path.dirname(outputfile)
    if directory and not _os.path.isdir(directory):
        raise NotADirectoryError(directory)

############## Run program #######################

depths = _vambtools.read_tsv(args.depthsfile)
tnf = _vambtools.read_tsv(args.tnffile)

vae, data_loader = trainvae(depths, tnf, nhiddens=args.nhiddens, latent=args.latent,
                            nepochs=args.nepochs, batchsize=args.batchsize,
                            cuda=args.cuda, lrate=args.lrate, verbose=True,
                           modelfile=args.modelfile)

latent = vae.encode(data_loader)

_vambtools.write_tsv(args.outfile, latent)
