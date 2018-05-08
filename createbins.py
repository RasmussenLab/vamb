
__doc__ = """Prints Fasta files for each contig in bins.

Inputs:
    clusterpath: Path of file w. tab-sep lines of clustername, contigname
    contigpath: Path of Fasta file with contigs.
    
Outputs:
    Creates a directory with one Fasta file per cluster
"""



import sys
import os
from collections import defaultdict
import argparse
from . import tools



def load_clusters(path):
    """Loads clusters and returns a {contig: cluster} dictionary.
    
    Input: Tab-sep lines of clustername, contigname
    Output: A {contig: cluster} dict
    """
    
    clusterof = dict()
    with open(clusterpath) as filehandle:
        for line in filehandle:
            stripped = line.rstrip()
            
            if stripped == '':
                continue
                
            cluster, contig = stripped.split('\t')
            clusterof[contig] = cluster
            
    return clusterof



def appendbins(directory, buffer):
    """Empties the buffer into files.
    
    Inputs:
        directory: Name of bin directory to put files
        buffer: A {binname: [list-of-FastaEntries of contigs in bin]}
        
    Outputs: None, but writes the content of the buffer to directory/binname
    """
    
    for cluster, contigs in buffer.items():
        path = os.path.join(directory, cluster) + '.fna'
        
        with open(path, 'a') as filehandle:
            for contig in contigs:
                print(contig, file=filehandle)



def main(clusterpath, contigpath, outputdir, emptyat=100000000):
    """Runs this module. See __doc__ for details"""
    
    os.mkdir(outputdir)

    clusterof = load_clusters(clusterpath)
    
    buffersize = 0
    buffer = defaultdict(list)
    
    with open(contigpath, 'rb') as filehandle:
        entries = vamb.tools.byte_iterfasta(filehandle)
        
        try:
            for entry in entries:
                buffersize += 100 + sys.getsizeof(entry.header) + len(entry)
                
                cluster = clusterof[entry.header]
                buffer[cluster].append(entry)
                
                if buffersize > emptyat:
                    appendbins(clusterpath, buffer)
                    buffer.clear()
                    buffersize = 0
                
        except KeyError as error:
            message = 'Contig {} not in clusters file'.format(error.args[0])
            raise KeyError(message) from None



if __name__ == '__main__':
    parserkws = {'prog': 'createbins.py',
                 'formatter_class': argparse.RawDescriptionHelpFormatter,
                 'usage': '%(prog)s CLUSTERPATH CONTIGPATH OUTPUTDIR',
                 'description': __doc__}

    # Create the parser
    parser = argparse.ArgumentParser(**parserkws)

    parser.add_argument('clusterpath', help='path of tab-sep lines of cluster, contig')
    parser.add_argument('contigpath', help='path of Fasta file of contigs')
    parser.add_argument('outputdir', help='output path')
    
    parser.add_argument('-b', dest='buffersize',
                        help='approximate size of buffer in MB [100]', default=100, type=int)

    # Print help if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    if not os.path.isfile(args.clusterpath):
        raise FileNotFoundError(args.clusterpath)

    if not os.path.isfile(args.contigpath):
        raise FileNotFoundError(args.contigpath)
        
    if os.path.exists(args.outputdir):
        raise FileExistsError(args.outputdir)
        
    if args.buffersize < 1:
        raise ValueError('Buffer must have at least 1 MB, not ' + str(args.buffersize))

    main(args.clusterpath, args.contigpath, args.outputdir, args.buffersize*1000000)

