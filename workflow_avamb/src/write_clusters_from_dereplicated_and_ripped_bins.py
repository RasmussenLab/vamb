import os
import sys
import getopt
import re

def main(argv):
    drep_dir = ''
    outdir = ''
    
    try:
        opts, args = getopt.getopt(argv, "d:o:")
    except getopt.GetoptError:
        print('error')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-d':
            drep_dir = arg
        elif opt == '-o':
            outdir = arg
        else:
            print('error')
            sys.exit(1)

    output_file = os.path.join(os.getcwd(), outdir, 'avamb', 'avamb_manual_drep_disjoint_clusters.tsv')
    
    print('creating z y v clusters from the final set of bins in %s'%output_file)
    
    for s in os.listdir(drep_dir):
        s_path = os.path.join(drep_dir, s)
        if os.path.isdir(s_path):
            for bin_file in os.listdir(s_path):
                if bin_file.endswith('.fna'):
                    cluster_name = re.sub(r'\.fna$', '', bin_file)
                    cluster_name = re.sub(r'\.fa$', '', cluster_name)
                    bin_path = os.path.join(s_path, bin_file)
                    
                    with open(bin_path, 'r') as bin_f, open(output_file, 'a') as out_f:
                        for line in bin_f:
                            if line.startswith('>'):
                                contig = line[1:].strip()
                                out_f.write(f"{cluster_name}\t{contig}\n")

if __name__ == "__main__":
    main(sys.argv[1:])

