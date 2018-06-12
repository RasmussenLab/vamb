# Troubleshooting

### Parsing the fasta file
__`parsecontigs.read_contigs` fails with ValueError: Non-ACGTN in line [LINE]: [CHAR]__

Vamb can only parse contigs consisting of any bytes in b'acgtnACGTN', so IUPAC ambigious DNA, uracils (U), unknowns (X) and anything else will cause an error.

To work around this, you can exploit the fact that `read_contigs` works with *any* iterator of binary lines. You can simply filter the sequences as you see fit within Python before handing it to the 
function, for example:

    with open('/path/to/contigs.fna', 'rb') as file:
        table = bytes.maketrans(b'YKRMSWBDVH', b'NNNNNNNNNN')        
        lines = (s.translate(table) if not s.startswith(b'>') else s for s in file)
        tnfs, contignames, lengths = vamb.parsecontigs.read_contigs(lines)


### Parsing the BAM files

### VAE or encoding

### Clustering

__It warns: Only [N]% of contigs has well-separated threshold__

See the issue below

__It warns: Only [N]% of contigs has *any* observable threshold__

This happens if the inter-contig distances do not neatly separate into close and far contigs, which can happen if:

* There is little data, e.g. < 50k contigs
* The VAE has not trained sufficiently
* There is, for some reason, little signal in your data

There is not much to do about it. We do not know at what percentage of well-separated or observable threshold Vamb becomes unusable. I would still use Vamb unless the following error is thrown:

__It warns: Too little data: [ERROR]. Setting threshold to 0.08__

This happens if less than 5 of the samples (default of 1000 samples) return any threshold. If this happens for a reasonable number of samples, e.g. > 100, I would not trust Vamb to deliver good results, 
and would use another binner instead.
