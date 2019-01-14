#!/usr/bin/env python3

__doc__ = """Iterative medoid clustering.

Usage:
TO DO: FILL THIS

Note: Currently can output same contigs in multiple clusters. Just cannot
pick any point as seed which has already been output OR which has already
been picked as a seed.

As a result, not all points are necessarily found in ANY clusters
(since they may be exlcuded from being picked by being a seed, but not part
of the output cluster from that seed due to the medoid wandering away)


"""

import argparse
import numpy as np
import pandas as pd
import torch as torch
import random as random

def _read_data(path):
    """Read in data"""

    # Read in as Numpy matrix with Numpy or Pandas
    if path.endswith('.npz'):
        data = np.load(path)['arr_0']
    elif path.endswith('.npy'):
        data = np.load(path)
    else:
        data = pd.read_csv(path, delimiter="\t")
        data = data.as_matrix()

    return data

def _pearson_distances(tensor, index):
    """Calculates the Pearson distances from row `index` to all rows
        in the matrix, including itself. Returns numpy array of distances"""

    # Distance D = (P - 1) / -2, where P is Pearson correlation coefficient.
    # For two vectors x and y with numbers xi and yi,
    # P = sum((xi-x_mean)*(yi-y_mean)) / (std(y) * std(x) * len(x)).
    # If we normalize matrix so x_mean = y_mean = 0 and std(x) = std(y) = 1,
    # this reduces to sum(xi*yi) / len(x) = x @ y.T / len(x) =>
    # D = ((x @ y.T) / len(x)) - 1) / -2 =>
    # D = (x @ y.T - len(x)) * (-1 / 2len(x))

    # Matrix should have already been zscore normalized by axis 1 (subtract mean, div by std)
    vectorlength = tensor.shape[1]
    result = torch.matmul(tensor, tensor[index])
    result -= vectorlength
    result *= -1 / (2 * vectorlength)

    return result

def _tensor_normalize(tensor, inplace=False):
    """Normalize tensor for faster pearson cor calculations"""

    ncontigs = tensor.shape[0]
    mean = tensor.mean(dim=1).view((ncontigs, 1))
    std = tensor.std(dim=1).view((ncontigs, 1))
    std[std == 0] = 1 # Avoid divide-by-zero errors

    if inplace:
        tensor -= mean
    else:
        tensor = tensor - mean

    tensor /= std
    return tensor

def _getcluster(tensor, kept_mask, point, threshold):
    """Returns a vector of distances to all point, a mask of distances within,
    and the average distance (float)"""

    distances = _pearson_distances(tensor, point)
    mask = (distances <= threshold) & kept_mask
    inner_dists = distances[mask]

    # This happens if std(tensor[point]) == 0, then all pearson distances
    # become 0.5, even the distance to itself.
    if len(inner_dists) == 0:
        average_distance = 0
        mask[point] = 1
        inner_points = _torch.Tensor([point])

    else:
        inner_points = mask.nonzero().reshape(-1)
        if len(inner_dists) == 1:
            average_distance = 0
        else:
            average_distance = inner_dists.sum().item() / (len(inner_dists) - 1)


    return distances, mask, average_distance, inner_points

def _sample_clusters(tensor, kept_mask, point, max_attempts, threshold, rng):
    """Keeps sampling new points within the inner points until it has sampled
        max_attempts without getting a new set of inner points with lower average
        distance"""

    futile_attempts = 0

    # Keep track of tried points to avoid sampling the same more than once
    tried = {point}

    distances, mask, average_distance, inner_points = _getcluster(tensor, kept_mask, point, threshold)

    while len(inner_points) - len(tried) > 0 and futile_attempts < max_attempts:
        sample = rng.choice(inner_points).item()

        while sample in tried: # Not sure there is a faster way to prevent resampling
            sample = rng.choice(inner_points).item()

        tried.add(sample)

        sample_dist, sample_mask, sample_average, sample_inner = _getcluster(tensor, kept_mask, sample, threshold)

        if sample_average < average_distance:
            point = sample
            inner_mask = sample_mask
            average_distance = sample_average
            distances = sample_dist
            futile_attempts = 0
            tried = {point}

        else:
            futile_attempts += 1

    return point, mask

def _cluster(tensor, threshold, max_steps, cuda):
    """Yields (medoid, points) pairs from a (obs x features) matrix"""

    device = 'cuda' if cuda else 'cpu'
    kept_mask = torch.ones(len(tensor), dtype=torch.uint8)
    indices = torch.arange(1, len(tensor) + 1)

    # This can be optimized - if you can figure out how to do this without
    # copying. Note that random.shuffle(tensor) does not work, somehow.
    rng = random.Random(324645)
    randomindices = torch.randperm(len(indices))
    indices = indices[randomindices]
    tensor = tensor[randomindices]
    del randomindices

    if cuda:
        tensor = tensor.cuda()
        kept_mask = kept_mask.cuda()
        indices = indices.cuda()

    # While not all points have been removed
    possible_seeds = kept_mask.nonzero().reshape(-1)
    while len(possible_seeds) > 0:
        # Take first index that has not been clustered
        seed = possible_seeds[0]

        # Find medoid using iterative sampling function above
        medoid, mask = _sample_clusters(tensor, kept_mask, seed, max_steps, threshold, rng)
        kept_mask &= ~mask # And add these new points to the set of removed points

        # Write data to output
        cluster = indices[mask]

        if cuda:
            cluster = cluster.cpu()
        yield medoid, set(cluster.numpy())

        possible_seeds = kept_mask.nonzero().reshape(-1)

def write_clusters(filehandle, clusters, max_clusters=None, min_size=1,
                 header=None):
    """Writes clusters to an open filehandle.

    Inputs:
        filehandle: An open filehandle that can be written to
        clusters: An iterator generated by function `clusters` or a dict
        max_clusters: Stop printing after this many clusters [None]
        min_size: Don't output clusters smaller than N contigs
        header: Commented one-line header to add

    Outputs:
        clusternumber: Number of clusters written
        ncontigs: Number of contigs written
    """

    if not hasattr(filehandle, 'writable') or not filehandle.writable():
        raise ValueError('Filehandle must be a writable file')

    if iter(clusters) is not clusters: # Is True if clusters is not iterator
        clusters = clusters.items()

    if max_clusters is not None and max_clusters < 1:
        raise ValueError('max_clusters must be at least 1.')

    if header is not None and len(header) > 0:
        if '\n' in header:
            raise ValueError('Header cannot contain newline')

        if header[0] != '#':
            header = '# ' + header

        print(header, file=filehandle)

    clusternumber = 0
    ncontigs = 0

    for clustername, contigs in clusters:
        if len(contigs) < min_size:
            continue

        clustername = 'cluster_' + str(clusternumber + 1)

        for contig in contigs:
            print(clustername, contig, sep='\t', file=filehandle)
        filehandle.flush()

        clusternumber += 1
        ncontigs += len(contigs)

        if clusternumber + 1 == max_clusters:
            break

    return clusternumber, ncontigs

def main(args):
    '''Main program'''

    # read data
    data = _read_data(args.i)

    # convert to tensor and normalize
    tensor = _tensor_normalize(torch.from_numpy(data))

    # check if cpu else use cuda
    if args.cpu:
        cuda = False
    else:
        cuda = True

    # create generator and write out clusters
    cl = _cluster(tensor, args.threshold, 15, cuda)
    with open (args.o, 'w') as file:
       written = write_clusters(file, cl, max_clusters=args.max_clusters)

if __name__ == '__main__':

    # create the parser
    parser = argparse.ArgumentParser(prog='pt.gpu_canopy.py', formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50, width=130), usage='%(prog)s [options]', description='Canopy-like clustering using pytorch and GPU')

    parser.add_argument('--i', help='input matrix', required=True)
    parser.add_argument('--cpu', help='use cpu instead of gpu (False)', default=False, action='store_true')
    parser.add_argument('--max_clusters', help='maximum number of clusters (all)', default=None, type=int)
    parser.add_argument('--threshold', help='clustering threshold', type=float)
    parser.add_argument('--o', help='output clustering', default='cl.out')

    args = parser.parse_args()
    #args = parser.parse_args('--i ../vamb_20180924/IGC.latent_L80_E500.npz --max_clusters 5 --o test.cl.txt --cpu'.split())

    main(args)
