## Withlist for potential improvements to Vamb

Vamb is under an MIT licence, so please feel free to fork, extend, or copy Vamb however you see fit. If you do, feel free to write us an email.

Here's a short wish list for improvements to Vamb we haven't had time for:

__Fix bad optimization of logsigma in Vamb__

In current code, the logsigma layer is calculated using a Linear layer followed by a softplus activation. The activation constrains logsigma to be in (0, ∞), e.g. σ is in (1, ∞).

However, reconstruction loss pushes σ ➡ 0, ⟹ logsigma ➡ -∞. And KLD pushes σ ➡ 1, ⟹ logsigma ➡ 0. With nonzero weight on reconstruction loss and KLD, optimal σ must be somewhere in (-∞, 0).

As this can never be reached due to the activation function, the optimizer _always_ pushes logsigma ➡ 0, meaning that the value before the activation function goes ➡ -∞. This is bad for two reasons:

* It means the optimal value of σ is actually impossible to even _approximately_ reach, probably leading to poorer performance,

* It creates a numerical instability due to float underflow. Luckily, the gradient also disappears as logsigma ➡ 0, so the underflow rarely occurs, but still.

I tried changing this, but this is so deep in Vamb that a removal of this activation function requires new hyperparameter optimization and benchmarking.

__Implement importance sampling in the VAE__

Idea from this paper: https://arxiv.org/abs/1509.00519
The general idea is that the CE/SSE losses are only calculated from a single sample of the latent distribution. This creates unavoidable sampling error, which can be reduced by sampling multiple values and taking an average of the CE/SSE loss.

Maybe try with 5 samples in the beginning, and see how it performs.

__Implement better user control of multithreading__

Vamb uses multiple threads/processes. Sadly, at the moment, there is no good way for the user to control the number of used threads - and this is harder to implement than one would think.

For parsing BAM files, Vamb spawns a number of subprocesses using the *subprocess* module. This number can easily be controlled, so no problem there. Training the VAE, encoding and clustering, however, is done with Numpy and PyTorch, which themselves call an underlying multithreaded BLAS library (which differ from computer to computer). By default, these libraries use all available threads.

Numpy does not provide any good API to limit the number of used threads. If one sets an environmental variables pertaining to the correct BLAS library *before* importing Numpy, then Numpy can limit the number of threads. Having to specify the number of thread before importing Numpy is no good: Users could import Numpy before using Vamb, in which case Vamb *can't* set the number of threads. Also, in order to be installed as a command-line tool, Vamb must parse the number of requested threads in a function call (`__main__.py`'s `main` function) - and imports are not permitted inside a function call.

PyTorch *does* provide `torch.set_num_threads`, but when I tested it, it simply didn't work and used all threads regardless. Vamb calls this function anyway, in case the torch folks fix it.

__Sinusoidal increase of batchsize__

When batchsize increases during training, we see an abrupt drop in loss. I suspect that this immediate drop has very little to do with the intention of increasing batchsize (finding more "flat" minima, e.g. robust minima, as well as simply decreasing stochastic movement similarly to simulated annealing). Rather, it seems that when changing the batch size, the network suddenly find that its local minima is no longer a minima, and jerks in a more favorable direction. Interestingly, this is *not* accompanied by an initial increase in loss, so it's an all-round win. In fact, in our tests, it seems to be an amazingly effective way to kick your network out of a local minima.

Cyclical changes in learning rate have been used widely in the literature. Perhaps we can alter batch size cyclically, such that it varies between e.g. 64 and 4096 (with more epochs at the larger batch sizes), perhaps with the minimal batch size increasing over time. It could also be interesting to look at what with a cyclical batch size with single epochs of very low batch sizes, like 32 or 16 - does this give even bigger kicks to the network, allowing it to explore even lower minima?

__The relationship between number of epochs, beta, and batchsize__

The best number of epochs might not be exactly 500 - but this number surely depends on a lot of other factors. It's probably best to explore the other hyperparameters with a fixed number of epochs (say 500, or even 1000), then later optimize number of epochs.

__Better exploitation of TNF information in clustering__

There's quite a lot of information in the TNF of contigs, and Vamb is fairly bad at exploiting it.

The expected TNF SSE distance between two contigs follows approximately a chi squared distribution (exercise left for reader: assuming all tetranucleotides have an independent probability of being observed, show that this is true). More importantly, in *practice*, it follows a chi square distibution. Similarly, the empirical probability of two contigs belonging to different species as a function of the TNF is well modeled by the cumulative density function of a chi square distribution.

However, the exact shape of the chi square distribution depends on the phylogenetic distance between the genomes of the contigs, and also the the lengths of the two contigs. Hence, a clustering algorithm which simply looks at the raw TNF value without taking in to account contig lengths is not as effective as it could be.

When experimenting with Vamb, we checked if we could, when presented with random pairs of contigs, heuristically estimate the parameters of the estimated chi square distribution of TNF distances between contigs of those lengths and based on that distribution predict whether or not the two contigs belonged in the same bin. It was quite accurate, but instantiating a `scipy.stats.chi2` object with the right parameters for each contig pair would make our clustering algorithm take weeks or months to run for a one-million-contig dataset.

A possible future approach could be to encode the depths and TNF independently with the VAE (although it could still train using both depths and TNF at the same time), and, when calculating the contig-contig distances during clustering, using a heuristic approximation of the chi distribution which can be computed quickly. Alternatively, one could cluster exclusively on depths, then afterwards identify contigs in bins with divergent TNF and/or recruit unbinned contigs to bins with similar TNF.

__Use Markov-normalized kmer frequencies instead of TNF__

A large part of the signal in TNF actually comes from the implicit kmer distribution of k = {1, 2, 3}. By normalizing 3-mer frequency by the expected values based on 2-mer frequencies, a purer form om 3-mer frequencies can be obtained. For example, the expected frequency f(ATG) can be estimated as f(AT)\*f(TG)/f(T).

We can then replace TNF with the concatenation of one neuron for GC content, 10 for Markov normalized 2-mer frequencies and 32 for Markov normalized 3-mer frequencies. We probably shouldn't do 4-mer frequencies, since that would create too many features - when Markov normalizing, the values become even more sensitive to short contig lengths. We can call this Markov Normalized Kmers (MNK)

In simple experiments, MNK separares strains and species better than TNF using cosine or euclidian distances, despite being less than 1/3 as many features. However, perhaps the VAE's encoder implicitly learns the interactions between the TNFs in a way that mimics the Markov normalization, which would actually make MNK less effective than TNF. It will have to be tested.

__Implement an optional two-step clustering method for large datasets, if possible__

So our clustering scales quadratically. That's alright for normally sized datasets, but at some point, someone is going to want to cluster 100 million sequences. Is there a way to avoid this scaling?

Well, if we look at Canopy clustering (not to be confused with the binning method referred to as Canopy), that prevents the issue by first creating large, intersecting (i.e. non-disjoint) sets of points which I will call `partitions`, and then clustering those partitions independently. After clustering those, the clusters may be merged together by simply removing points that are present in multiple clusters from all their clusters but one.

The problem is this: When splitting the dataset, it's highly likely that the points of a true cluster will end up in different partitions. If that happens, it's impossible for the bin to be reconstructed. And we can't easily partition in a manner which preserves the bins' integrity, because that would require us to know the bins beforehand which is the entire point of clustering!

We can solve it with the following splitting function:

    function split(set contigset, float INNER, float OUTER):
        while there are more than 20,000 contigs in contigset:
            S = random contig from contigset
            partition = All contigs in contigset within OUTER distance of S
            yield partition
            superfluous = All contigs in contigset within INNER distance of S
            delete superfluous from contigset

        yield contigset # last partition with at most 20,000 elements

We can prove this works:

    For any bin B, for any partition seed S, let C be the contig in B closest to S
    Let F be the contig in B furthest from S
    If |SC| > INNER, all contigs in B remains in contigset and the bin is not split (1)
    If |SF| ≤ OUTER, all contigs in B is contained in the partition, bin is not split (2)
    Let us assume we have picked values of INNER and OUTER such that OUTER-INNER > |CF| (cond. A)

    If |SC| > INNER:
        Bin B is not split since (1)

    Else it must be the case that:
        |SC| ≤ INNER, which can be rearranged by (A) to
        |SC| ≤ OUTER - |CF|, adding |CF| gives
        |SC| + |CF| ≤ OUTER, and by the tringle inequality |SF| ≤ |SC| + |CF|:
        |SF| ≤ OUTER, which, by (2) means bin B is not split

Hence we just need to pick values for `INNER` and `OUTER` to follow condition A, which means that the difference between `INNER` and `OUTER` should be above the cluster diameter for most realistic bins. No problem.

Now, this relies on the triangle inequality `|SF| ≤ |SC| + |CF|` which does not hold true for Pearson distance, cosine distance etc. However, for cosine distance, the distances can be converted to radians, which **do** follow the triangle inqeuality. E.g, you can add cosine distances with this function:

    def add_distances(a, b):
        radians_a = math.acos(1 - 2*a)
        radians_b = math.acos(1 - 2*b)
        radians_sum = min(math.pi, radians_a + radians_b)
        return 0.5 - 0.5 * math.cos(radians_sum)

However, in my initial analysis, even for relatively small values of INNER, if the threshold is around 0.1, OUTER will become close to 0.5, meaning that each partition will remove only a small fraction of the points (e.g. 1/500), while still containing half the points in the original dataset.
