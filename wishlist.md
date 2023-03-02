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

