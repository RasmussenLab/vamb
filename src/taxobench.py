from math import log
from pathlib import Path

from vamb.taxonomy import (
    ContigTaxonomy,
    PredictedContigTaxonomy,
    Taxonomy,
    PredictedTaxonomy,
)

# The score is computed as the log of the probability assigned to the right species.
# At any clade, we assume there are e^2+1 children, and all the children not predicted
# have been given the same score.

# Examples:
# 1) Correct guess at species level. The predictor predicts the species with score 0.8:
#    Result: log(0.8)

# 2) Correct guess at genus level; wrong at species level with score 0.8:
# The remaining score of 0.8 is divided by the remaining e^2 children:
#    Result: log(0.2 / e^2) = log(0.2) - 2

# 3) Correct guess at family level; wrong at genus level with score 0.8:
# The remaining score of 0.2 is divided among e^2 children, each whom have e^2+1 children.
#    Result: log(0.2 / (e^2 * (e^2 + 1))) - we round this off to log(0.2 / (e^2 * e^2)) = log(0.2) - 4

# So: Result is: If correct, log of last score. If N levels are incorrect, it's log(1 - score at first level) - 2N


# INVARIANT: Must be canonical
def pad_tax(x: list):
    x = x.copy()
    if len(x) > 6:
        return x
    x.extend([None] * (7 - len(x)))
    x.reverse()
    return x


def score(true: ContigTaxonomy, pred: PredictedContigTaxonomy) -> float:
    for rank, ((true_tax, pred_tax, prob)) in enumerate(
        zip(true.ranks, pred.contig_taxonomy.ranks, pred.probs)
    ):
        if true_tax != pred_tax:
            wrong_ranks = 7 - rank
            return log(1 - prob) - 2 * wrong_ranks

    for n_wrong_minus_one, (truerank, predrank, prob) in enumerate(
        zip(pad_tax(true.ranks), pad_tax(pred.contig_taxonomy.ranks), pred.probs)
    ):
        if truerank != predrank:
            return log(1 - prob) - 2 * (n_wrong_minus_one + 1)
    return log(pred.probs[-1])


def load_scores(truth_path: Path, pred_path: Path) -> list[tuple[str, int, float]]:
    truth = dict(Taxonomy.parse_tax_file(truth_path, True))
    pred = PredictedTaxonomy.parse_tax_file(pred_path, True)
    return [
        (name, length, score(truth[name], contig_pred))
        for (name, length, contig_pred) in pred
    ]


def weighted_score(lst: list[tuple[str, int, float]]) -> float:
    return sum(i[1] * i[2] for i in lst) / sum(i[1] for i in lst)
