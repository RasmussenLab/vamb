import string
import os
import pathlib

import vamb

PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(PARENTDIR, "test", "data")
BAM_FILES = sorted(
    [
        pathlib.Path(DATADIR).joinpath("bam").joinpath(i)
        for i in os.listdir(os.path.join(DATADIR, "bam"))
    ]
)
AEMB_DIR = os.path.join(DATADIR, "aemb")
AEMB_FILES = sorted([pathlib.Path(AEMB_DIR).joinpath(i) for i in os.listdir(AEMB_DIR)])

BAM_NAMES = [
    "S27C175628",
    "S27C95602",
    "S27C25358",
    "S26C115410",
    "S4C529736",
    "S27C181335",
    "S4C222286",
    "S27C38468",
    "S11C13125",
    "S4C480978",
    "S27C255582",
    "S27C170328",
    "S7C221395",
    "S26C281881",
    "S12C228927",
    "S26C86604",
    "S27C93037",
    "S9C124493",
    "S27C236159",
    "S27C214882",
    "S7C273086",
    "S8C93079",
    "S12C85159",
    "S10C72456",
    "S27C19079",
]

BAM_SEQ_LENS = [
    2271,
    3235,
    3816,
    2625,
    2716,
    4035,
    3001,
    2583,
    5962,
    3774,
    2150,
    2161,
    2218,
    2047,
    5772,
    2633,
    3400,
    3502,
    2103,
    4308,
    3061,
    2464,
    4099,
    2640,
    2449,
]


def make_randseq(rng, frm: int, to: int) -> vamb.vambtools.FastaEntry:
    name = rng.choice(string.ascii_uppercase) + "".join(
        rng.choices(string.ascii_lowercase, k=11)
    )
    seq = "".join(
        rng.choices(
            "acgtACGTnNywsdbK",
            weights=[0.12] * 8 + [0.005] * 8,
            k=rng.randrange(frm, to),
        )
    )
    return vamb.vambtools.FastaEntry(name.encode(), bytearray(seq.encode()))
