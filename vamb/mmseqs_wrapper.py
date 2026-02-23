
from ntpath import exists
from pymmseqs.utils import run_mmseqs_command
import subprocess
from pymmseqs.utils import get_mmseqs_binary
from pathlib import Path
from typing import List
from loguru import logger
import shlex
import argparse
import sys
from enum import Enum
import os

class MMseqsRunner:
    """
    Class for building and executing a mmseqs command
    """

    dry_run_command = False
    binary = get_mmseqs_binary()

    def __init__(self) -> None:
        self._argument_holder = [self.binary]

    def add_arguments(self, arguments: List):
        self._argument_holder += arguments
        return self

    def prettyprint_args(self):
        command_str = shlex.join(map(str, self._argument_holder))
        logger.debug(f"Executing command: {command_str}")

    def run(self):
        if self.dry_run_command:
            logger.debug(self._argument_holder)
        else:
            self.prettyprint_args()
            subprocess.run(self._argument_holder)
            self.prettyprint_args()

class Database(Enum):
    GTDB = "GTDB"
    KALAMARI  = "Kalamari"
    FILTERED_DB = "filtered_db"

class Mmseqs():
    def __init__(self, dbdir: Path, database: Database):
        self.dbdir = dbdir
        self.database = database

    def installDatabase(self):
        """
        Install a mmseqs database
        """

        if self.DatabaseExist():
            raise FileExistsError(f"Database at `{self.dbdir}` allready exist")

        # MMseqs downloads databases for path a/b/c as: for directory b it creates files c.1 c.2 .. inside it. 
        # This is not that logical so instead we change it so users pass a path and 
        # the program makes a directory there with the database files
        self.dbdir.mkdir(parents=True, exist_ok=True)
        DBdownloadlocationFiles = self.dbdir / self.database.value

        tmpdir: Path = (self.dbdir / "tempfiles_dbdownload_taxvamb_231510") # Random numbers to make tempdir unlikly to exist
        tmpdir.mkdir(exist_ok=True)

        logger.info(f"Installing {self.database.value} to directory {self.dbdir}")
        try:
            MMseqsRunner().add_arguments(["databases", self.database.value, self.dbdir, tmpdir, '--remove-tmp-files']).run()
            if not self.DatabaseExist():
                raise Exception
        except Exception as err:
            logger.error(f"Downloading database failed, rerun to continue installation: {err}")
            raise err
        else:
            # mmseqs has a built in flag to remove tmp files -- but it does not work.
            self.removeTmpFiles(tmpdir, database.value) # We could use a built in tempdir, but since we want to continue a download if it failed we clean up manually

    def DatabaseExist(self) -> bool:
        """
        Check if required files are in the database location.
        """

        files_which_should_exist = [
            f"{self.database.value}",
            f"{self.database.value}.dbtype",
            f"{self.database.value}.index",
            f"{self.database.value}.lookup",
            f"{self.database.value}.source",
            f"{self.database.value}_h",
            f"{self.database.value}_h.dbtype",
            f"{self.database.value}_h.index",
            f"{self.database.value}_mapping",
            f"{self.database.value}_taxonomy",
            # f"{database}.version",  A filtered db does not contain .version, therefore do not require it
        ]
        for file in files_which_should_exist:
            if not (self.dbdir / file).exists():
                logger.debug(f"{self.dbdir / file} does not exist. DBdownloadlocation: {self.dbdir}, database: {self.database}")
                return False
        return True

    def assignTaxonomy(self, contigs:Path, output: Path) -> Path:

        tmp_output  = output / "tmp"
        tmp_output.mkdir(parents=True)
        output_tsv = output / "mmseqs"

        arguments = [
            "easy-taxonomy",  
            contigs,
            self.dbdir / self.database.value, 
            output_tsv, 
            tmp_output, 
            "--tax-lineage", "1",
            "--search-type", "3"
        ]

        MMseqsRunner().add_arguments(arguments).run()

        return Path(str(output_tsv) + "_lca.tsv")

    def removeTmpFiles(self, tmpdir: Path, database: Path):
        # Remove the temp files. Delete specific files for safety. The argument ("--remove-tmp-files") for mmseqs does not work
        tmp_files = (tmpdir / "latest")
        if tmp_files.is_symlink(): # tmpfiles should be referenced by a symlink
            taxonomy_dir = tmp_files.resolve() / "taxonomy"
            (taxonomy_dir / "createindex.sh").unlink()
            (tmp_files.resolve() / f"{database.lower()}.tsv").unlink()
            taxonomy_dir.rmdir()
            (tmp_files.resolve()).rmdir()
            tmp_files.unlink()
            tmpdir.rmdir()

def download_mmseqs_db(args: argparse.Namespace) -> None:
    dbdir: Path = args.output
    if dbdir is None:
        logger.error("--output argument is required, see --help for more information")
        sys.exit()

    if dbdir.exists():
        logger.error("--output allready exist")
        sys.exit()

    mmseqs = Mmseqs(dbdir=dbdir, database=Database.GTDB)
    mmseqs.installDatabase()

def run_mmseqs(dbdir: Path, contigs: Path, outdir: Path) -> Path:

    mmseqs = Mmseqs(dbdir=dbdir, database=Database.FILTERED_DB) # TODO: Go back to GTDB
    output_taxonomy_file = mmseqs.assignTaxonomy(contigs, outdir / "mmseqs")
    return output_taxonomy_file



