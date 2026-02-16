
from pymmseqs.utils import run_mmseqs_command
import subprocess
from pymmseqs.utils import get_mmseqs_binary
from pathlib import Path
from typing import List
from loguru import logger
import shlex

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
            logger.debug("Would run:")
            logger.debug(self._argument_holder)
        else:
            logger.debug("Running:")
            self.prettyprint_args()
            subprocess.run(self._argument_holder)
            print("Ran:")
            self.prettyprint_args()

class Mmseqs():
    def installDatabase(self, DBdownloadlocation: Path, tmpdir: Path, database: str):
        """
        Install a mmseqs database
        """

        if database not in ["Kalamari"]:
            raise ValueError()
        if self.DatabaseExist(DBdownloadlocation=DBdownloadlocation, database=database):
            raise FileExistsError(f"downloadlocation: {DBdownloadlocation} allready exist")
        if tmpdir.exists():
            raise FileExistsError(f"tmpdir: {tmpdir} allready exist")

        # MMseqs downloads databases for path a/b/c as: for directory b it creates files c.1 c.2 .. inside it. 
        # This is not that logical so instead we change it so users pass a path and 
        # the program make a directory there with the database files
        DBdownloadlocation.mkdir(parents=True)
        DBdownloadlocationFiles = DBdownloadlocation / database

        logger.info(f"Installing {database} to directory {DBdownloadlocation}")
        try:
            MMseqsRunner().add_arguments(["databases", database, DBdownloadlocationFiles, tmpdir, '--remove-tmp-files']).run()
            if not self.DatabaseExist(DBdownloadlocation=DBdownloadlocation, database=database):
                raise Exception
        except Exception as err:
            logger.error(f"Downloading database failed, rerun to contine installation: {err}")
            raise err
        else:
            # mmseqs has a built in flag to remove tmp files -- but it does not work.
            self.removeTmpFiles(tmpdir, database) # We could use a built in tempdir, but since we want to continue a download if it failed we clean up manually

    def DatabaseExist(self, DBdownloadlocation, database) -> bool:

        files_which_should_exist = [
            f"{database}",
            f"{database}.dbtype",
            f"{database}.index",
            f"{database}.lookup",
            f"{database}.source",
            f"{database}_h",
            f"{database}_h.dbtype",
            f"{database}_h.index",
            f"{database}_mapping",
            f"{database}_taxonomy",
            # f"{database}.version",  A filtered db does not contain .version, therefore do not require it
        ]
        for file in files_which_should_exist:
            if not (DBdownloadlocation / file).exists():
                print(f"{file} does not exist. DBdownloadlocation: {DBdownloadlocation}, database: {database}")
                return False
        return True
                


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

    def assignTaxonomy(self, database: Path, contigs:Path, output: Path):
        # mmseqs easy-taxonomy {input.contigs_decompressed} {params.db} {output.mmseqs2} {output.tmp} --tax-lineage 1

        tmp_output  = output / "tmp"
        tmp_output.mkdir(parents=True)
        output_tsv = output / "tsv"

        arguments = [
            "easy-taxonomy",  
            contigs,
            database, 
            output_tsv, 
            tmp_output, 
            "--tax-lineage", "1",
            "--search-type", "3"
        ]

        MMseqsRunner().add_arguments(arguments).run()


            






