
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
        if DBdownloadlocation.exists():
            raise FileExistsError(f"downloadlocation: {DBdownloadlocation} allready exist")
        if tmpdir.exists():
            raise FileExistsError(f"tmpdir: {tmpdir} allready exist")


        # MMseqs downloads databases for path a/b/c as: for directory b it creates files c.1 c.2 .. inside it. 
        # This is not that logical so instead we change it so users pass a path and 
        # the program make a directory there with the database files
        DBdownloadlocation.mkdir(parents=True)
        DBdownloadlocation = DBdownloadlocation / database


        MMseqsRunner().add_arguments(["databases", database, DBdownloadlocation, tmpdir, '--remove-tmp-files']).run()
        self.removeTmpFiles(tmpdir, database)

    def removeTmpFiles(self,tmpdir: Path, database):
        # Remove the temp files. Delete specific files for safety. The argument ("--remove-tmp-files") for mmseqs does not work
        tmp_files = (tmpdir / "latest")
        if tmp_files.is_symlink(): # tmpfiles should be refered by a symlink
            taxonomy_dir = tmp_files.resolve() / "taxonomy"
            (taxonomy_dir / "createindex.sh").unlink()
            (tmp_files.resolve() / f"{database.lower()}.tsv").unlink()
            taxonomy_dir.rmdir()
            (tmp_files.resolve()).rmdir()
            tmp_files.unlink()
            tmpdir.rmdir()
            






