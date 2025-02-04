import os
import shutil
import subprocess
from pathlib import Path
from typing import List

import rich_click as click
from loguru import logger


class CliRunner:
    """
    Class for building and executing command-line commands.
    """

    dry_run_command = False

    def __init__(self) -> None:
        self._argument_holder = []
        self._command_has_been_added = False

    def add_command_to_run(self, command_to_run):
        if self._command_has_been_added:
            raise Exception(
                f"A command has allready been added: {self._argument_holder[0]}"
            )
        self._argument_holder = [command_to_run] + self._argument_holder
        self._command_has_been_added = True

    def add_arguments(self, arguments: List):
        self._argument_holder += arguments

    def prettyprint_args(self):
        [print(x, end=" ") for x in self._argument_holder]
        print()

    def run(self):
        if self.dry_run_command:
            logger.info("running:")
            logger.info(self._argument_holder)
        else:
            print("Running:")
            self.prettyprint_args()
            subprocess.run(self._argument_holder)
            print("Ran:")
            self.prettyprint_args()


class SnakemakeRunner(CliRunner):
    """
    Class to run snakemake workflows from python.
    """

    output_directory = os.getcwd()
    _snakemake_path = shutil.which("snakemake")
    _src_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent

    def __init__(self, snakefile: str = "snakefile.smk"):
        super().__init__()
        self._config_options = []
        self.to_print_while_running_snakemake = None
        self.add_command_to_run(self._snakemake_path)
        self.snakefile_path = Path(Path(self._src_dir) / snakefile)
        self.add_arguments(["--snakefile", self.snakefile_path])
        self.add_arguments(["--rerun-triggers", "mtime"])
        self.add_arguments(["--nolock"])
        self.validate_paths()

    def validate_paths(self):
        if not self.snakefile_path.exists():
            raise click.UsageError(
                f"Could not find snakefile, tried: {self.snakefile_path}"
            )

        if self._snakemake_path is None:
            raise click.UsageError(
                """Could not find snakemake, is it installed?
See following installation guide: https://snakemake.readthedocs.io/en/stable/getting_started/installation.html"""
            )

        if shutil.which("mamba") is None:
            logger.warning(
                "Could not find mamba installation, is the correct environment activated?"
            )
            logger.warning(
                "Defaulting to use conda to build environments for snakemake, this will be slower"
            )
            self.add_arguments(["--conda-frontend", "conda"])

    def add_to_config(self, to_add):
        self._config_options += [to_add]

    def run(self):
        # use conda: always
        self.add_arguments(["--use-conda"])
        # Always rerun incomplete
        self.add_arguments(["--rerun-incomplete"])

        self.add_to_config(f"output_directory={self.output_directory}")
        self.add_to_config(f"src_dir={self._src_dir}")
        # Add config options -- these needs to be passed together to snakemake
        # eg. --config <arg1> <arg2> Therefore we collect them in the self._config_options variable and
        # append them at the end
        self.add_arguments((["--config"] + self._config_options))
        # Possibility of adding info to snakemake for when it runs
        if self.to_print_while_running_snakemake is not None:
            logger.info(self.to_print_while_running_snakemake)
        # Run snakemake
        super().run()
