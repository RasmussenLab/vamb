import os
import shutil
from pathlib import Path

from loguru import logger

from workflow_plamb.command_line_runners import CliRunner, SnakemakeRunner


class EnvironmentManager:
    _src_dir = Path(Path(os.path.dirname(os.path.realpath(__file__)))).parent.parent
    _genomad_dir = _src_dir / "genomad_db"
    _genomad_db_exist = (_genomad_dir).exists()

    def install_genomad_db(self):
        logger.info(f"Installing Genomad database (~3.1 GB) to {self._genomad_dir}")
        snakemake_runner = SnakemakeRunner()
        # Download the directory in the location of the current file
        snakemake_runner.add_arguments(["--directory", self._src_dir])
        # We need to download the genomad tools first so conda needs to be passed to snakemake
        snakemake_runner.add_arguments(["--use-conda"])
        snakemake_runner.add_arguments(["-c", "1"])
        # Set target rule to genomad_db to create the database
        snakemake_runner.add_arguments(["download_genomad_db"])
        snakemake_runner.run()

    def install_conda_environments(self):
        logger.info(f"Installing conda environments")
        snakemake_runner = SnakemakeRunner()
        snakemake_runner.add_arguments(["--use-conda", "--conda-create-envs-only"])
        snakemake_runner.run()

    def setup_environment(self):
        logger.info("Setting up environment")
        if not self._genomad_db_exist:
            self.install_genomad_db()

    def check_if_everything_is_setup(self):
        if True not in [self._genomad_db_exist]:
            logger.info("It seems the environment has not been setup")
            return False
        if not self._genomad_db_exist:
            raise click.UsageError(
                f"Could not find the genomad database, try running the tool with --setup_env"
            )
        return True
