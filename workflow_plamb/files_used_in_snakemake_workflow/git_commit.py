import os
import subprocess


def get_git_commit(script_path):
    try:
        # Get the directory of the script
        script_dir = os.path.dirname(os.path.abspath(script_path))

        # Run the Git command in the script's directory
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=script_dir,  # Specify the working directory for the Git command
            )
            .decode("utf-8")
            .strip()
        )
        return commit
    except subprocess.CalledProcessError:
        return "No Git commit found"
    except FileNotFoundError:
        return "Git not found"
