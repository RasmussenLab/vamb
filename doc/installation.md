# How to install Vamb
Vamb is in continuous development, and the latest versions are significantly better than older versions.
For the best results, make sure to install the released latest version.

## Recommended: Install with `pip`
Recommended: Vamb can be installed with `pip` (thanks to contribution from C. Titus Brown):
```shell
pip install vamb
```

Note: Check that you've installed the latest version by comparing the installed version with [the latest version on PyPI](https://pypi.org/project/vamb/#history).

Note: An active Conda environment can hijack your system's linker, causing an error during installation.
If you see issues, either deactivate `conda`, or delete the `~/miniconda/compiler_compats` directory before installing with pip.

## Install a specific version of Vamb
If you want to install the latest version from GitHub, or you want to change Vamb's source code, you should install it like this:

```shell
# Clone the desired branch from the repository, here master
git clone https://github.com/RasmussenLab/vamb -b master
cd vamb
# The `-e` flag will make Vamb change if the source code is changed after install
pip install -e .
```

__Note that the master branch is work-in-progress, have not been thoroughly tested, and is expected to have more bugs.__

## Avoid using Conda to install Vamb
The version of Vamb currently on BioConda is out of date, and significantly less accurate than the latest current version.
We have also experienced that our users have more issues with installations from Conda.
We will only be releasing new versions to be installable with `pip`. 
