# Contributing to Vamb
The Git repository is currently hosted at https://github.com/RasmussenLab/vamb

## Git workflow
In order for your contribution to be easily integrated into a package that is concurrently worked on by multiple people, it's important that you adhere to the Git workflow that we use for this repo.

#### Feature branches
We never push directly to master. Instead, create a new feature branch on your own fork, and make a PR from your fork to master.
Feature branches are any branch that contain new code that will eventually be merged to master, whether this is an actual feature, or a bugfix or whatever.
We recommend creating your feature branch from an updated version of master, to make it easier to merge into master again.

For large features, feature branches can contain huge changes, and be in development over months. Rebase on master as often as possible.
However, where feasible, keep your feature branches' diff relative to master small. If your feature branch contain multiple independent changes, instead make multiple different PRs on different feature branches. This is easier to review, and to bisect if necessary.

Make sure to squash your commits on your feature branches as necessary to keep the history clean.
A good rule of thumb is that 1 commit = 1 PR, but there may be exceptions.
Also, please delete your feature branches after they've been merged to master so they don't accumulate.

#### Release branches
Releases are only cut from release branches.
The purpose of release branches is to keep a version of Vamb that is more stable than the development version found on master.
This stability is achieved by only adding bugfixes to release branches, not new features. Over time, the bugfixes will accumulate, while the new features (which mostly are where new bugs come from), are added to master only.
Release branches are named "release", plus the major and minor version, like so: "release-4.1". They are always cut from master.
We only backport bugfixes to one, or a few release branches at a time, so old release branches quickly get outdated. However, we will not remove them.

Release branches are never merged back to master. If commits from master are needed in a release branch, you may cherry-pick them from master.
This is the only case where commits may be duplicated on two different branches.

#### Tags
Each release of Vamb (from a release branch) is tagged with a lowercase "v", then a SemVer 2.0 version, e.g. "v4.1.3".
A tag unambiguously refers to a commit, and is never removed.
Ideally, the tagged commit should be the one that updates the version in `vamb/__init__.py`.

#### Testing
Our CI pipeline currently uses a formatter and a linter to check for issues (currently, the Ruff formatter and linter).
To quicken development time, you can install these locally so you can catch these issues before they are caught in CI.

#### Dependencies
Please avoid adding new dependencies if at all practical.
We already have lots of issues with out dependencies, and don't want any more.

## Example commands
We assume:
    - The https://github.com/RasmussenLab/vamb repo is added as a remote with the name `upstream`
    - Your own fork of Vamb is added as a remote called `origin`
### Making an new PR
Syncronize the master branches between your repo and upstream.
Do this before making any new branches from master.
```shell
$ git switch master
$ git pull upstream master
$ git push origin master
```

Make a new branch, with a feature, here for example "kmer-compression".
Name your branch accordingly.
```shell
$ git switch -c kmer-compression
```

Write your code, then test it.
This requires you to have installed Vamb (preferentially with `pip install -e .`),
and installed `pytest` and `ruff`:
```shell
$ python -m pytest # test the code
$ ruff check . # run the linter
$ ruff format . # run the formatter
```

Commit it, then push to `origin`
```shell
$ git add * # add your files
$ git status # check you didnt add spurious unneeded files
$ git commit # then write a commit message
$ git push -u origin kmer-compression
```

Navigate to Vamb's GitHub repo (the `upstream` one), then on the Pull Requests tab make a new PR from `kmer-compression`
to `upstream`'s master.
