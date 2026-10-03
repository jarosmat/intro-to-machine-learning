This repository is a fork of a repository mentioned below. It contains lecture notes and other stuff important for the
class. On top of that it contains my solutions of assignments from this class.

# Machine Learning for Greenhorns – ÚFAL Course NPFL129

This repository contains materials to the
[Machine Learning for Greenhorns (ÚFAL course NPFL129)](http://ufal.mff.cuni.cz/courses/npfl129).

All created content is available under
[CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/), while all existing materials (mostly images and
excerpts from papers) are properly referenced and are subject to original licensing.

# Dependency management with Nix

This project uses `devenv` so this must be installed. To install dependencies, run:

```shell
devenv shell
```

After dependencies are installed, run:

```shell
devenv up
```

To start server for Jupyter notebooks. This local server then may be used to view Jupyter notebooks from browser or even
in IDEs like Pycharm. After running `devenv up` a URL pointing to localhost should be displayed, open it in browser. If
you wish to use it to run Jupyter notebooks in Pycharm, open a Jupyter notebook, in top right corner click on Jupyter
server, click on Configure Jupyter Server, a window pops up, Add a new server configuration - external server, copy the
displayed URL into field Server URL.