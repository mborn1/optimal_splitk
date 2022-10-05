Design
======

Installation
------------
The installation is easily possible by using ``pip install .`` or directly from git. There are two additional
options: `dev` and `examples` (installed by ``pip install .[dev]`` and ``pip install .[examples]`` respectively). 
The first adds extra dependencies to update the documentation, the
second add some dependencies (like pandas) to run the examples.

Folder structure
----------------
The project contains a couple folders:

* `src`: This contains the source code of the package
* `examples`: Contains some example scripts for using the package
* `docs`: Contains the documentation
* `assets`: Contains possible assets (like figures) for the readme.

Code design
-----------
The code is designed using a couple modules under ``src/doe_gensplit``. The `doe` module contains the main
entry point for generating designs. Besides the main function, the user could add new optimization
criteria by adding a new Python file in the ``optim`` directory. Please inspect the other (D-optimal
and I-optimal closely). Finally the other modules contain helper functions and code, they are
all documented in the API section.
