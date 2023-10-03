==========
HemeFinder
==========

HemeFinder is a computational tool for detecting heme binding sites based exclusively on the protein structure and the geometrical predisposition of heme binding sites. This software relies on structural and physico-chemical characteristics of heme sites, including shape, residue composition, and three geometric descriptors of the protein's backbone.  HemeFinder is able to predict natural heme-binding sites and explore the potential to design new ArM based on heme.





Features
--------

**Possible applications:**

* Identification of possible heme binding sites
* Detection of heme binding pathways
* Design of new ArM based on heme 

Installation
--------

The installtion requires a conda environment with few dependencies:

```bash
> conda create -n {name} python=3.9
> conda activate {name}
> pip install git+https://github.com/laura-tiessler/hemefinder/
> pip install pyKVFinder
> conda install -c anaconda numpy 
> conda install -c jmcmurray json 
```

Installation
--------

The software is run from the terminal and if defaulf parameters are used only the input PDB is needed.

* Example with PDB

```bash
> hemefinder target_name.pdb
```

* Example of downloading directy from PDB server:

```bash
> biobrigit target 
```

The main parameters that can be tuned for calculations are the following:

* `--output`: Directory where outputs should be stored. 
* `--coordinators`: List of possible coordinating residues
* `--mutations`: List of possible mutating residues
* `--num_coordinants`: List of possible mutating residues


License
--------
* Free software: BSD license

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
