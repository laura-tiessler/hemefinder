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


Usage
--------

The software is run from the terminal and if defaulf parameters are used only the input PDB is needed.

* Example with PDB
  
```bash
    > hemefinder target_name.pdb
```

* Example of downloading directy from PDB server:

```bash
    > hemefinder target_name
```

The main parameters that can be tuned for calculations are the following:

* `--output`: Directory where outputs should be stored. 
* `--coordinators`: List of possible coordinating residues
* `--mutations`: List of possible mutating residues
* `--num_coordinants`: List of possible mutating residues


Other parameters can also be tuned, but it is not recommended:

* `--output`: Directory where outputs should be stored. 


Example:
--------

The software print de results in the terminal, but also generates two output files. 

1. A json file that contains the possible heme coordinating residues and its corresponding scores, sorted by score. 
2. A PDB file that contains the centroid of the coordinating probes, all the probes that make up the ellipsoid and the coordinating probes. Each result is represented by different atom types (Centroid = He, ellipsoid = Xe and coordinating probes = Ne).

![Output](/docs/Tutorial_heme.png)

Example:
--------

Example of search for heme binding site with minimum of 2 coordinating His residues.

    > hemefinder 1dkh --coordinators 1 --output results_1dkh


License
--------

* Free software: BSD license

Credits
-------
This package was created with Cookiecutter_ and 
the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
