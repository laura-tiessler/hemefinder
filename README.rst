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

.. code-block:: bash

    > conda create -n {name} python=3.9
    > conda activate {name}
    > pip install git+https://github.com/laura-tiessler/hemefinder/
    > pip install pyKVFinder
    > conda install -c anaconda numpy 
    > conda install -c jmcmurray json 