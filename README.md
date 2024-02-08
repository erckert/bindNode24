# bindNode24
Implementation is heavily based on the works of Littmann, et al. with [bindEmbed21DL](https://github.com/Rostlab/bindPredict/blob/master/bindEmbed21DL.py) [1].
The implementation of bindNode24 builds on top of bindNode23 [2], a method to predict whether a residue in a protein is binding to metal ions, nucleic acids (DNA or RNA), or small molecules that has been developed by Franz Birkeneder under my supervision. For the Graph Neural Net method, bindNode23 uses ProtT5 embeddings [3] as input to a 2-layer GNN. Since bindNode23 is based on single sequences, it can easily be applied to any protein sequence.
 
## Table of Contents

- [Usage](#Usage)
- [Data](#Data)
- [Development Set](#Development Set)
- [Requirements](#Requirements)
- [Team](#Team)
- [License](#License)
- [Citation](#Citation)
- [References](#References)
 
### Usage

`develop_bindNode23.py` provides the code to reproduce the bindEmbed21DL development (hyperparameter optimization, training, performance assessment on the test set).

All needed files and paths can be set in `config.py` (marked as TODOs).

### Data

#### Development Set

The data set used for training and testing was extracted from BioLip [4]. The UniProt identifiers for the 5 splits used during cross-validation (DevSet1014), the test set (TestSet300), and the independent set of proteins added to BioLip after November 2019 (TestSetNew46) as well as the corresponding FASTA sequences and used binding annotations are made available in the `data` folder.

The trained models are available in the `trained_models` folder.

ProtT5 embeddings can be generated using [the bio_embeddings pipeline](https://github.com/sacdallago/bio_embeddings) [5]. To use them with `bindEmbed21`, they need to be converted to use the correct keys. A script for the conversion can be found in the folder `utils`.

### Requirements

bindEmbed21 is written in Python3. In order to execute bindEmbed21, Python3 has to be installed locally. Additionally, the following Python packages have to be installed:
- numpy
- scikit-learn
- torch
- pandas
- h5py
- pyg

### Team

Technical University of Munich - Rostlab

| Kyra Erckert | Franz Birkeneder | Burkhard Rost |
|:------------:|:-------------:|:-------------:|
|<img width=120/ src="https://github.com/erckert/bindNode24/raw/main/images/erckert.jpg"> |<img width=120/ src="https://github.com/erckert/bindNode24/raw/main/images/birkeneder.jpg">||<img width=120/ src="https://github.com/erckert/bindNode24/raw/main/images/rost.jpg">|


### License

The pretrained models and provided code are released under terms of the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)

### Citation

If you use this code or our pretrained models for your publication, please cite the original paper:
```
@ARTICLE
{XXXXXXX,
author={Erckert, Kyra and Birkeneder, Franz and Rost, Burkhard},
journal={XXXXXX},
title={bindNode24: Competitive binding residue prediction performance with 80% fewer parameters},
year={2024},
volume={},
number={},
pages={XXXXXXXX},
doi={XXXXXXXXX}}
```


### References

[1] Littmann M, Heinzinger M, Dallago C, Weissenow K, Rost B. Protein embeddings and deep learning predict binding residues for various ligand classes. *Sci Rep* **11**, 23916 (2021). https://doi.org/10.1038/s41598-021-03431-4

[2] Birkeneder F, Erckert K, Rost B (2023, July 26). Exploring Graph-based Ligand Binding: Introducing bindNode23 for Residue Classification [Poster Presentation]. ISMB/ECCB 2023 Lyon, France. 

[3] Elnaggar A, Heinzinger M, Dallago C, Rihawi G, Wang Y, Jones L, Gibbs T, Feher T, Angerer C, Bhowmik D, Rost B (2021). ProtTrans: towards cracking the language of life's code through self-supervised deep learning and high performance computing. bioRxiv.

[4] Yang J, Roy A, Zhang Y (2013). BioLip: a semi-manually curated database for biologically relevant ligand-protein interactions. Nucleic Acids Research, 41.

[5] Dallago C, Schütze K, Heinzinger M, Olenyi T, Littmann M, Lu AX, Yang KK, Min S, Yoon S, Morton JT, & Rost B (2021). Learned embeddings from deep learning to visualize and predict protein sets. Current Protocols, 1, e113. doi: 10.1002/cpz1.113