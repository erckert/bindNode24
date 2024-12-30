# bindNode24
Implementation is heavily based on the works of Littmann, et al. with [bindEmbed21DL](https://github.com/Rostlab/bindPredict/blob/master/bindEmbed21DL.py) [1].
The implementation of bindNode24 builds on top of bindNode23 [2], a method to predict whether a residue in a protein is binding to metal ions, nucleic acids (DNA or RNA), or small molecules that has been developed by Franz Birkeneder under my supervision. For the Graph Neural Net method, bindNode24 uses ProtT5 embeddings [3] as input to a 2-layer GNN.
 
## Table of Contents

- [Usage](#Usage)
- [Data](#Data)
- [Requirements](#Requirements)
- [Team](#Team)
- [License](#License)
- [Citation](#Citation)
- [References](#References)
 
### Usage


All needed files and paths can be set in `config.ini`.
Pretrained model dictionaries for bindNode24 are available in `trained_model_dict/bindNode24`.

#### Run code

To run the code execute:
```
python src/main.py
```

If you want to use a different `.ini` file use 
```
python src/main.py -f FILEPATH
```
instead.

Warning: If PDB structures or DSSP files are unavailable, the corresponding entries in the dataset will be replaced with 0. A notification will be printed in the command line. Predictions generated in this way may be very unreliable. 

### Data

#### Development

The data set used for training and testing was extracted from BioLip [4]. The UniProt identifiers for the 5 splits used during cross-validation (DevSet1014), and the test set (TestSet300) as well as the corresponding FASTA sequences, embeddings, predicted structures, precomputed DSSP features and used binding annotations are made available in the `dataset` folder.

The weights for pretrained models are available in the `trained_model_dicts\bindNode24` folder.

ProtT5 embeddings can be generated using [the bio_embeddings pipeline](https://github.com/sacdallago/bio_embeddings) [5]. Embeddings may require to be reformatted to use the correct protein ids as keys

#### Output files

##### Predictions
The following example shows the output file formatting for predicted proteins if `write_ri=true` in the config file:
```
Position	Metal.RI	Metal.Class	Nuc.RI	Nuc.Class	Small.RI	Small.Class	Any.Class
1	9.000	nb	9.000	nb	8.000	nb	nb
2	9.000	nb	9.000	nb	9.000	nb	nb
3	9.000	nb	9.000	nb	9.000	nb	nb
4	9.000	nb	9.000	nb	8.000	nb	nb
5	9.000	nb	9.000	nb	5.000	nb	nb
6	9.000	nb	9.000	nb	1.000	nb	nb
```
"nb" indicates that the position was predicted as non-binding for the specific binding type and "b", that the position was predicted to be binding. 
RI indicates the reliability index for the corresponding predictions. Reliability indexes are computed as introduced by Littmann et al [1]. 
Columns are tab separated for easy further processing.

If `write_ri=false` the predictions will contain the predicted probabilities instead of the reliability index:
```
Position	Metal.Prob	Metal.Class	Nuc.Prob	Nuc.Class	Small.Prob	Small.Class	Any.Class
...
28	0.000	nb	0.000	nb	0.001	nb	nb
29	0.265	nb	0.001	nb	0.198	nb	nb
30	0.009	nb	0.010	nb	0.276	nb	nb
31	0.017	nb	0.006	nb	0.378	nb	nb
32	0.002	nb	0.003	nb	0.651	b	b
33	0.024	nb	0.049	nb	0.564	b	b
34	0.009	nb	0.004	nb	0.166	nb	nb
...
```
Probabilities only show 3 decimal positions.

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
|<img width=120/ src="https://github.com/erckert/bindNode24/raw/main/images/erckert.jpg"> |<img width=120/ src="https://github.com/erckert/bindNode24/raw/main/images/birkeneder.jpg">|<img width=120/ src="https://github.com/erckert/bindNode24/raw/main/images/rost.png">|


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