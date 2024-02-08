# bindNode23
Implementation is heavily based on the works of Littmann, et al. with bindEmbed21DL: 
Littmann M, Heinzinger M, Dallago C, Weissenow K, Rost B. Protein embeddings and deep learning predict binding residues for various ligand classes. *Sci Rep* **11**, 23916 (2021). https://doi.org/10.1038/s41598-021-03431-4
https://github.com/Rostlab/bindPredict/blob/master/bindEmbed21DL.py
bindNode23 is a method to predict whether a residue in a protein is binding to metal ions, nucleic acids (DNA or RNA), or small molecules. For the Graph Neural Net method, bindNode23 uses ProtT5 embeddings [2] as input to a 2-layer GNN. Since bindNode23 is based on single sequences, it can easily be applied to any protein sequence.

# Q&A
# When presentation? can be in October, also new studies solvable
# When roughly submission? beginning of september
# Batchsize is 406 proteins in both implementations, but only 203 get loaded (split)
# put 
# tables need captions. I think I do have them? look it up
# 0 in performance comparison on devset

# what state should the repository be in?
# maybe a jupyter notebook that runs on fasta file


# 
## Usage

`develop_bindNode23.py` provides the code to reproduce the bindEmbed21DL development (hyperparameter optimization, training, performance assessment on the test set).

All needed files and paths can be set in `config.py` (marked as TODOs).

## Data

### Development Set

The data set used for training and testing was extracted from BioLip [3]. The UniProt identifiers for the 5 splits used during cross-validation (DevSet1014), the test set (TestSet300), and the independent set of proteins added to BioLip after November 2019 (TestSetNew46) as well as the corresponding FASTA sequences and used binding annotations are made available in the `data` folder.

The trained models are available in the `trained_models` folder.

ProtT5 embeddings can be generated using [the bio_embeddings pipeline](https://github.com/sacdallago/bio_embeddings) [4]. To use them with `bindEmbed21`, they need to be converted to use the correct keys. A script for the conversion can be found in the folder `utils`.

## Requirements

bindEmbed21 is written in Python3. In order to execute bindEmbed21, Python3 has to be installed locally. Additionally, the following Python packages have to be installed:
- numpy
- scikit-learn
- torch
- pandas
- h5py
- pyg





## References
[1] Steinegger M, Söding J (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat Biotechnol 35.

[2] Elnaggar A, Heinzinger M, Dallago C, Rihawi G, Wang Y, Jones L, Gibbs T, Feher T, Angerer C, Bhowmik D, Rost B (2021). ProtTrans: towards cracking the language of life's code through self-supervised deep learning and high performance computing. bioRxiv.

[3] Yang J, Roy A, Zhang Y (2013). BioLip: a semi-manually curated database for biologically relevant ligand-protein interactions. Nucleic Acids Research, 41.

[4] Dallago C, Schütze K, Heinzinger M, Olenyi T, Littmann M, Lu AX, Yang KK, Min S, Yoon S, Morton JT, & Rost B (2021). Learned embeddings from deep learning to visualize and predict protein sets. Current Protocols, 1, e113. doi: 10.1002/cpz1.113

[5] Olenyi T, Marquet C, Heinzinger M, Kröger B, Nikolova T, Bernhofer M, Sändig P, Schütze K, Littmann M, Mirdita M, Steinegger M, Dallago C, & Rost B (2022). LambdaPP: Fast and accessible protein-specific phenotype predictions. bioRxiv
