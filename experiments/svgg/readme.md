# Stein Variational Goal Generation for Hard Exploration Reinforcement Learning Problems


### Nicolas Castanet, Sylvain Lamprier, Olivier Sigaud

This is code for replicating our (https://arxiv.org/abs/2206.06719).

The launch commands used for the main experiments are in `svgg_experiments_commands.txt`, which call the `train_svgg.py` launch script. 

To run a SVGG agent use `--ag_curiosity svgg`. To use OMEGA use `--transition_to_dg`. The actual implementation of SVGG is the `SvgdEntropy` from `mrl.modules.curiosity`.



### Bibtex

```
@misc{https://doi.org/10.48550/arxiv.2206.06719,
  doi = {10.48550/ARXIV.2206.06719},
  
  url = {https://arxiv.org/abs/2206.06719},
  
  author = {Castanet, Nicolas and Lamprier, Sylvain and Sigaud, Olivier},
  
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Stein Variational Goal Generation For Reinforcement Learning in Hard Exploration Problems},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

```
