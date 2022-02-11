# Continual Learning with Invertible Generative Models

This repository contains a PyTorch implementation of the paper: 

### Continual Learning with Invertible Generative Models
<!--[Pseudo-Rehearsal for Continual Learning withNormalizing Flows](https://arxiv.org/abs/2003.00952)\ -->
[Jary Pomponi](https://www.semanticscholar.org/author/Jary-Pomponi/1387980523), [Simone Scardapane](http://ispac.diet.uniroma1.it/scardapane/), [Aurelio Uncini](http://www.uncini.com/)

### Abstract
Catastrophic forgetting (CF) happens whenever a neural network overwrites past knowledge while being trained on new tasks. 
Common techniques to handle CF include regularization of the weights (using, e.g., their importance on past tasks), and rehearsal strategies, where the network is constantly re-trained on past data. Generative models have also been applied for the latter, in order to have endless sources of data. In this paper, we propose a novel method that combines the strengths of regularization and generative-based rehearsal approaches. Our generative model consists of a normalizing flow (NF), a probabilistic and invertible neural network, trained on the internal embeddings of the network. By keeping a single NF throughout the training process, we show that our memory overhead remains constant. In addition, exploiting the invertibility of the NF, we propose a simple approach to regularize the network's embeddings with respect to past tasks. We show that our method performs favorably with respect to state-of-the-art approaches in the literature, with bounded computational power and memory overheads.
### Main Dependencies
* pytorch==1.4.0
* python=3.6
* cudatoolkit=10.1
* torchvision==0.5.0
* yaml

The complete list can be found in the environment file env.yml. 

### Training
The folder './files/' contains some yaml files that can be passed as argument to the main.py script. 
The main.py produces a path in the save_path indicated by the yaml file. The resulting folder contains: 

* One folder for each experiment. Each folder contains:
    * models [if save=yes]: a folder which contains the model checkpoints, one for each task.
    * plots [if method == 'prer']: a folder that contains the plots produced by the method.
    * results [if save=yes]: the serialized results obtained on the tasks. 
    * final_results.pkl: the final results serialized using pickle. 
    * experiment.log: the log file containing information about the current experiment. 
* final_score.log: the final log file which reports the final scores obtained.  

Please refer to the yaml files to understand how they can be formatted, and to the methods to understand the parameters that can be used. 

<!--[
### Cite


Please cite our work if you find it useful:

```
@article{pomponi2020pseudo,
  title={Pseudo-Rehearsal for Continual Learning with Normalizing Flows},
  author={Pomponi, Jary and Scardapane, Simone and Uncini, Aurelio},
  journal={arXiv preprint arXiv:2007.02443},
  year={2020}
}
```
]-->