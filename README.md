# Weakly-Supervised Text Classification


## Requirements

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

Python 3.6 is strongly recommended; using older python versions might lead to package incompatibility issues.

## Inputs

* A text corpus to be classified. It needs to be named as ```text.txt``` where each line is one document.
* Class names. You need to provide class names of each class in ```classes.txt``` where each line is one class name. (Note: You need to make sure all the class names appear in the corpus.) 
* (Optional) Ground truth labels. You can choose to provide ground-truth labels of all the documents to evaluate the performance/tune parameters. The ground truth labels are for evaluation purpose only and will **not** be used by the algorithm during training. Ground truth labels should be provided as class label ids in ```truth.txt```. (Note: class label ids start from ```0```, and the order should be consistent with the class name order in ```classes.txt```.)

Examples are given under ```./agnews/``` and ```./yelp/```.

## Outputs

The final results (document labels) will be written in ```./${dataset}/out.txt```, where each line is the class label id for the corresponding document. (Note: class label ids start from ```0```, and the order is consistent with the class name order in ```classes.txt```.)

Intermediate results (e.g. trained network weights, self-training logs) will be saved under ```./results/${dataset}/${model}/```.

## Running on a New Dataset

To execute the code on a new dataset, you need to 

1. Create a directory named ```${dataset}```.
2. Put the raw corpus ```text.txt``` under ```./${dataset}```.
3. Put the class name file ```classes.txt``` under ```./${dataset}```.
4. (Optional) Put the ground truth label file ```truth.txt``` under ```./${dataset}```.
5. Modify ```run.sh``` to appropriately indicate dataset directory and whether ground truth labels are available for evaluation.
6. Execute ```run.sh```.

