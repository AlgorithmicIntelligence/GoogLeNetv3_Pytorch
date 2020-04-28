Implement the code from the [Paper](https://arxiv.org/pdf/1512.00567.pdf).

But the paper is ambiguous about the setting of the architecture, like how many filters in the inception models, so I refer the part of setting from [here](https://github.com/tensorflow/models/tree/master/research/inception). Besides, I didn't realize the regularization "Label Smoothing Regularization", because I think it just improved merely on the accuacy(improvement 0.2% on top1-error), it needs more experiments to prove that the regularization "indeed" helps on overfitting.

The accuracy and loss are shown as below:(100 classes)

<img src="https://github.com/AlgorithmicIntelligence/GoogLeNetv3_Pytorch/blob/master/README/Accuracy.png" width="450"><img src="https://github.com/AlgorithmicIntelligence/GoogLeNetv3_Pytorch/blob/master/README/Loss.png" width="450">


