# Daily Insight Sharing (DIS) 2018 

##### List of papers that we discussed in our reading group at Visual Understanding Lab
Papers are mostly related to Image Retrieval, Deep Metric Learning, Image Recognition, Generative Adversarial Networks and applications

* [3/14] **Why M Heads are Better than One: Training a Diverse Ensemble of Deep Networks**
[arxiv](https://arxiv.org/abs/1511.06314)
<details><summary></summary>
(+) Sharing lower layers is a little better than full ensemble. <br>
(+) Can learn a lesson from their failure. Don't mess with the independency of ensemble<br>
(+) Interesting metric called oracle accruacy of ensemble (T if any learner is correct)<br>
(-) Not accepted. A little improvement of performance<br>
</details>
<br>

* [3/14] **Metric Learning-based Generative Adversarial Network**
[arxiv](https://arxiv.org/abs/1711.02792)
<details><summary></summary>
(+) Try to combining metric learning & GAN which is related to our work<br>
(-) Applying metric learning for better GAN, while we need GAN for metric learning<br>
</details>
<br>

* [3/19] **Averaging weights leads to wider optima and better generalization**
[arxiv](https://arxiv.org/abs/1803.05407)
<details><summary></summary>
(+) Well citation and summary of previous findings about SGD and loss function<br>
(-) Limited understanding of super-high dimensional space by 1D/2D projection<br>
</details>
<br>

* [3/22] **Meta-Learning for Semi-Supervised Few-Shot Classification**
[arxiv](https://arxiv.org/abs/1803.00676)
<details><summary></summary>
(+) First paper for semi-supervised meta-learning (metric learning)<br>
(+) Semi-supervised setting with distractor is realistic<br>
(+) Benchmark for semi-supervised metric learning<br>
(-) The method is somewhat heuristic<br>
</details>
<br>

* [3/26] **Directional Statistics-based Deep Metric Learning for Image Classification and Retrieval**
[arxiv](https://arxiv.org/abs/1802.09662)
<details><summary></summary>
(+) New approach for searching each class meaning direction in metric learning<br>
(+) Well-organized retrieval experiment using von-mises-fisher distribution<br>
(-) Easy to understand, but the performance of SOP retrieval performance isn't high<br>
</details>
<br>

* [3/26] **Don't Decay the Learning Rate, Increase the Batch Size**
[arxiv](https://arxiv.org/abs/1711.00489)
<details><summary></summary>
(+) "Increasing batchsize A times" is equivalent to "decreasing learning rate A times"<br>
(+) Possible gain in speed<br>
(-) no improve in the accuracy<br>
() The title says it all, it's more like an empirical paper<br>
</details>
<br>

* [3/28] **Unsupervised Representation Learning by Predicting Image Rotations**
[arxiv](https://arxiv.org/abs/1803.07728)
<details><summary></summary>
(+) Simple self-supervised learning with 4 ouptuts (0, 90, 180, 270 degree rotation)<br>
(-) Used old structure (Alexnet)<br>
(+) Interesting claims ( 90 degrees do not generate artifacts, well-poseness is necessary)<br>
</details>
<br>

* [3/28] **A Bayesian Perspective on Generalization and Stochastic Gradient Descent**
[arxiv](https://arxiv.org/abs/1710.06451)
<details><summary></summary>
() They proposed to use marginal likelihood (evidence) as a measure of generalization ability<br>
() They interpriate SGD as a stochastic differential equations and derived the linear relationship between learning rate and batch size in a noise scale<br>
</details>
<br>

* [3/29] **Group Normalization**
[arxiv](https://arxiv.org/abs/1803.08494)
<details><summary></summary>
(+) Comprehensive illustrations of various normalizations<br>
(+) GN(Group normalization) is a little worse than BN(Batch Normalization) but better than others.<br>
(+) GN is feasible with small batch settings such as detection models.<br>
(-) No need to read if you are using enough batch size or familiar with BN.<br>
</details>
<br>

* [4/2] **Attention-based Deep Multiple Instance Learning**
[arxiv](https://arxiv.org/abs/1802.04712)
<details><summary></summary>
() proposed to use attention for MIL pooling<br>
(+) main body of the paper is well-written<br>
(-) performance is not very good and limited to small training data case<br>
</details>
<br>

* [4/4] **Bayesian Gradient Descent: Online Variational Bayes Learning with Increased Robustness to Catastrophic Forgetting and Weight Pruning**
[arxiv](https://arxiv.org/abs/1803.10123)
<details><summary></summary>
(+) Concise introduction to Bayesian optimization<br>
(+) More generic solution to the catastrophic forgetting<br>
(-) A worse than the previous approach<br>
</details>
<br>

* [4/5] **Large-Scale Image Retrieval with Attentive Deep Local Features**
[arxiv](https://arxiv.org/abs/1612.06321)
<details><summary></summary>
() Attended local features, pyramid image, RANSAC verification for retieving landmark images<br>
</details>
<br>

* [4/6] **Optimizing the letent space of generative networks**
[arxiv](https://arxiv.org/abs/1707.05776)
<details><summary></summary>
(+) Use learnable variables for the latent space, without any encoder, only train the decoder<br>
(+) Non-parametric latent space, with simple optimization technique<br>
(+) Room to improve the performance<br>
(-) Inconclusive results, rejected from ICLR2018<br>
</details>
<br>

* [4/9] **CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise**
[arxiv](https://arxiv.org/abs/1711.07131)
<details><summary></summary>
(+) Nice and practical problem setting<br>
(-) less convincing solution (heuristic method)<br>
(-) lack of details of dataset, ablation study<br>
</details>
<br>

* [4/11] **Qualitatively Characterizing Neural Network Optimization Networks**
[arxiv](https://arxiv.org/abs/1412.6544)
<details><summary></summary>
() The path a network takes from initialzation to solution is smooth<br>
() No local minimas encountered during the path in popular models<br>
</details>
<br>

* [4/16] **Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking**
[arxiv](https://arxiv.org/abs/1803.11285)
<details><summary></summary>
(+) Revise oxford5k, paris6k dataset + 1M filtered distractor without false negative<br>
(+) Well orgarized experiments for comparing state of the arts<br>
</details>
<br>

* [4/18] **Cognitive Psychology for Deep Neural Networks: A Shape Bias Case Study**
[arxiv](https://arxiv.org/abs/1706.08606)
<details><summary></summary>
() Applied methodology of cognitive psychology for interpreting neural network's behavior<br>
() Inception model trained with ImageNet has a shape bias<br>
() different randomly initialized models shows various strength of shape bias in spite of similar classfication accruacy<br>
</details>
<br>

* [4/30] **Borrowing Treasures from the Wealthy: Deep Transfer Learning through Selective Joint Fine-tuning**
[arxiv](https://arxiv.org/abs/1702.08690)
<details><summary></summary>
(+) Subsampling training data (of source domain) is better than using entire one<br>
</details>
<br>

* [5/2] **PixelCNN++: A PixelCNN Implementation with Discretized Logistic Mixture Likelihood and Other Modifications,**
[arxiv](https://arxiv.org/abs/1701.05517)
<details><summary></summary>
(+) Good modelling of discrete pixel values with logistic distribution<br>
(+) Performance improvement of PixelCNN<br>
(+) Full code is uploaded<br>
(-) Incremental modification and inherites some weakness of PixelCNN, e.g.) sampling time<br>
</details>
<br>

* [5/16] **Realistic Evaluation of Deep Semi-Supervised Learning Algorithms**
[arxiv](https://arxiv.org/abs/1804.09170)
<details><summary></summary>
(+) integrated implementation for test bed with previous methods<br>
(+) optimized baseline (fully supervised learning / transfer learning)<br>
(+) realistic scenario of semi-supervised learning (distribution mismatch between labeled and unlabeled data / limited amount of validation data)<br>
</details>
<br>

* [5/17] **Label Refinery: Improving ImageNet Classification through Label Progression**
[arxiv](https://arxiv.org/abs/1805.02641)
<details><summary></summary>
() Using dynamically generated label from label refinery model to train better model<br>
(-) very similar to teacher-student training<br>
(+) propose to use extreme data augmentation, adversariel examples in training<br>
(+) through experiments<br>
</details>
<br>

* [5/24] **Born Again Neural Networks**
[arxiv](https://arxiv.org/abs/1805.04770)
<details><summary></summary>
(+) First paper for teacher student training with identical architectures<br>
</details>
<br>

* [5/30] **Do Better Imagenet Models Transfer Better?**
[arxiv](https://arxiv.org/abs/1805.08974)
<details><summary></summary>
(+) Thourough survey of different architectures and different datasets for transfer learning<br>
(+) Better statistical method for comparison: Instead of directly comparing accuracy on Imagenet and transfer datasets, it compares the additive change in logit-transformed accuracy<br>
() Shows better model on Imagenet perform better on transfer datasets too on finetuning almost always<br>
</details>
<br>

* [5/31] **Spatial Transformer Introspective Neural Network**
[arxiv](https://arxiv.org/abs/1805.06447)
<details><summary></summary>
(+) Improvement over baseline discriminative models, especially in Few-shot learning problem<br>
(+) The original introspective model is intriguing, "Learning Genetative Models via Discriminative Approaches"<br>
(-) Marginal improvement over introspective models<br>
</details>
<br>

* [6/1] **How Does Batch Normalization Help Optimization? (No, It Is Not About Internal Covariate Shift)**
[arxiv](https://arxiv.org/abs/1805.11604)
<details><summary></summary>
() Batch-norm is not reducing covariance shift<br>
() It smooth the loss landscape<br>
</details>
<br>

* [6/4] **The Conditional Analogy GAN: Swapping Fashion Articles on People Images**
[arxiv](https://arxiv.org/abs/1709.04695)
<details><summary></summary>
(+) interesting problem: Given the image of fashion article alone, generating an image with human model<br>
(-) not good results nor experimental section<br>
</details>
<br>

