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

* [6/4] **AutoAugment: Learning Augmentation Policies from Data**
[arxiv](https://arxiv.org/abs/1805.09501)
<details><summary></summary>
(+) Good way to use large number of augmentation techniques simultaneously, show significant improvement<br>
(-) Search algorithm procedure not provided in detail<br>
</details>
<br>

* [6/8] **Knowledge Distillation with Adversarial Samples Supporting Decision Boundary**
[arxiv](https://arxiv.org/abs/1805.05532)
<details><summary></summary>
(+) combining two ideas: adverarial attack and knowledge distillation<br>
(+) using samples on the decision boundary for knowledge distillation<br>
</details>
<br>

* [6/14] **why do deep convolutional networks generalize so poorly to small image transformations?**
[arxiv](https://arxiv.org/abs/1805.12177)
<details><summary></summary>
() CNNs are very sensitive to translation and scale change<br>
() Architecture does not explicitly designed for it due to subsampling<br>
() Data has photographer's bias<br>
</details>
<br>

* [6/20] **Deep Adversarial Metric Learning**
[Link](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)
<details><summary></summary>
() Generating adversarial sample for negative examples<br>
</details>
<br>

* [6/20] **Large-Scale Distance Metric Learning With Uncertainty**
[arxiv](https://arxiv.org/abs/1805.10384)
<details><summary></summary>
() Generating and using latent examples without uncertainty for training<br>
</details>
<br>

* [6/25] **Deep Learning is Robust to Massive Label Noise**
[arxiv](https://arxiv.org/abs/1705.10694)
<details><summary></summary>
(+) Extensive experimental results<br>
(-) Limited setting<br>
() with noise, larger batchsize helps<br>
</details>
<br>

* [6/27] **Training Convolutional Networks with Noisy Labels**
[arxiv](https://arxiv.org/abs/1406.2080)
<details><summary></summary>
() ICLR 2015 workshop<br>
() Using linear layer to model noise distribution<br>
(-) class-level noise modeling<br>
</details>
<br>

* [7/4] **Training Deep Neural Networks on Noisy Labels with Bootstrapping**
[arxiv](https://arxiv.org/abs/1412.6596)
<details><summary></summary>
(+) Realistic testbed for noisy dataset by using subjective and ambiguous labelling of human emotion classification without injecting additional noise.<br>
</details>
<br>

* [7/5] **Knowledge Concentration: Learning 100K Object Classifiers in a Single CNN**
[arxiv](https://arxiv.org/abs/1711.07607)
<details><summary></summary>
(+)<br>
</details>
<br>

* [7/16] **Beyond One-hot encoding lower dimensional terget embedding**
[arxiv](https://arxiv.org/abs/1806.10805)
<details><summary></summary>
(+) Good survey of the encoding approaches<br>
(+) An alternative to the clustering approaches<br>
(-) Disappointing experimental results<br>
</details>
<br>

* [7/18] **Creating Capsule Wardrobes from Fashion Images**
[arxiv](https://arxiv.org/abs/1712.02662)
<details><summary></summary>
() After attribution prediction (multi-task classification), it apply topic model (Correlated Topic Models (CTM)) for modelting generative models. They are used to modeling compatibility and versatibility.<br>
(+) Unsupervised method to learn compatibility and style (style for versatibility)<br>
</details>
<br>

* [7/20] **Semantically decomposing the latent space of generative adversarial network
Generating a Fusion Image: One's Identity and Another's Shape**
[arxiv](https://arxiv.org/abs/1705.07904)
<details><summary></summary>
(+) Comparator-based discriminator<br>
<br>
</details>
<br>

* [7/23] **Learning the Latent "Look": Unsupervised Discovery of a Style-Coherent Embedding from Fashion Images**
[arxiv](https://arxiv.org/abs/1707.03376)
<details><summary></summary>
(+) Unsupervised learning for style discovery<br>
() Using polylingual topic model assuming multiple clothes in an outfit as translations of same document in different languages<br>
</details>
<br>

* [7/25] **Deep Clustering for Unsupervised Learning of Visual Features**
[arxiv](https://arxiv.org/abs/1807.05520)
<details><summary></summary>
(+) Unsupervised learning can learn competitive deep visual representations on ImageNet<br>
(-) Still, worse than supervised learning<br>
</details>
<br>

* [7/26] **Learning Type-Aware Embeddings for Fashion Compatibility**
[arxiv](https://arxiv.org/abs/1803.09196)
<details><summary></summary>
(+) Interesting way to handle improper triangle issue (improper transitivity)<br>
(-) Results are not thorough, doesnt cover current state-of-the-art results<br>
</details>
<br>

* [8/3] **The Devil of Face Recognition is in the Noise**
[arxiv](https://arxiv.org/abs/1807.11649)
<details><summary></summary>
() It shows clearer dataset gives better results<br>
() Different noise level prefers different loss functions<br>
() It presents noise-free public dataset<br>
</details>
<br>

* [8/6] **Spectral Normalization for Generative Adversarial Networks**
[arxiv](https://arxiv.org/abs/1802.05957)
<details><summary></summary>
(+) neat theoretical analysis and practical improvement<br>
(+) Good comparison with other regularizers<br>
(-) Leaving some questions about the choices of the paper regarding the alternative designs based on the claims<br>
</details>
<br>

* [8/6] **The GAN Landscape: Losses, Architectures, Regularization, and Normalization**
[arxiv](https://arxiv.org/abs/1807.04720)
<details><summary></summary>
(+) Comprehensive empirical comparison of the latest GAN models<br>
(+) Disciplined way of experiments<br>
(-) No theoretical analysis<br>
</details>
<br>

* [8/6] **Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition**
[arxiv](https://arxiv.org/abs/1806.05372)
<details><summary></summary>
Very similar to work on ABE<br>
(+) Efficient way to implement attention module using squeeze excitation<br>
(+) Divergence in attention using triplet loss, thorough analysis for all types of triplets possible with attention<br>
(-) No significant improvement in results<br>
</details>
<br>

* [8/8] **Unified Deep Supervised Domain Adaptation and Generalization**
[arxiv](https://arxiv.org/abs/1709.10190)
<details><summary></summary>
() Classification loss + semantic alignment loss + seperation loss<br>
() later two results in contrastive loss in penultimate layer<br>
</details>
<br>

* [8/13] **Scatter Component Analysis: A Unified Framework for Domain Adatation and Domain Generalization**
[arxiv](https://arxiv.org/abs/1510.04373)
<details><summary></summary>
() Finding feature transform h which minimizes different of P(h(x)) in multiple domains<br>
(+) generalizes previous algorithms<br>
</details>
<br>

* [8/13] **Domain Generalization via Conditional Invariant Representations**
[Link](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16595)
<details><summary></summary>
() Finding feature transform h which minimizes different of P(h(x)|y) in multiple domains<br>
(-) incremental work of above work (from different research group)<br>
</details>
<br>

