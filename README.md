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

* [8/16] **Joint Distribution Optimal Transportation for Domain Adaptation**
[arxiv](https://arxiv.org/abs/1705.08848)
<details><summary></summary>
() 2nd paper in the series of OT-->JDOT-->DEEPJDOT for DA<br>
() formulizing optimal transfortation which minimizes wasserstein distance between two joint distributions P(x, y)<br>
() target domain data is unlabeled y is replaced with f(x)<br>
</details>
<br>

* [8/20] **Fast, Better Training Trick - Random Gradient**
[arxiv](https://arxiv.org/abs/1808.04293)
<details><summary></summary>
(+) Simple idea: multiply the loss by a random variable~U(0,1) <br>
(-) Marginal improvement<br>
</details>
<br>

* [8/20] **Droupout is a special case of the stochastic delta rule: faster and more accurate deep learning**
[arxiv](https://arxiv.org/abs/1808.03578)
<details><summary></summary>
(+) Simple idea: adding a gaussian noise to the gradient (with variances proportional to the gradient)<br>
(+) Substantial improvement on CIFAR-10/100<br>
(-) Poor expressions <br>
(-) Code is provided but looks different from the paper's description<br>
</details>
<br>

* [8/22] **Deep Randomized Ensembles for Metric Learning**
[arxiv](https://arxiv.org/abs/1808.04469)
<details><summary></summary>
() radomly merging multiple classes into one<br>
(+) R@1=80.5 and 89.8 for CUB200 and CARS196 with an ensemble of 48 models<br>
</details>
<br>

* [8/23] **Reversed Active Learning based Atrous DenseNet for Pathological Image Classification**
[arxiv](https://arxiv.org/abs/1807.02420)
<details><summary></summary>
() Automatic removal of noisy data in the training set<br>
</details>
<br>

* [8/24] **Training confidence-calibrated classfiers for detecting out-of-distribution samples**
[arxiv](https://arxiv.org/abs/1711.09325)
<details><summary></summary>
(+) Well-written, comprehensive explanation and experiments<br>
(+) Solved the problem with simple but effective approches<br>
(-) Missed the combined performance of confidence+joint-confidence loss<br>
</details>
<br>

* [8/29] **DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation**
[arxiv](https://arxiv.org/abs/1803.10081)
<details><summary></summary>
() While similar to JDOT, it learns feature extractors g during optimization<br>
() Deals with large dataset by adapting mini-batch based optimization<br>
</details>
<br>

* [8/30] **Part-Aligned Bilinear Representations for Person Re-identification**
[arxiv](https://arxiv.org/abs/1804.07094)
<details><summary></summary>
() two stream architecture: appearance network and part network<br>
() part network is initialized with OpenPose network<br>
() two features are combined with bilinear pooling<br>
</details>
<br>

* [8/31] **Learning to Support: Exploiting Structure Information in Support Sets for One-Shot Learning**
[arxiv](https://arxiv.org/abs/1808.07270)
<details><summary></summary>
(+) Simple idea: generate multiple representative features by combining multiple feature<br>
(+) Another evidence of the effect of Cycling Learning Rate<br>
(-) Exaggerating rhetoric<br>
</details>
<br>

* [9/3] **Deeply-Learned Part-Aligned Representations for Person Re-Identification**
[arxiv](https://arxiv.org/abs/1707.07256)
<details><summary></summary>
() K branches with attention module<br>
() similar architecture with ABE-M without div loss <br>
() global pooling following right after attention<br>
</details>
<br>

* [9/7] **Mining on Manifolds: Metric Learning without Labels**
[arxiv](https://arxiv.org/abs/1803.11095)
<details><summary></summary>
() Unsupervised metric learning (w/ pretrained network)<br>
() Using Euclidean and manifold distance to mine positive and negative samples<br>
() Manifold distance is calculated via Random Walk<br>
</details>
<br>

* [9/10] **Learning to Generate and Reconstruct 3D Meshes with only 2D Supervision**
[arxiv](https://arxiv.org/abs/1807.09259)
<details><summary></summary>
() encoder-decoder framework <br>
(+) train of 3d reconstruction only with images without annotations<br>
(-) limited synthesized test setting<br>
</details>
<br>

* [9/13] **CBAM: Convolutional Block Attention Module**
[arxiv](https://arxiv.org/abs/1807.06521)
<details><summary></summary>
() attention as a building block of general purpose architecture<br>
(-) simple, shallow main body<br>
(+) extensive experiments for design choise and benchmark performances<br>
</details>
<br>

* [9/17] **Probabilistic Binary Neural Networks**
[arxiv](https://arxiv.org/abs/1809.03368)
<details><summary></summary>
(+) Simple approach of learning stochastic weights and binarized activation<br>
(+) Enables batch normalization and ensemble of sampled weights<br>
(-) Performance of MAP weights is simliar to the pervious binarized network<br>
<br>
</details>
<br>

* [9/19] **Orientation Invariant Feature Embedding and Spatial Temporal Regularization for Vehicle Re-identification**
[Link](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Orientation_Invariant_Feature_ICCV_2017_paper.pdf)
<details><summary></summary>
(-) additional annotation for keypoint detecting<br>
(+) prediction for visible sides<br>
(+) orientation aware feature embedding<br>
</details>
<br>

* [9/21] **Pairwise Confusion for Fine-Grained Visual Classification**
[arxiv](https://arxiv.org/abs/1705.08016)
<details><summary></summary>
(+) Regularization technique for Fine Grained Visual Classification<br>
(+) Good Quantative Analysis with baseline using saliency maps from Grad-CAM<br>
(+) Good mathematical justfication of proposed confusion metric by deriving upper and lower bounds<br>
</details>
<br>

* [9/27] **Detecting and Recognizing Human-Object Interactions**
[arxiv](https://arxiv.org/abs/1704.07333)
<details><summary></summary>
() a reletively new task of detecting human-object interactions<br>
() extracting <human, verb, object> triplets<br>
() modify Faster-RCNN with additional head for action classifier and object location regresser for each action<br>
</details>
<br>

* [10/1] **Exponential Discriminative Metric Embedding in Deep Learning**
[Link](https://www.sciencedirect.com/science/article/pii/S0925231218301759)
<details><summary></summary>
() another loss which can be added to softmax loss (IE loss)<br>
() quite similar to center loss<br>
</details>
<br>

* [10/4] **Deep Metric Learning and Image Classification with Nearest Neighbour Gaussian Kernels**
[arxiv](https://arxiv.org/abs/1705.09780)
<details><summary></summary>
(+) Combines metric learning loss and classification loss into a single loss using gaussian kernel and approximate nearest neighbour search, Significant improvement in image retrieval tasks<br>
(-) Thorough analysis is not given, 4 page restricition<br>
</details>
<br>

* [10/5] **Large Scale GAN Training For High Fidelity Natual Image Synthesis**
[arxiv](https://arxiv.org/abs/1809.11096)
<details><summary></summary>
(+) Concise summary of the latest Image GAN models<br>
(+) Large batch & wide generator with relaxed orthognal regularizer for discriminator<br>
(+) Experimental results of combined approaches of SOTAs which we can not get easily<br>
(-) No theoretical or fundamental progress<br>
</details>
<br>

* [10/8] **Rethinking the Value of Network Pruning**
[arxiv](https://arxiv.org/abs/1810.05270)
<details><summary></summary>
() Pruned architecture can be trained from the scratch (accuracies are similar to fine-tuning)<br>
</details>
<br>

* [10/11] **Taiming VAEs**
[arxiv](https://arxiv.org/abs/1810.00597)
<details><summary></summary>
(+) Inversion of the goal of the VAE(variational auto encoder) with theoretical analysis<br>
(+) Achieving much lower KL divergence of the posterior in the latent dimension<br>
(+) Mitigating the "hole" problem of VAE and gettting a generative model <br>
(-) No practical implication for now<br>
<br>
</details>
<br>

* [10/12] **Confidence Calibration in Deep Neural Networks through Stochastic Inferences**
[arxiv](https://arxiv.org/abs/1809.10877)
<details><summary></summary>
() Using stochastic inference to weight two cross entropy in loss; ce with gt and ce with uniform distribution<br>
(+) calibrated confidence with a single forward pass<br>
</details>
<br>

* [10/15] **FD-GAN: Pose-guided Feature Distilling GAN for Robust Person Re-identification**
[arxiv](https://arxiv.org/abs/1810.02936)
<details><summary></summary>
(+) Using the generative model for a regularizer, this improves the SOTA re-identification accuracy by 5~9%<br>
(+) One of a few successful cases that a generative model actually helps <br>
(-) No through analysis of what's going on under the hood<br>
(-) genarative model + 3 discriminators in addtion to the primary loss, which would reduce the batch size.<br>
</details>
<br>

* [10/18] **Deep Networks with Stochastic Depth**
[arxiv](https://arxiv.org/abs/1603.09382)
<details><summary></summary>
(+) better than resnet baseline in CIFAR10, CIFAR100, and SVHN (not in ImageNet)<br>
(+) first to train deeper (>1000 layers) network to be better than less deep model (1202 layers > 101 layers)<br>
</details>
<br>

* [10/22] **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles**
[arxiv](https://arxiv.org/abs/1612.01474)
<details><summary></summary>
(+) Non-bayesian approach for uncertainty estimation<br>
() 3 components: proper scoring rule as loss, adversarial training, ensemble<br>
(+) better than MC-dropout<br>
</details>
<br>

