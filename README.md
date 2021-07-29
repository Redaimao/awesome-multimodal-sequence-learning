# Awesome-Multimodal-Sequence-Learning
Reading list for multimodal sequence learning


## Table of Contents

* [Survey Papers](#survey-papers)
* [Research Areas](#core-areas)
  * [Representation Learning](#representation-learning)
  * [Multimodal Fusion](#multimodal-fusion)
  * [Analysis of Multimodal Models](#analysis-of-multimodal-models)    
  * [Multimodal Pretraining](#multimodal-pretraining)
  * [Self-supervised Learning](#self-supervised-learning)
  * [Generative Multimodal Models](#generative-multimodal-models)
  * [Multimodal Adversarial Attacks](#multimodal-adversarial-attacks)
* [Research Tasks](#research-tasks)   
  * [Sentiment and Emotion Analysis](#sentiment-and-emotion-analysis)
  * [Trajectory and Motion Forecasting](#trajectory-and-motion-forecasting)
* [Datasets](#datasets)  
* [Tutorials](#tutorials)

## Survey Papers

[Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), TPAMI 2019

[Multimodal Intelligence: Representation Learning, Information Fusion, and Applications](https://arxiv.org/abs/1911.03977), arXiv 2019

[Deep Multimodal Representation Learning: A Survey](https://ieeexplore.ieee.org/abstract/document/8715409), arXiv 2019

[Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/6.5538), TPAMI 2013


## Research Areas

### Representation Learning

[Parameter Efficient Multimodal Transformers for Video Representation Learning](https://openreview.net/pdf?id=6UdQLhqJyFD), ICLR 2021

[Viewmaker Networks: Learning Views for Unsupervised Representation Learning](https://openreview.net/pdf?id=enoVQWLsfyL), ICLR 2021, [[code]](//github.com/alextamkin/viewmaker)

[Representation Learning for Sequence Data with Deep Autoencoding Predictive Components](https://openreview.net/pdf?id=Naqw7EHIfrv), ICLR 2021

[Improving Transformation Invariance in Contrastive Representation Learning](https://openreview.net/pdf?id=NomEDgIEBwE), ICLR 2021

[Active Contrastive Learning of Audio-Visual Video Representations](https://openreview.net/pdf?id=OMizHuea_HB), ICLR 2021

[Parameter Efficient Multimodal Transformers for Video Representation Learning](https://openreview.net/pdf?id=6UdQLhqJyFD), ICLR 2021

[i-Mix: A Domain-Agnostic Strategy for Contrastive Representation Learning](https://openreview.net/pdf?id=T6AxtOaWydQ), ICLR 2021

[Seq2Tens: An Efficient Representation of Sequences by Low-Rank Tensor Projections](https://openreview.net/pdf?id=dx4b7lm8jMM), ICLR 2021


[Adaptive Transformers for Learning Multimodal Representations](https://arxiv.org/abs/2005.07486), ACL 2020

[Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf), arXiv 2020 [[blog]](https://openai.com/blog/clip/) [[code]](https://github.com/openai/CLIP)

[12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/abs/1912.02315), CVPR 2020 [[code]](https://github.com/facebookresearch/vilbert-multi-task)

[Watching the World Go By: Representation Learning from Unlabeled Videos](https://arxiv.org/abs/2003.07990), arXiv 2020




[Multi-Head Attention with Diversity for Learning Grounded Multilingual Multimodal Representations](https://arxiv.org/abs/1910.00058), EMNLP 2019

[Visual Concept-Metaconcept Learning](https://papers.nips.cc/paper/8745-visual-concept-metaconcept-learning.pdf), NeurIPS 2019 [[code]](http://vcml.csail.mit.edu/)

[ViCo: Word Embeddings from Visual Co-occurrences](https://arxiv.org/abs/1908.08527), ICCV 2019 [[code]](https://github.com/BigRedT/vico)

[Unified Visual-Semantic Embeddings: Bridging Vision and Language With Structured Meaning Representations](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Unified_Visual-Semantic_Embeddings_Bridging_Vision_and_Language_With_Structured_Meaning_CVPR_2019_paper.pdf), CVPR 2019

[Multi-Task Learning of Hierarchical Vision-Language Representation](https://arxiv.org/abs/1812.00500), CVPR 2019

[Learning Factorized Multimodal Representations](https://arxiv.org/abs/1806.06176), ICLR 2019 [[code]](https://github.com/pliang279/factorized/)


[Learning Video Representations using Contrastive Bidirectional Transformer](https://arxiv.org/abs/1906.05743), arXiv 2019

[OmniNet: A Unified Architecture for Multi-modal Multi-task Learning](https://arxiv.org/abs/1907.07804), arXiv 2019 [[code]](https://github.com/subho406/OmniNet)

[Learning Representations by Maximizing Mutual Information Across Views](https://arxiv.org/abs/1906.00910), arXiv 2019 [[code]](https://github.com/Philip-Bachman/amdim-public)




[A Probabilistic Framework for Multi-view Feature Learning with Many-to-many Associations via Neural Networks](https://arxiv.org/abs/1802.04630), ICML 2018

[Do Neural Network Cross-Modal Mappings Really Bridge Modalities?](https://aclweb.org/anthology/P18-2074), ACL 2018



[Learning Robust Visual-Semantic Embeddings](https://arxiv.org/abs/1703.05908), ICCV 2017

[Deep Multimodal Representation Learning from Temporal Data](https://arxiv.org/abs/1704.03152), CVPR 2017



[Is an Image Worth More than a Thousand Words? On the Fine-Grain Semantic Differences between Visual and Linguistic Representations](https://www.aclweb.org/anthology/C16-1264), COLING 2016

[Combining Language and Vision with a Multimodal Skip-gram Model](https://www.aclweb.org/anthology/N15-1016), NAACL 2015


[Learning Grounded Meaning Representations with Autoencoders](https://www.aclweb.org/anthology/P14-1068), ACL 2014

[Deep Fragment Embeddings for Bidirectional Image Sentence Mapping](https://arxiv.org/abs/1406.5679), NIPS 2014

[Multimodal Learning with Deep Boltzmann Machines](https://dl.acm.org/citation.cfm?id=2697059), JMLR 2014

[DeViSE: A Deep Visual-Semantic Embedding Model](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model), NeurIPS 2013

[Multimodal Deep Learning](http//dl.acm.org/citation.cfm?id=3104569), ICML 2011


### Multimodal Fusion

[Understanding and Improving Encoder Layer Fusion in Sequence-to-Sequence Learning](https://openreview.net/pdf?id=n1HD8M6WGn), ICLR 2021

[Cross-Attentional Audio-Visual Fusion for Weakly-Supervised Action Localization](https://openreview.net/pdf?id=hWr3e3r-oH5), ICLR 2021



[MMFT-BERT: Multimodal Fusion Transformer with BERT Encodings for Visual Question Answering](https://arxiv.org/abs/2010.14095), EMNLP 2020

[VolTAGE: Volatility Forecasting via Text Audio Fusion with Graph Convolution Networks for Earnings Calls](https://www.aclweb.org/anthology/2020.emnlp-main.643.pdf), EMNLP 2020

[Dual Low-Rank Multimodal Fusion](https://www.aclweb.org/anthology/2020.findings-emnlp.35.pdf), EMNLP Findings 2020



[Trusted Multi-View Classification](https://openreview.net/forum?id=OOsR8BzCnl5), ICLR 2021 [[code]](https://github.com/hanmenghan/TMC)

[Deep-HOSeq: Deep Higher-Order Sequence Fusion for Multimodal Sentiment Analysis](https://arxiv.org/pdf/2010.08218.pdf), ICDM 2020 

[Removing Bias in Multi-modal Classifiers: Regularization by Maximizing Functional Entropies](https://arxiv.org/abs/2010.10802), NeurIPS 2020 [[code]](https://github.com/itaigat/removing-bias-in-multi-modal-classifiers)

[Deep Multimodal Fusion by Channel Exchanging](https://arxiv.org/abs/2011.05005?context=cs.LG), NeurIPS 2020 [[code]](https://github.com/yikaiw/CEN)

[What Makes Training Multi-Modal Classification Networks Hard?](https://arxiv.org/abs/1905.12681), CVPR 2020

[DeepCU: Integrating Both Common and Unique Latent Information for Multimodal Sentiment Analysis](https://www.ijcai.org/proceedings/2019/503), IJCAI 2019 [[code]](https://github.com/sverma88/DeepCU-IJCAI19)

[Deep Multimodal Multilinear Fusion with High-order Polynomial Pooling](https://papers.nips.cc/paper/9381-deep-multimodal-multilinear-fusion-with-high-order-polynomial-pooling), NeurIPS 2019

[XFlow: Cross-modal Deep Neural Networks for Audiovisual Classification](https://ieeexplore.ieee.org/abstract/document/8894404), IEEE TNNLS 2019 [[code]](https://github.com/catalina17/XFlow)

[MFAS: Multimodal Fusion Architecture Search](https://arxiv.org/abs/1903.06496), CVPR 2019

[The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision](https://arxiv.org/abs/1904.12584), ICLR 2019 [[code]](http://nscl.csail.mit.edu/)

[Dynamic Fusion for Multimodal Data](https://arxiv.org/abs/1911.03821), arXiv 2019




[Unifying and merging well-trained deep neural networks for inference stage](https://www.ijcai.org/Proceedings/2018/0283.pdf), IJCAI 2018 [[code]](https://github.com/ivclab/NeuralMerger)

[Efficient Low-rank Multimodal Fusion with Modality-Specific Factors](https://arxiv.org/abs/1806.00064), ACL 2018 [[code]](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)

[Memory Fusion Network for Multi-view Sequential Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17341/16122), AAAI 2018 [[code]](https://github.com/pliang279/MFN)

[Tensor Fusion Network for Multimodal Sentiment Analysis](https://arxiv.org/abs/1707.07250), EMNLP 2017 [[code]](https://github.com/A2Zadeh/TensorFusionNetwork)

[Jointly Modeling Deep Video and Compositional Text to Bridge Vision and Language in a Unified Framework](http://web.eecs.umich.edu/~jjcorso/pubs/xu_corso_AAAI2015_v2t.pdf), AAAI 2015


### Analysis of Multimodal Models

[Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers](https://arxiv.org/abs/2103.15679), ICCV 2021, [[code]](https://github.com/hila-chefer/Transformer-MM-Explainability)

[Does my multimodal model learn cross-modal interactions? It’s harder to tell than you might think!](https://arxiv.org/abs/2010.06572), EMNLP 2020

[Decoupling the Role of Data, Attention, and Losses in Multimodal Transformers](https://arxiv.org/abs/2102.00529), TACL 2021

[Blindfold Baselines for Embodied QA](https://arxiv.org/abs/1811.05013), NIPS 2018 Visually-Grounded Interaction and Language Workshop

[Analyzing the Behavior of Visual Question Answering Models](https://arxiv.org/abs/1606.07356), EMNLP 2016

### Multimodal Pretraining

[Multi-stage Pre-training over Simplified Multimodal Pre-training Models](), ACL 2021

[Integrating Multimodal Information in Large Pretrained Transformers](), ACL 2020

[Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling](https://arxiv.org/abs/2102.06183), CVPR 2021 [[code]](https://github.com/jayleicn/ClipBERT)

[Large-Scale Adversarial Training for Vision-and-Language Representation Learning](https://arxiv.org/abs/2006.06195), NeurIPS 2020 [[code]](https://github.com/zhegan27/VILLA)

[Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision](https://arxiv.org/abs/2010.06775), EMNLP 2020 [[code]](https://github.com/airsplay/vokenization)

[Integrating Multimodal Information in Large Pretrained Transformers](https://arxiv.org/abs/1908.05787), ACL 2020

[Transformer is All You Need: Multimodal Multitask Learning with a Unified Transformer](https://arxiv.org/abs/2102.10772), arXiv 2021



[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265), NeurIPS 2019 [[code]](https://github.com/jiasenlu/vilbert_beta)

[LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490), EMNLP 2019 [[code]](https://github.com/airsplay/lxmert)

[VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766), ICCV 2019

[Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training](https://arxiv.org/abs/1908.06066), arXiv 2019

[M-BERT: Injecting Multimodal Information in the BERT Structure](https://arxiv.org/abs/1908.05787), arXiv 2019

[VL-BERT: Pre-training of Generic Visual-Linguistic Representations](https://arxiv.org/abs/1908.08530), arXiv 2019 [[code]](https://github.com/jackroos/VL-BERT)

[VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557), arXiv 2019 [[code]](https://github.com/uclanlp/visualbert)



### Self-supervised Learning

[Self-supervised Representation Learning with Relative Predictive Coding](https://openreview.net/pdf?id=068E_JSq9O), ICLR 2021

[Exploring Balanced Feature Spaces for Representation Learning](https://openreview.net/pdf?id=OqtLIabPTit), ICLR 2021

[There Is More Than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking With Sound by Distilling Multimodal Knowledge](https://openaccess.thecvf.com/content/CVPR2021/html/Valverde_There_Is_More_Than_Meets_the_Eye_Self-Supervised_Multi-Object_Detection_CVPR_2021_paper.html), CVPR 2021, [[code]](http://rl.uni-freiburg.de/research/multimodal-distill), [[homepage]]( http://rl.uni-freiburg.de/research/multimodal-distill)

[Self-Supervised Learning by Cross-Modal Audio-Video Clustering](https://arxiv.org/abs/1911.12667), NeurIPS 2020 [[code]](https://github.com/HumamAlwassel/XDC)

[Self-Supervised MultiModal Versatile Networks](https://arxiv.org/abs/2006.16228), NeurIPS 2020 [[code]](https://tfhub.dev/deepmind/mmv/s3d/1)

[Labelling Unlabelled Videos from Scratch with Multi-modal Self-supervision](https://arxiv.org/abs/2006.13662), NeurIPS 2020 [[code]](https://www.robots.ox.ac.uk/~vgg/research/selavi/)

[Self-Supervised Learning from Web Data for Multimodal Retrieval](https://arxiv.org/abs/1901.02004), arXiv 2019

[Self-Supervised Learning of Visual Features through Embedding Images into Text Topic Spaces](https://ieeexplore.ieee.org/document/8099701), CVPR 2017

[Multimodal Dynamics : Self-supervised Learning in Perceptual and Motor Systems](https://dl.acm.org/citation.cfm?id=1269207), 2016


### Generative Multimodal Models

[Relating by Contrasting: A Data-efficient Framework for Multimodal Generative Models](https://openreview.net/pdf?id=vhKe9UFbrJo), ICLR 2021

[Generalized Multimodal ELBO](https://openreview.net/pdf?id=5Y21V0RDBV), ICLR 2021

[UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](https://arxiv.org/abs/2012.15409), ACL 2021

[Few-shot Video-to-Video Synthesis](https://arxiv.org/abs/1910.12713), NeurIPS 2019 [[code]](https://nvlabs.github.io/few-shot-vid2vid/)

[Multimodal Generative Models for Scalable Weakly-Supervised Learning](https://arxiv.org/abs/1802.05335), NeurIPS 2018 [[code1]](https://github.com/mhw32/multimodal-vae-public) [[code2]](https://github.com/panpan2/Multimodal-Variational-Autoencoder)

[Look, Imagine and Match: Improving Textual-Visual Cross-Modal Retrieval with Generative Models](https://arxiv.org/abs/1711.06420), CVPR 2018

[The Multi-Entity Variational Autoencoder](http://charlienash.github.io/assets/docs/mevae2017.pdf), NeurIPS 2017


### Multimodal Adversarial Attacks


[Attend and Attack: Attention Guided Adversarial Attacks on Visual Question Answering Models](https://nips2018vigil.github.io/static/papers/accepted/33.pdf), NeurIPS Workshop on Visually Grounded Interaction and Language 2018

[Attacking Visual Language Grounding with Adversarial Examples: A Case Study on Neural Image Captioning](https://arxiv.org/abs/1712.02051), ACL 2018 [[code]](https://github.com/huanzhang12/ImageCaptioningAttack)

[Fooling Vision and Language Models Despite Localization and Attention Mechanism](https://arxiv.org/abs/1709.08693), CVPR 2018


## Research Tasks

### Sentiment and Emotion Analysis

[MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation](https://arxiv.org/abs/2107.06779), ACL 2021

[Multimodal Sentiment Detection Based on Multi-channel Graph Neural Networks](), ACL 2021

[Dual Graph Convolutional Networks for Aspect-based Sentiment Analysis](), ACL 2021

[Multi-Label Few-Shot Learning for Aspect Category Detection](https://arxiv.org/abs/2105.14174), ACL 2021

[Directed Acyclic Graph Network for Conversational Emotion Recognition](https://arxiv.org/abs/2105.12907), ACL 2021

[CTFN: Hierarchical Learning for Multimodal Sentiment Analysis Using Coupled-Translation Fusion Network](), ACL 2021

[Learning Language and Multimodal Privacy-Preserving Markers of Mood from Mobile Data](https://arxiv.org/abs/2106.13213), ACL 2021

[A Text-Centered Shared-Private Framework via Cross-Modal Prediction for Multimodal Sentiment Analysis](), ACL Findings 2021



### Trajectory and Motion Forecasting
[HalentNet: Multimodal Trajectory Forecasting with Hallucinative Intents](https://openreview.net/pdf?id=9GBZBPn0Jx), ICLR 2021

[Multimodal Motion Prediction with Stacked Transformers](https://arxiv.org/abs/2103.11624), CVPR 2021 [[code]](https://decisionforce.github.io/mmTransformer)

[Social NCE: Contrastive Learning of Socially-aware Motion Representations](https://arxiv.org/abs/2012.11717), ICCV 2021, [[code]](https://github.com/vita-epfl/social-nce-crowdnav)

[The Garden of Forking Paths: Towards Multi-Future Trajectory Prediction](https://arxiv.org/abs/1912.06445), ECCV 2020, [[code]](https://github.com/JunweiLiang/Multiverse)

## Datasets

[MultiMET: A Multimodal Dataset for Metaphor Understanding](), ACL 2021

[A Large-Scale Chinese Multimodal NER Dataset with Speech Clues](), ACL 2021

[A Recipe for Creating Multimodal Aligned Datasets for Sequential Tasks](https://arxiv.org/abs/2005.09606), ACL 2020

[CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality](https://www.aclweb.org/anthology/2020.acl-main.343.pdf), ACl 2020, [[code]](https://github.com/thuiar/MMSA)

[CMU-MOSEAS: A Multimodal Language Dataset for Spanish, Portuguese, German and French](https://www.aclweb.org/anthology/2020.emnlp-main.141.pdf), EMNLP 2020 


## Tutorials
