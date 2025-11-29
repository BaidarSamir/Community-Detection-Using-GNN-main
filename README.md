# Community Detection Using Graph Neural Networks

## Abstract
We study community detection within the GitHub developer ecosystem as a binary node-classification problem. A Graph Neural Network (GNN) built with Graph Convolutional Networks (GCNs) is trained to differentiate machine learning (ML) developers from web developers using metadata scraped from GitHub’s public API. The method is benchmarked against Gaussian Naive Bayes and Logistic Regression baselines. Our best GNN reaches 87.0% validation accuracy and 86.4% test accuracy, outperforming feature-only models and highlighting the value of graph-aware learning for social-network analysis.

## 1. Introduction
Understanding developer communities helps quantify technology diffusion, collaboration patterns, and workforce needs. Public GitHub metadata encodes these relationships through profile attributes and social edges. Community labels are inferred from repository stars, job titles, and other profile signals, enabling supervised classification. We compare GNNs with conventional classifiers to answer two questions:

1. Does message passing on the GitHub interaction graph improve community prediction?
2. How do classical baselines fare when provided with the same one-hot encoded metadata?

## 2. Data Description
### 2.1 Sources
- **musae_git_features.json**: sparse feature dictionaries per node.
- **musae_git_target.csv**: binary `ml_target` label (1 = ML developer, 0 = web developer).
- Metadata was collected via GitHub’s public API in June 2019 covering 37,700 developers.

### 2.2 Graph Statistics
```
Data(x=[37,700, 4,005], edge_index=[2, 578,006], y=[37,700],
     train_mask=[37,700], val_mask=[37,700], test_mask=[37,700])
```
- Training / validation / test splits: 22,620 / 11,310 / 3,770 nodes (60/30/10).
- Graph density remains low; most nodes have fewer than 20 connections.

### 2.3 Feature Landscape
One-hot encoding yields 4,005 binary indicators spanning locations, employers, starred repositories, and email domains. Feature sparsity and frequency are illustrated in Figure 1.

![Top features heatmap](visuals/top_features_heatmap.png)
*Figure 1. Most frequent metadata attributes across 37,700 developers.*

![Sparse feature matrix](visuals/sparse_features.png)
*Figure 2. Sparse structure of the encoded feature matrix.*

### 2.4 Graph Visualization
We visualize a light subset of the constructed PyTorch Geometric data object in Figure 3.

![Graph sample](visuals/graph_sample.png)
*Figure 3. Sample of the developer network (red = web, gray = ML).*

## 3. Methodology
### 3.1 Feature Engineering
1. Parse the JSON feature list per node.
2. Aggregate counts to identify informative attributes.
3. Build one-hot encodings for all nodes, supporting a “light” subset for visualization.

### 3.2 Graph Construction
The adjacency information is constructed from the musae edge list. Encoded features are attached to each node, and labels from `ml_target` are stored in PyG tensors along with boolean masks for the three splits.

### 3.3 Models
#### Graph Neural Network (SocialGNN)
- **Architecture**: Two-layer GCN with ReLU in the hidden layer and linear output projecting to two logits.
- **Hidden size**: 16.
- **Loss**: Cross-entropy with masking to restrict computation to labeled nodes.
- **Optimization**: Adam with StepLR decay (step size 10, gamma 0.9).
- **Regularization**: Dropout applied between GCN layers (p = 0.5 in notebook).

#### Naive Bayes
- Gaussian Naive Bayes on the transposed encoded feature matrix.
- Evaluated with 4-fold cross-validation.

#### Logistic Regression
- L2-regularized logistic regression trained with LBFGS.
- 4-fold cross-validation with shuffled splits.

## 4. Experimental Setup
### 4.1 Training Protocol
Two GNN runs are reported:

| Epoch Budget | Initial LR | Best Epoch (Val) | Train Acc | Val Acc | Test Acc |
|--------------|------------|------------------|-----------|---------|----------|
| 50           | 0.10       | 45–47            | 0.8789    | 0.8701  | 0.8610   |
| 100          | 0.01       | 46               | 0.8744    | 0.8691  | 0.8621   |

Key milestones for the 50-epoch schedule include early improvements from 74% to 85% accuracy by epoch 6 and stabilization near 87% by epoch 27 onwards. The 100-epoch schedule follows a smoother trajectory, transitioning from 73% to 86% within the first 30 epochs before slowly converging (see Figures 4 and 5).

### 4.2 Software Stack
- Python 3.12
- PyTorch 2.x and PyTorch Geometric
- scikit-learn for baselines and evaluation
- NetworkX and Matplotlib for visualization
- Jupyter notebook: `GNN_community_detection.ipynb`

## 5. Results
### 5.1 GNN Learning Dynamics

![GNN losses](visuals/gnn_losses.png)
*Figure 4. Training vs. validation loss for the GNN (50-epoch schedule).*

![GNN accuracies](visuals/gnn_accuracies.png)
*Figure 5. Training, validation, and test accuracies over epochs.*

The GNN converges reliably, with validation accuracy plateauing around 86.8–87.0%. The test curve mirrors validation performance, indicating minimal overfitting despite the high-dimensional feature space.

### 5.2 Baseline Classifiers

| Model             | Validation Strategy | Mean Accuracy | Notes |
|-------------------|---------------------|---------------|-------|
| Gaussian Naive Bayes | 4-fold CV          | 44.45%        | Sensitive to sparse, high-dimensional features; confusion matrix shows class imbalance challenges. |
| Logistic Regression | 4-fold CV          | 83.42%        | Robust to sparsity; predictions align closely with labels (Figure 6). |

![Naive Bayes confusion](visuals/confusion_naivebayes.png)
*Figure 6. Confusion matrix for a representative Naive Bayes fold.*

![Logistic Regression confusion](visuals/confusion_logistic.png)
*Figure 7. Confusion matrix for Logistic Regression.*

![Actual vs predicted LR](visuals/actual_vs_predicted_lr.png)
*Figure 8. Logistic Regression label distribution compared with ground truth.*

### 5.3 Comparative Analysis

![Model comparison](visuals/model_comparison.png)
*Figure 9. Test accuracies across models.*

The GNN outperforms Logistic Regression by ~3 percentage points and Naive Bayes by over 40 points, demonstrating the benefit of incorporating relational context. Logistic Regression remains a strong baseline thanks to the informative metadata, while Naive Bayes suffers from correlated features and skewed distributions.

## 6. Discussion
- **Graph signal strength**: The improvement over Logistic Regression indicates that peer labels in the GitHub network are predictive and complementary to individual profile metadata.
- **Optimization stability**: Both learning-rate schedules converge to similar plateaus, suggesting the architecture is capacity-limited rather than undertrained. Deeper GNNs or attention layers could unlock further gains.
- **Baseline behavior**: Naive Bayes’ low recall for ML developers (Figure 6) highlights the necessity of modeling dependencies between features such as employer keywords and starred repositories.

## 7. Conclusion and Future Work
Graph-based learning provides meaningful lifts for community detection on GitHub. Future directions include:
1. Testing higher-capacity architectures (GraphSAGE, GAT, or MixHop) that can capture heterophily.
2. Incorporating node embeddings from Node2Vec or contrastive pretraining before supervised fine-tuning.
3. Expanding labels beyond a binary taxonomy to represent multiple community memberships.
4. Adding fairness and temporal analyses to understand how community membership evolves.

## 8. Reproduction Guide
1. Clone this repository.
2. Open `GNN_community_detection.ipynb` in Jupyter or Google Colab.
3. Install dependencies inside the notebook (`pip install numpy pandas matplotlib networkx torch torch_geometric scikit-learn`).
4. Execute cells sequentially to download raw data, build encodings, construct the graph, and train all models.
5. Regenerate figures with the plotting cells—outputs are saved in the `visuals/` directory.

## References
1. SNAP GitHub Social Network Dataset: https://snap.stanford.edu/data/github-social.html  
2. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io  
3. Awadelrahman, “Tutorial: Graph Neural Networks on Social Networks,” Kaggle, 2021.  
4. TensorFlow, “Intro to Graph Neural Networks (ML Tech Talks),” 2021.  
5. Khare, P., “Unravelling Node2Vec,” Medium, 2023.  
6. Awan, A. A., “A Comprehensive Introduction to Graph Neural Networks,” 2022.  
