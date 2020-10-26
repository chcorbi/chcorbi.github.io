---
title: "Decomposition of $\\textrm{KL}_{\\textrm{Pred}}$"
date: 2020-10-26
permalink: /posts/2020/10/klpred-decompo/
tags:
  - Dirichlet Prior Networks
---

Notations
------

Let us consider a training dataset $\mathcal{D}$ consisting of $N$ *i.i.d.* samples, 

$$\begin{equation*} \mathcal{D}= \{ (\boldsymbol{x}_i, y^*_i) \}_{i=1}^N \in (\mathcal{X} \times \mathcal{Y})^N \end{equation*}$$

where $\mathcal{X}$ represents the input space and $\mathcal{Y}=\{1,\ldots,K\}$ is a set of labels. \\
Samples drawn from $\mathcal{D}$ follow an unknown conditional probability distribution $p(\mathbf{y} \vert \mathbf{x})$ where $\mathbf{x}$ and $\mathbf{y}$ are random variables over input space and label space respectively. 

Let $f^{\boldsymbol{\theta}}: \mathcal{X} \rightarrow \mathcal{X'}$ be a neural network (NN) parametrized by $\boldsymbol{\theta}$ where $\mathcal{X'} = \mathbb{R}^K$ is the logit space. We consider categorical probabilities over labels as random variable $\mathbf{z}$. Following {% cite malinin2018 %}, a NN explicitly parametrizes a distribution $p\_{\theta}(\mathbf{z} \vert \mathbf{x})$ over categorical probabilities on a simplex. For its conjugate properties with categorical distributions, we chose to model $p\_{\theta}(\mathbf{z} \vert \mathbf{x})$ as a Dirichlet distribution whose concentration parameters $\boldsymbol{\alpha}(\boldsymbol{x}, \boldsymbol{\theta}) = \exp (f^{\boldsymbol{\theta}}(\boldsymbol{x}))$ are given by the output of the NN:
\begin{equation}
    p_{\theta}(\mathbf{z} \vert \mathbf{x} = \boldsymbol{x}) = \mathrm{Dir} \big ( \mathbf{z} \vert \boldsymbol{\alpha}(\boldsymbol{x}, \boldsymbol{\theta}) \big )
    = \frac{\Gamma(\alpha_0 (\boldsymbol{x}, \boldsymbol{\theta}))}{\prod_c \Gamma(\alpha_c(\boldsymbol{x}, \boldsymbol{\theta}))} \prod_{j=1}^K z_j^{\alpha_c(\boldsymbol{x}, \boldsymbol{\theta}) - 1}
\end{equation}
where $\Gamma$ is the Gamma function,  $\forall c \in \mathcal{Y}, \alpha_j > 0$ , $\alpha_0 = \sum_c \alpha_c$ and $\sum_c z_c$ = 1 such that $\mathbf{z}$ lives in 
the $(K-1)$-dimensional unit simplex $\triangle^{K-1}$.

TTest

References
----------

{% bibliography --cited %}