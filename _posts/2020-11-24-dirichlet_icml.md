---
title: "Preparation on Dirichlet - ICML 2021"
date: 2020-11-26
permalink: /posts/2020/11/dirichlet-icml/
tags:
  - Dirichlet Networks
---

## Table of contents
1. [Context](#context)
2. [Contributions](#contributions)
3. [Thought experiment with Gaussian distributions](#gaussian)
4. [Experiments](#experiments)


Context <a name="context"></a>
------

Current metrics to measure in-domain and out-domain uncertainty include 
* maximum class probability $$MCP = \max_c p(y=c \vert \boldsymbol{x^*}; \hat{\boldsymbol{\theta}})$$
* predictive entropy $$\mathcal{H} \big [ p( y \vert \boldsymbol{x}, \hat{\boldsymbol{\theta}}) \big ] $$
* mutual information $$ \mathcal{MI} \Big [y, \boldsymbol{\pi} \vert \boldsymbol{x^*}; \hat{\boldsymbol{\theta}} \Big ] $$ 
* differential entropy $$\mathcal{H} \big [ p( \boldsymbol{\pi} \vert \boldsymbol{x}, \hat{\boldsymbol{\theta}}) \big ] $$
* Dirichlet precision $$\alpha_0 = \sum_c \alpha_c$$

In many papers {% cite malinin2018 %}, they consider measures from the expected predictive categorical distribution $$p(y \vert \boldsymbol{x^*}, \mathcal{D})$$ as a mesure of **total uncertainty**. This includes MCP and predictive entropy. 

Based on {% cite depeweg2018decomposition %}, they decompose the predictive entropy into two terms :

$$
\begin{equation}
\underbrace{\mathcal{H} \Big [ \mathbb{E}_{p(\boldsymbol{\pi} \vert \boldsymbol{x^*}; \hat{\boldsymbol{\theta}})} \big [ p(y \vert \boldsymbol{\pi}) \big] \Big ]}_{\text{Total Uncertainty}} = \underbrace{\mathbb{E}_{p(\boldsymbol{\pi} \vert \boldsymbol{x^*}; \hat{\boldsymbol{\theta}})} \Big [ \mathcal{H} \big [ p(y \vert \boldsymbol{\pi}) \big] \Big]}_{\text{Expected Aleatoric Uncertainty}} + \underbrace{\mathcal{MI} \Big [y, \boldsymbol{\pi} \vert \boldsymbol{x^*}; \hat{\boldsymbol{\theta}} \Big ]}_{\text{Epistemic Uncertainty}}
\end{equation}
$$ 

Consequently, mutual information is an epistemic uncertainty measure.

{:refdef: style="text-align: center;"}
![desired_behavior](/images/desired_behavior.png)
{: refdef}

Finally, Dirichlet precision is also expected to be an epistemic uncertainty measure as it measures the dispersion of the Dirichlet distribution on the simplex. The higher the epistemic uncertainty the higher we desire the dispersion to be.


Contributions <a name="contributions"></a>
------

### Summary
* A new total uncertainty measure for Dirichlet Networks, $$\textrm{KL}_{\textrm{Pred}}$$, with properties similar to MCP
* $$\textrm{KL}_{\textrm{Pred}}$$ can be decomposed into two terms which allow to distinguish aleatoric and epistemic uncertianty
* This new criterion enable to consider an improved version based on the *ground-truth* class and which improves misclassification detection

### A new total uncertainty measure 

In neural network trained to minimize cross-entropy, which is also equivalent to the KL-divergence between model distribution and empirical true distribution:

$$
\begin{equation}
 \mathcal{L}_{\textrm{XE}}(\boldsymbol{x},y^*, \boldsymbol{\theta}) = \textrm{KL} \Big ( \hat{p}(y \vert \boldsymbol{x}) \vert \vert \textrm{Cat}(y \vert \boldsymbol{x}, \boldsymbol{\theta}) \Big ) = - \log p(y=y^* \vert \boldsymbol{x}, \boldsymbol{\theta})
\end{equation}
$$

The standard uncertainty measure is MCP, which is the estimated likelihood $$p(y=\hat{y} \vert \boldsymbol{x^*}, \hat{\boldsymbol{\theta}})$$ at sample $\boldsymbol{x^*}$ by the model.

With Dirichlet Networks, training is achieved by minimizing the reverse KL-divergence with a sharp target Dirichlet distribution

$$
\begin{equation}
 \mathcal{L}_{\textrm{RKL}}(\boldsymbol{x},y^*, \boldsymbol{\theta}) = \textrm{KL} \Big ( \textrm{Dir} ( \boldsymbol{\pi} \vert \boldsymbol{x}, \boldsymbol{\theta} ) ~\vert \vert~  \textrm{Dir} ( \boldsymbol{\pi} \vert \boldsymbol{\beta}^{(y^*)}) \Big ) 
\end{equation}
$$

In a similar way than for cross-entropy, we propose an uncertainty criterion, denoted $\\textrm{KL}\_{\\textrm{Pred}}$, which measures the KL-divergence between NN's output and a sharp Dirichlet distribution with concentration parameters $\boldsymbol{\gamma}\_{\hat{y}}$ focused on the *predictive* class $\hat{y}$:

$$
\begin{equation}
    \textrm{KL}_{\textrm{Pred}}(\boldsymbol{x}) = \textrm{KL} \Big ( \textrm{Dir} (\boldsymbol{\pi} \vert \boldsymbol{x}, \boldsymbol{\hat{\theta}} ) ~\vert \vert~ \textrm{Dir} ( \boldsymbol{\pi} \vert \boldsymbol{\gamma}^{(\hat{y})} \big ) \Big )
\end{equation} 
$$

To ensure an accurate estimation of concentration parameters $\boldsymbol{\gamma}^{(\hat{y})}$, we compute the empirical exponential logits mean of the predicted class $\hat{y}$ on training set $\mathcal{D}$:

$$
\begin{equation*}
    \boldsymbol{\gamma}^{(\hat{y})} = \frac{1}{N^{(\hat{y})}} \sum_{i: y_i=\hat{y}}^N \boldsymbol{\alpha}(\boldsymbol{x_i}, \boldsymbol{\hat{\theta}}), \quad \quad  \textrm{with}~~ \boldsymbol{\alpha}(\boldsymbol{x_i}, \boldsymbol{\hat{\theta}}) = \exp (f(\boldsymbol{x}_i,\boldsymbol{\hat{\theta}}))
\end{equation*}
$$

where $N^{(\hat{y})}$ is the number of training samples with label $\hat{y}$.

![simplex_behavior](/images/klpred_behavior.png){:height="700px" width="700px"}

The lower $\textrm{KL}\_{\textrm{Pred}}$ is, the more certain we are in the prediction. Previous figure illustrates the fact that correct predictions will have Dirichlet distributions similar to the computed mean distribution for the predicted class, and thus associated with a low uncertainty measure. Misclassified predictions are expected to present different concentration parameters than the average computed on training set resulting in a higher $\textrm{KL}\_{\textrm{Pred}}$ measure.

**For now, this is just another measure, we don't have any theoretical explanation whether it is a better to evaluate total uncertainty. Without further justification, it might be as good as MCP or predictive entropy.**


### Decomposition of $$\textrm{KL}_{\textrm{Pred}}$$


Such as done for the reverse KL-divergence loss in {% cite malinin2019 %}, we decompose $$\textrm{KL}_{\textrm{Pred}}$$ into the reverse cross-entropy and the negative differential entropy:

$$
\begin{equation}
\textrm{KL}_{\textrm{Pred}}(\boldsymbol{x}) = \underbrace{\mathbb{E}_{p \big ( \boldsymbol{\pi} \vert \boldsymbol{x}, \boldsymbol{\hat{\theta}} ) \big )} \Big [- \log \textrm{Dir} \big ( \boldsymbol{\pi}  \vert \boldsymbol{\gamma}_{(\hat{y})} \big ) \Big ]}_\text{Reverse Cross-Entropy} - \underbrace{\mathcal{H} \Big [ \textrm{Dir} \big (\boldsymbol{\pi} \vert \boldsymbol{x}, \boldsymbol{\hat{\theta}}) \big )  \Big ]}_\text{Differential Entropy}
\end{equation}
$$

where we can show that the Reverse Cross-Entropy (RCE) correspond to measuring aleatoric uncertainty and the Differential Entropy measures the dispersion of the Dirichlet distribution, hence epistemic uncertainty.

**Still, no justification if it is better than the previous proposed decomposition of predictive entropy in {% cite malinin2018 %}.**


### Improving misclassification detection for Dirichlet Networks

Similarly to ConfidNet, we can improve the evaluation of misclassification detection by considering measure in respect to the *ground-truth* class.

$$
\begin{equation}
    \textrm{KL}_{\textrm{True}}(\boldsymbol{x}) = \textrm{KL} \Big ( \textrm{Dir} \big (\mathbf{z} \vert \boldsymbol{\alpha}(\boldsymbol{x}, \boldsymbol{\hat{\theta}}) \big ) ~\vert \vert~ \textrm{Dir} \big ( \mathbf{z} \vert \boldsymbol{\gamma}_{y^*} \big ) \Big )
\end{equation} 
$$

where $$\boldsymbol{\gamma}_{y^*}$$ corresponds to a Dirichlet distribution whose concentration parameters are the empirical mean of the *true* class $$y^*$$  of sample $$\boldsymbol{x}$$. When evaluated in experiments, $$\textrm{KL}_{\textrm{True}}$$ criterion leads to a near-perfect separation of correct and erroneous predictions.

However, the true class of an output is obviously not available when estimating confidence on test samples. We propose to learn $$\textrm{KL}_{\textrm{True}}$$ by introducing an auxiliary confidence neural network, termed *KLNet*, with parameters $$\boldsymbol{\omega}$$, which outputs a confidence prediction $$\hat{c}(\boldsymbol{x}, \boldsymbol{\omega})$$.


**Now we can justify than *KLNet* improves total uncertainty by better evaluating aleatoric uncertainty. Nevertheless, we have no guarantees about epistemic uncertainty. Plus, current KLNet training doesn't take into account OOD samples.**


Thought experiment with Gaussian distributions <a name="gaussian"></a>
------

Let's suppose the random variable over categorical probabilities $\boldsymbol{\pi} =[\pi_1,...,\pi_C]$ is now parametrize as a $K$-multivariate Gaussian distribution:

$$
\begin{equation}
	p(\boldsymbol{\pi} \vert \boldsymbol{x^*}, \mathcal{D}) = \mathcal{N}(\boldsymbol{\pi} \vert \boldsymbol{\mu_1}, \boldsymbol{\Sigma_1})
\end{equation}
$$

For instance, an output of a neural network could be $\boldsymbol{\mu_1} = [0.98, 0.01, .01]$ and $\boldsymbol{\Sigma_1}$ = [0.05, 0.03, 0.02].

> On a related matter, {% cite kendall2017 %} propose in classification to consider a Gaussian over $\boldsymbol{\pi} \vert \boldsymbol{w}$ and parametrized by a NN outputing logits $f(\boldsymbol{x}, \boldsymbol{w})$ (e.g  $[98.1, 1.2, 1.3]$) and a scalar $\boldsymbol{\sigma}$ (e.g $[2.1,0.6,3.3]$). Mean probabilities corresponds to the softmax of the mean corrupted by Gaussian noise:
>
>$$
>\begin{equation}
>	\boldsymbol{\hat{\pi}}_t = \mathrm{Softmax}(\boldsymbol{\pi}_t)~~~~\mathrm{with }~ \boldsymbol{\pi}_t = f(\boldsymbol{x}, \boldsymbol{w}) +\boldsymbol{\sigma}\epsilon_t,~~~\epsilon_t \sim \mathcal{N}(0,I)
>\end{equation}
>$$


On the simplex, the distribution can be represented with $\boldsymbol{\mu_1}$ as its center position and dispersion corresponds to $\boldsymbol{\Sigma_1}$.
Predictions are based on the argmax of the mean parameter, which is the first moment of the distribution:

$$
\begin{equation}
\hat{y} = arg\,max_{c}~ \mathbb{E}_{p(\boldsymbol{\pi} \vert \boldsymbol{x^*}, \mathcal{D})}[y] = arg\,max_{c}~ \boldsymbol{\mu_1}
\end{equation}
$$

In this case, **the entropy corresponds to computing the entropy on the mean of the distribution**: $\mathcal{H} \big [ \mathbb{E}_{p(\boldsymbol{\pi} \vert \boldsymbol{x^*}, \mathcal{D})}[y] \big ]$.

To reflect epistemic uncertainty, we should also consider the second moment of the distribution $\boldsymbol{\Sigma_1}$. For instance, the higher $\boldsymbol{\Sigma_1}[\hat{y}]$ is, the higher would be the epistemic uncertainty.

In order to consider both aleatoric and epistemic uncertainty, we could try to derive some statistics on $p(\boldsymbol{\pi} \vert \boldsymbol{x^*}, \mathcal{D})$. Given a target distribution $p(\boldsymbol{\pi} \vert \boldsymbol{\mu_2}, \boldsymbol{\Sigma_2}) = \mathcal{N}(\boldsymbol{\pi} \vert \boldsymbol{\mu_2}, \boldsymbol{\Sigma_2})$ also Gaussian, the **KL-divergence** between the two distributions is:

$$
\begin{align}
\mathrm{KL} \big [ p(\boldsymbol{\pi} \vert \boldsymbol{x^*}, \mathcal{D}) \vert \vert p(\boldsymbol{\pi} \vert \boldsymbol{\mu_2}, \boldsymbol{\Sigma_2})] = \frac{1}{2} \Big (\log \vert\boldsymbol{\Sigma_2}\vert - &\log \vert\boldsymbol{\Sigma_1}\vert - K + tr( \boldsymbol{\Sigma_2}^{-1} \boldsymbol{\Sigma_1}) \\ \nonumber
 &+ (\boldsymbol{\mu_2} - \boldsymbol{\mu_1})\boldsymbol{\Sigma_2}^{-1}(\boldsymbol{\mu_2}-\boldsymbol{\mu_1}) \Big ) 
\end{align}
$$

Setting aside the terms which do not depend on the input, we can identity two term. The first one relates to the first moment of the distribution, $(\boldsymbol{\mu_2}-\boldsymbol{\mu_1})\boldsymbol{\Sigma_2}^{-1}(\boldsymbol{\mu_2}-\boldsymbol{\mu_1})$ and the second one involves only the variance $- \log \vert\boldsymbol{\Sigma_1}\vert + tr( \boldsymbol{\Sigma_2}^{-1} \boldsymbol{\Sigma_1})$.

Empirical Experiments <a name="experiments"></a>
------

Two models were trained:
* with standard cross-entropy (**XE**)
* with contrastive reverse KL-divergence (**Dirichlet**)


### CIFAR-10 

* In-distribution dataset: CIFAR-10
* Out-distribution training dataset: CIFAR-100
* Network Architecture: VGG-16
* Concentrations parameters $\alpha= \exp{f(x, \theta)}$
* Target concentrations for in-domain: $$\beta_{\text{in}}$$ = 10
* Training details: Adam, LR 5e-5, 1-cyclic scheduler for 45 epochs

Accuracy is 93.5% for XE model and 92.9% for Dirichlet model.

Presented results are for **TinyImageNet** as OOD dataset (% AUC)

| Training | Method | Mis. Detection | OOD Detection | Mis.+OOD Detection |
| :------: |:-------| --------------:| -------------:| ------------------:|
| **XE**   | MCP<br>ODIN<br>Mahalanobis<br>ConfidNet | $92.6\%$<br>$91.4\%$<br>$90.2\%$<br>$92.6\%$ | $86.8\%$<br>$88.2\%$<br>$83.0\%$<br>$87.8\%$ | $91.2\%$<br>$91.1\%$<br>$87.8\%$<br>$91.6\%$ |
| **Dirichlet** | MCP<br>ODIN<br>Mutual Information<br>Mahalanobis<br>ConfidNet<br>**KLNet (Ours)** | $91.6\%$<br>$91.6\%$<br>$91.2\%$<br>$92.1\%$<br>$93.5\%$<br>$\boldsymbol{93.7\%}$  | $92.8\%$<br>$92.9\%$<br>$92.9\%$<br>$86.6\%$<br>$93.1\%$<br>$\boldsymbol{93.3\%}$<br> | $93.1\%$<br>$93.1\%$<br>$92.9\%$<br>$91.1\%$<br>$94.2\%$<br>$\boldsymbol{94.5\%}$ | 


##### Effect of the computed mean $$\gamma^{(\hat{y})}$$

We have three cases about how to compute the target distribution in the criterion  $\\textrm{KL}\_{\\textrm{Pred}}$

* **KL\_Original**: Take exactly the target distribution use in training:

$$
\begin{equation*}
 \gamma^{(0, \hat{y})} = \big [ 1,..., \beta_{\text{in}},...,1 \big ]
\end{equation*}
$$

* **KL\_Full**: Compute the empirical exponential logits mean of the predicted class $\hat{y}$ on training set $\mathcal{D}$ and use **all values**:

$$
\begin{equation*}
    \boldsymbol{\gamma}^{(1, \hat{y})} = \frac{1}{N^{(\hat{y})}} \sum_{i: y_i=\hat{y}}^N \boldsymbol{\alpha}(\boldsymbol{x_i}, \boldsymbol{\hat{\theta}}), \quad \quad  \textrm{with}~~ \boldsymbol{\alpha}(\boldsymbol{x_i}, \boldsymbol{\hat{\theta}}) = \exp (f(\boldsymbol{x}_i,\boldsymbol{\hat{\theta}}))
\end{equation*}
$$

where $N^{(\hat{y})}$ is the number of training samples with label $\hat{y}$.

* **KL\_Pred**: Compute the empirical exponential logits mean of the predicted class $\hat{y}$ on training set $\mathcal{D}$ and use **only the $\hat{y}$-value**:

$$
\begin{equation*}
    \boldsymbol{\gamma}^{(2, \hat{y})} = \big [ 1,..., \boldsymbol{\gamma}^{(1, \hat{y})}[\hat{y}],...,1 \big ]
\end{equation*}
$$

Results using the Dirichlet model trained with $$\beta_{\text{in}}=10$$ are available in the table below:

| Method         |  Mis. Detection | OOD Detection | Mis.+OOD Detection |
| :------------- | --------------: | ------------: | -----------------: |
| KL\_Original   | $92.6\%$        | $93.0\%$      | $93.7\%$           |
| KL\_Pred       | $92.5\%$        | $93.2\%$      | $92.8\%$           |
| KL\_Full       | $92.2\%$        | $93.2\%$      | $93.6\%$           |

Actually, I observe that using more complicated form of target distribution do not impact performance.


##### Effect of $$\beta_{\text{in}}$$

We vary the chosen value for in-domain target concentrations $$\beta_{\text{in}}$$ from 10 to 10,000. Below are the results for the criterion KL\_Original:

| $$\beta_{\text{in}}$$ | Accuracy  | Mis. Detection | OOD Detection | Mis.+OOD Detection |
| :-------------------: | -------: | --------------:| -------------:| ------------------:|
| 10                    | $92.9\%$  | $92.6\%$       | $93.0\%$      | $93.7\%$           |
| 100                   | $93.0\%$  | $91.7\%$       | $92.2\%$      | $92.8\%$           |
| 1000                  | $93.5\%$  | $90.2\%$       | $90.3\%$      | $91.1\%$           |
| 10000                 | $92.6\%$  | $88.5\%$       | $88.1\%$      | $89.3\%$           |

We also note that the lower $$\beta_{\text{in}}$$ is, the less confident in-domain softmax probabilities will be. For instance, in the case of $$\beta_{\text{in}}=10$$, they range from $0.1$ to $0.57$.

> When $$\beta_{\text{in}} \geq 100$$, KL\_Original becomes worse compared to KL\_Pred or KL\_Full. This may be due to the fact that logits variation are important, making them deviate from the original target distribution.


##### Empirical evaluation of the decomposed measures

The criterion $\\textrm{KL}\_{\\textrm{Pred}}$ actually correspond to the negative reverse KL-divergence used in training. Hence, it can be decomposed into the reverse cross-entropy and the differential entropy:

$$
\begin{equation}
\textrm{KL}_{\textrm{Pred}}(\boldsymbol{x}) = - \underbrace{\mathbb{E}_{p \big ( \mathbf{z} \vert \boldsymbol{\alpha}(\boldsymbol{x}, \boldsymbol{\hat{\theta}}) \big )} \Big [- \log \textrm{Dir} \big ( \mathbf{z} \vert \boldsymbol{\gamma}_{\hat{y}} \big ) \Big ]}_\text{Reverse Cross-Entropy} + \underbrace{\mathcal{H} \Big [ \textrm{Dir} \big (\mathbf{z} \vert \boldsymbol{\alpha}) \big )  \Big ]}_\text{Differential Entropy}
\end{equation}
$$

In the synthetic experiment, we observe that the differential entropy correlates with the **epistemic uncertainty** and the reverse cross-entropy (RCE) correlates with the **aleatoric uncertainty**.

We use those decomposed metric here to evaluate their effectiveness on misclassification detection and OOD detection in the following table.
(Experiment done using the Dirichlet model trained with $$\beta_{\text{in}}=10$$)

| Method         |  Mis. Detection | OOD Detection | 
| :------------- | --------------: | ------------: |
| RCE            | $91.3\%$        | $92.8\%$      | 
| Diff. Ent.     | $91.2\%$        | $92.8\%$      | 
| KL\_Original   | $92.6\%$        | $93.0\%$      |

It doesn't seem RCE or differential entropy are more inclined to measure one type of uncertainty. Only thing we can conlude is that combine them into KL\_Original improves performances.


##### Ablation study

As observed in our Neurips submission, KLNet mainly improves misclassification detection, with a slight but uncontrolled benefit on OOD detection:

| Method         |  Mis. Detection | OOD Detection | Mis.+OOD Detection |
| :------------- | --------------: | ------------: | -----------------: |
| KL\_Original   | $92.6\%$        | $93.0\%$      | $93.7\%$           |
| KLNet_Classic  | $93.0\%$        | $93.3\%$      | $94.0\%$           |
| KLNet_Cloning  | $93.7\%$        | $93.3\%$      | $94.5\%$           |


### CIFAR-100

* In-distribution dataset: CIFAR-100
* Out-distribution training dataset: CIFAR-10
* Network Architecture: VGG-16
* Concentrations parameters $\alpha= \exp{f(x, \theta)}$
* Target concentrations for in-domain: $$\beta_{\text{in}}$$ = 10
* Training details: Adam, LR 5e-5, 1-cyclic scheduler for 45 epochs

Accuracy is 73.2% for XE model and 71.5% for Dirichlet model.

Presented results are for **TinyImageNet** as OOD dataset (% AUC)

| Training | Method | Mis. Detection | OOD Detection | Mis.+OOD Detection |
| :------: |:-------| --------------:| -------------:| ------------------:|
| **XE**   | MCP<br>ODIN<br>Mahalanobis<br> | $86.6\%$<br>$84.9\%$<br>$80.9\%$ | $75.9\%$<br>$77.4\%$<br>$73.3\%$ | $86.0\%$<br>$85.4\%$<br>$81.3\%$ |
| **Dirichlet** | MCP<br>ODIN<br>Mutual Information<br>Mahalanobis<br>**KL\_Original**<br>**KLNet**<br>**KLNet_Cloning** | $83.7\%$<br>$83.7\%$<br>$82.8\%$<br>$83.4\%$<br>$87.3\%$<br>$86.8\%$<br>$\boldsymbol{87.6\%}$ | $76.0\%$<br>$76.0\%$<br>$76.0\%$<br>$71.9\%$<br>$77.0\%$<br>$76.9\%$<br>$\boldsymbol{77.8\%}$ | $84.3\%$<br>$84.3\%$<br>$83.8\%$<br>$82.9\%$<br>$87.3\%$<br>$86.8\%$<br>$\boldsymbol{87.8\%}$ | 


> Training details
> * on CIFAR-10, KLNet training without weight decay, then add it on second cloning phase and disable data augmentation
> * same for ConfidNet on CIFAR-10


References
----------

{% bibliography --cited %}