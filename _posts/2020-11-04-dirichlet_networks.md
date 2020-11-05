---
title: "Dirichlet Networks"
date: 2020-11-04
permalink: /posts/2020/11/dirichlet-networks/
tags:
  - Dirichlet Networks
---

Before detailing Dirichlet-based networks, we first review the Dirichlet-multinomial model, a classic problem in machine learning which will help to better understand the intution behind the method.


Preliminaries: the Dirichlet-multinomial model
------

Let us take the problem of infering the probability that a dice with $K$ sides comes up as face $c$. (This example is heavily inspired by Section 3.4 of {% cite murphy2013machine %}'s book).

Suppose we observe $N$ dice rolls. As we always roll the same dice, our training dataset consists of $$\mathcal{D} = \{(x, y_i)\}_{i=1}^N$$ where $$y_i \in \{1,...,K\}$$. If we assume data is *i.i.d*, the likelihood has the form:

$$
\begin{equation}
p(\mathcal{D} \vert \boldsymbol{\theta}) = \mathrm{Cat}(\mathcal{D} \vert \boldsymbol{\theta}) \prod_{c=1}^K \theta_c^{N_c}
\label{eq:likelihood_multinomial}
\end{equation}
$$

where $$N_c = \sum_{i=1}^N \mathbb{I}(y_i = c)$$ is the number of observations of class $c$ among the $N$ dice rolls (sufficient statistics).

For its conjugate properties with categorical distributions, we chose to model the prior distribution as a Dirichlet distribution:

$$
\begin{equation}
p(\boldsymbol{\theta}) = \mathrm{Dir} \big (\boldsymbol{\theta} \vert \boldsymbol{\alpha} \big ) =  \frac{\Gamma(\alpha_0)}{\prod_c \Gamma(\alpha_c)} \prod_{c=1}^K \theta_c^{\alpha_c- 1}
\end{equation}
$$

Multiplying the likelihood by the prior, we find that the posterior is also Dirichlet :
$$
\begin{align}
p(\boldsymbol{\theta} \vert \mathcal{D}) &\propto p(\mathcal{D} \vert \boldsymbol{\theta}) p(\boldsymbol{\theta}) \\
    &\propto \mathrm{Cat}(\mathcal{D} \vert \boldsymbol{\theta}) \mathrm{Dir} \big (\boldsymbol{\theta} \vert \boldsymbol{\alpha} \big ) \\
    &= \prod_{c=1}^K \theta_c^{N_c} \theta_c^{\alpha_c - 1} \\
    &= \mathrm{Dir} \big (\boldsymbol{\theta} \vert \alpha_1 + N_1,..., \alpha_K + N_K \big )
\end{align}
$$

Now, what we're interested is to compute the posterior predictive distribution for a single multinouilli trial:

$$
\begin{align}
p(y=c \vert x, \mathcal{D}) &= \int p(y=c \vert \boldsymbol{\theta}) p(\boldsymbol{\theta} \vert \mathcal{D}) d\boldsymbol{\theta} \\
    &= \int \theta_c p(\theta_c \vert \mathcal{D}) d\theta_c \\
    &= \mathbb{E} \big [\theta_c \vert \mathcal{D}] = \frac{\alpha_c + N_c}{\alpha_0 + N}
\end{align}
$$

We observe that the prior distribution acts as a **Bayesian smoothing** by adding pseudo-count $\boldsymbol{\alpha}$ to the true count.



Dirichlet Network, a Bayesian approach
------

We extend the Bayesian treatment of a single categorical distribution to classification. In classification tasks, we predict the class label $y_i$ from a different categorical distribution for each input $\boldsymbol{x}_i$. Dataset is known a collection of *i.i.d* samples $$\mathcal{D} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^N$$.

Let us denote $\boldsymbol{\pi} =[\pi_1,...,\pi_K]$ the random variable of categorical probabilities over labels on a sample $\boldsymbol{x}$. The likelihood for a sample $\boldsymbol{x}_i$ has the form:

$$
\begin{equation}
p(\mathcal{D} \vert \boldsymbol{\pi}) = \mathrm{Cat}(\mathcal{D} \vert \boldsymbol{\pi}) \prod_{c=1}^K \pi_c^{\tilde{N}_c}
\end{equation}
$$

The difference with Equation (\ref{eq:likelihood_multinomial}) is that $$\tilde{N}_c$$ now represents a label frequency count at point $$\boldsymbol{x}$$. Obviously, for an unknown test sample $$\boldsymbol{x}^*$$, we don't have access to its label frequency count when infering. Consequently, we are not able to estimate the posterior predictive distribution:

$$
\begin{align}
p(y=c \vert x^*, \mathcal{D}) &= \int p(y=c \vert \boldsymbol{\pi}) p(\boldsymbol{\pi} \vert \boldsymbol{x}^*, \mathcal{D}) d\boldsymbol{\pi} \\
    &= \mathbb{E}_{p(\boldsymbol{\pi} \vert \boldsymbol{x}^*, \mathcal{D})} \big [ p(y=c \vert \boldsymbol{\pi}) \big ]
\end{align}
$$

Approaches like ensembles {% cite deepensembles2017 %} and dropout {% cite mcdropout2016 %} model implicity the posterior distribution over categorical probabilities $$p(\boldsymbol{\pi} \vert \boldsymbol{x}^*, \mathcal{D})$ and estimate the predictive distribution thanks to Monte-Carlo Sampling. **With Dirichlet models, we now explicitly parametrizes $p(\boldsymbol{\pi} \vert \boldsymbol{x}^*, \mathcal{D})$ with a Dirichlet distribution.** This effectively emulate an ensemble without sampling approximation, thanks a closed-form, and which requires only one forward pass.

In particular, Dirichlet networks {% cite malinin2018 sensoy2018 malinin2019 vardir2019 beingbayesian2020 maximize-representation-gap2020 %} enable to account distinctly the aleatoric aleatoric and the epistemic uncertainty on a sample. The aleatoric uncertainty is irreducible from the data due to class overlap or noise, *e.g.* a fair coin has 50/50 chance for head. The epistemic uncertainty is due to the lack of knowledge about unseen data, *e.g.* an image of an unknown object or an outlier in the data.

Epistemic uncertainty relates to the spread of the categorical disribution $$p(\boldsymbol{\pi} \vert \boldsymbol{x}^*, \mathcal{D})$$ on the simplex, which corresponds to $$\alpha_0 = \sum_{c=1}^K \alpha_c$$ for a Dirichlet distribution. Aleatoric uncertainty is linked to the position of the mean on the simplex. Equipped with this configuration, we would like Dirichlet network to yield desired behaviors shown in the figure below:

{:refdef: style="text-align: center;"}
![desired_behavior](/images/desired_behavior.png)
{: refdef}

When it is confident in its prediction a Dirichlet network should yield a sharp distribution centered on one of the corners of the simplex (*Fig a.*). For an input in a region with high degrees of noise or class overlap (aleatoric uncertainty), it should yield a sharp distribution focused on the center of the simplex, which corresponds to being confident in predicting a flat categorical distribution over class labels (known-unknown) (*Fig b.*). Finally, for out-of-distribution inputs the Dirichlet Network should yield a flat distribution over the simplex, indicating large epistemic uncertainty (unknown-unknown) (*Fig c.*).


Now the remaining question are:
* **how to induce such desired behavior when training Dirichlet networks?**
* **what measure shoulw we use for each type of uncertainty?**

In the rest of this post, we will review approaches proposed in the recent litterature.


Prior Networks
------




References
----------

{% bibliography --cited %}