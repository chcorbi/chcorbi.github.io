---
title: "Preparation on Dirichlet - ICML 2020"
date: 2020-11-12
permalink: /posts/2020/11/dirichlet-icml/
tags:
  - Dirichlet Networks
---


Experiments <a name="experiments"></a>
------

Two models were trained:
* with standard cross-entropy (**XE**)
* with contrastive reverse KL-divergence (**Dirichlet**)


### CIFAR-10 

* In-distribution dataset: CIFAR-10
* Out-distribution training dataset: CIFAR-100
* Network Architecture: VGG-16
* Concentrations parameters $\alpha= \exp{f(x, \theta)}$
* Target concentrations for in-domain: $$\beta_{\text{in}}$$ = 100
* Training details: Adam, LR 5e-5, 1-cyclic scheduler for 45 epochs

Accuracy is 93.5% for XE model and 93.0% for Dirichlet model.

Presented results are for **TinyImageNet** as OOD dataset (% AUC)

| Training | Method | Mis. Detection | OOD Detection | Mis.+OOD Detection |
| :------: |:-------| --------------:| -------------:| ------------------:|
| **XE**   | MCP<br>ODIN<br>Mahalanobis | $\boldsymbol{92.6\%}$<br>$91.4\%$<br>$90.2\%$ | $86.8\%$<br>$88.2\%$<br>$83.0\%$ | $91.2\%$<br>$91.1\%$<br>$87.8\%$ |
| **Dirichlet** | MCP<br>ODIN<br>Mutual Information<br>Mahalanobis<br>KL\_Pred\_Full<br>KLNet_MSE<br>KLNet_MSE_Cloning | $91.6\%$<br>$91.6\%$<br>$91.2\%$<br>$92.1\%$<br>$92.0\%$<br>$92.3\%$<br>$\boldsymbol{92.5\%}$  | $92.8\%$<br>$\boldsymbol{92.9\%}$<br>$\boldsymbol{92.9\%}$<br>$86.6\%$<br>$92.4\%$<br>$92.7\%$<br>$92.8\%$<br> | $93.1\%$<br>$93.1\%$<br>$92.9\%$<br>$91.1\%$<br>$93.1\%$<br>$93.4\%$<br>$\boldsymbol{93.5\%}$ | 



### CIFAR-100

* In-distribution dataset: CIFAR-100
* Out-distribution training dataset: CIFAR-10
* Network Architecture: VGG-16
* Concentrations parameters $\alpha= \exp{f(x, \theta)}$
* Target concentrations for in-domain: $$\beta_{\text{in}}$$ = 100
* Training details: Adam, LR 5e-5, 1-cyclic scheduler for 45 epochs

Accuracy is 73.2% for XE model and 72.0% for Dirichlet model.

Presented results are for **TinyImageNet** as OOD dataset (% AUC)

| Training | Method | Mis. Detection | OOD Detection | Mis.+OOD Detection |
| :------: |:-------| --------------:| -------------:| ------------------:|
| **XE**   | MCP<br>ODIN<br>Mahalanobis | $\boldsymbol{87.3\%}$<br>$85.7\%$<br>$81.8\%$ | $75.9\%$<br>$\boldsymbol{77.7\%}$<br>$74.3\%$ | $86.5\%$<br>$86.1\%$<br>$82.2\%$ |
| **Dirichlet** | MCP<br>ODIN<br>Mutual Information<br>Mahalanobis<br>KL\_Pred\_Full<br>KLNet_MSE<br>KLNet_MSE_Cloning | $84.0\%$<br>$83.9\%$<br>$83.3\%$<br>$87.2\%$<br>$\boldsymbol{88.0\%}$<br><br><br>  | $75.9\%$<br>$75.8\%$<br>$75.8\%$<br>$74.1\%$<br>$77.5\%$<br><br><br> | $84.6\%$<br>$84.5\%$<br>$84.2\%$<br>$86.7\%$<br>$\boldsymbol{87.8\%}$<br><br><br> | 