# VolaDiff

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Poetry](https://img.shields.io/badge/poetry-1.5-blue?logo=poetry)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
### *Generative Implied Volatility Modelling in the Era of 0DTE*

<p align="center">
  <img src="https://github.com/user-attachments/assets/3bad2197-665e-4a60-9bbb-f6347737f783" alt="VolaDiff Demo"><br>
  <strong>Three example diffusions generated by VolaDiff.</strong>
</p>

Since the introduction of **0DTE SPX** options, modelling the SPX volatility surface has become incredibly complex due to highly opinionated IV curve shapes and tight bid-ask spreads. This repo contains the code accompanying my ETH Mathematics Master's thesis, which investigates the daily variation of the volatility surface by decomposing its dynamics into an endogenous maturation of the surface, and an exogenous volatility shock. *VolaDiff* is a diffusion model capable of learning the distribution of implied total variance shocks since 2022 by leveraging a rigorous approach to unconditional diffusion. A novel conditioning mechanism allows *VolaDiff* to condition on **arbitrary alpha signals**: by feeding VolaDiff a data loader which resamples from the k-nearest neighbours of any target condition, it generates sensible, arbitrage-free generalisations of observed market conditions. These forecasts can be used by SPX/NDX/RUT market makers to **assess vega exposure** across the entire surface, **forecast VIX futures and volatility derivatives**, and perform **scenario generation**.

Various technical details regarding the diffusion architecture and a precise definition of volatility shocks can be found in the [VolaDiff Thesis PDF](https://github.com/JPNotleks/VolaDiff/blob/main/VolaDiff%20Thesis.pdf). Installation and usage are straightforward; see the [**main.ipynb**](https://github.com/JPNotleks/VolaDiff/blob/main/main.ipynb) file for a precise demo of how to train and sample from the diffusion model.

Data was generously sponsored by Option Research and Technology Services ([ORATS](https://orats.com/)).
