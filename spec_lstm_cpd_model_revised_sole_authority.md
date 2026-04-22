# Spec — LSTM CPD Model from *Slow Momentum with Fast Reversion*

## 0. Scope, provenance, and rules

This spec covers **only** the paper’s **LSTM + CPD** model: an online Gaussian-process changepoint detector whose outputs are appended to a Deep Momentum Network built around an LSTM, trained directly on a Sharpe-ratio objective.

This spec does **not** include:

- classical benchmark strategies except where they define shared return notation,
- any handcrafted regime classifier,
- the “multiple CPD modules in parallel” variant the paper says performed worse,
- the transaction-cost-adjusted loss as a default training objective,
- any details borrowed from paper [1] unless they are explicitly restated in this paper.

Whenever the paper delegates details elsewhere or leaves something ambiguous, this spec marks it as **Not specified in paper** instead of filling it in.

**Section 14 is normative and overrides earlier paper-ambiguity notes.**

---

## 1. Model identity and intended behavior

The model is a **two-stage pipeline**:

1. **CPD module**  
   For each asset and each time step, fit a Gaussian-process changepoint model on a rolling return window and output:

   - a **changepoint severity score** \(\nu_t^{(i)} \in (0,1)\),
   - a **normalized changepoint location** \(\gamma_t^{(i)} \in (0,1)\).

2. **LSTM Deep Momentum Network (DMN)**  
   Feed the CPD outputs, together with normalized multi-horizon returns and MACD indicators, into an LSTM.  
   The LSTM output is passed through a **time-distributed dense layer with tanh activation** to produce a continuous position \(X_t^{(i)} \in (-1,1)\).

The paper’s key claim is that the LSTM learns to balance:

- **slow momentum** over longer trends, and
- **fast mean reversion** around local disturbances,

using the CPD features as continuous signals rather than hard regime labels. There is **no explicit slow/fast branch** in the architecture and **no hard regime-switch rule** in the final model.

---

## 2. High-level pipeline

```text
Daily close prices
    -> arithmetic returns
    -> volatility estimate
    -> winsorization (feature-level; see Section 14)
    -> CPD rolling GP fit on standardized return windows
    -> per-timestep feature vector:
         [5 normalized returns, 3 MACDs, 2 CPD features]
    -> non-overlapping sequences of length 63
    -> stateless single-layer LSTM
    -> time-distributed dense(tanh)
    -> position sequence in (-1, 1)
    -> next-period volatility-scaled returns
    -> Sharpe loss over all asset-time pairs
```

Important structural point: the CPD module is **precomputed** and fed into the LSTM as features. The paper does **not** jointly backpropagate through the GP fits. The coupling between CPD and LSTM is through the chosen CPD lookback window \(l\) and the outer hyperparameter search.

---

## 3. Data contract from the paper

### 3.1 Raw input data

For asset \(i\), the paper uses a sequence of **daily closing prices**:

$$
\{p_t^{(i)}\}_{t=1}^{T}
$$

The paper does **not** use OHLC feature engineering; it formulates everything from **closing price** and derived returns.  
So the paper-defined model is a **daily-close** model, not an intraday or full-OHLC model.

### 3.2 One-day arithmetic returns

The paper defines arithmetic return as:

$$
r_{t-1,t}^{(i)} = \frac{p_t^{(i)} - p_{t-1}^{(i)}}{p_{t-1}^{(i)}}
$$

Then it abbreviates \(r_{t-1,t}\) as \(r_t\).  
This is important: the paper uses **arithmetic returns**, not log returns.

### 3.3 Ex-ante volatility estimate

The per-asset ex-ante volatility estimate \(\sigma_t^{(i)}\) is computed with a:

- **60-day exponentially weighted moving standard deviation**.

This volatility is used both:

- to normalize some model inputs, and
- to scale realized strategy returns.

### 3.4 Winsorization

Appendix B states:

- “We winsorise our data by limiting it to be within **5 times** its exponentially weighted moving standard deviations from its exponentially weighted moving average, using a **252-day half life**.”

Per Section 14, this is resolved to apply at the **feature level** to the five normalized-return features and three MACD features, not to raw prices.

### 3.5 Chronological split

For each asset:

- first **90%** of observations -> training,
- last **10%** -> validation.

This is explicitly chronological, not random.

---

## 4. Per-timestep feature specification

At each timestep, the LSTM receives:

### 4.1 Normalized multi-horizon returns (5 features)

The paper states:

$$
r_{t-t',t}^{(i)} / \sigma_t^{(i)} \sqrt{t'}
$$

for offsets

$$
t' \in \{1, 21, 63, 126, 256\}
$$

These correspond to:

- daily,
- monthly,
- quarterly,
- biannual,
- annual horizons.

#### Final specification

**Status: paper-implied but unresolved, resolved by justified replication choice.**

For each horizon \(t'\), define the normalized return feature as:

$$
z_{t,t'}^{(i)} = \frac{r_{t-t',t}^{(i)}}{\sigma_t^{(i)}\sqrt{t'}}
$$

with the multi-period return defined as the **arithmetic close-to-close interval return**:

$$
r_{t-t',t}^{(i)} := \frac{p_t^{(i)}}{p_{t-t'}^{(i)}} - 1
$$

where \(p_t^{(i)}\) denotes the **daily close price** and \(t'\) counts **trading-day observations**, not calendar days.

An equivalent implementation may compute the same quantity by compounding daily arithmetic returns over the interval:

$$
r_{t-t',t}^{(i)}
=
\prod_{u=t-t'+1}^{t} \left(1+r_{u-1,u}^{(i)}\right)-1
$$

This is algebraically equivalent to the close-to-close interval-return definition above and is therefore permitted as an implementation form, but the canonical specification is:

$$
r_{t-t',t}^{(i)} := \frac{p_t^{(i)}}{p_{t-t'}^{(i)}} - 1
$$

#### Important mismatch to preserve

The input feature set uses an **annual offset of 256 days**, while a benchmark strategy elsewhere in the paper uses **252-day annual return**. That discrepancy is in the paper and should not be silently normalized away.

### 4.2 MACD indicators (3 features)

The paper appends **three MACD indicators** with short/long pairs:

$$
(8,24),\ (16,28),\ (32,96)
$$

#### Final specification

**Status: paper-implied but unresolved, resolved by justified replication choice.**

For each pair \((S,L)\in\{(8,24),(16,28),(32,96)\}\), define \(m_t^{(i)}(x)\) as the exponentially weighted moving average of daily close prices with time-scale \(x\), where

$$
HL(x)=\frac{\log(0.5)}{\log(1-1/x)}.
$$

Define

$$
MACD_t^{(i)}(S,L)=m_t^{(i)}(S)-m_t^{(i)}(L),
$$

$$
q_t^{(i)}(S,L)=\frac{MACD_t^{(i)}(S,L)}{\operatorname{std}(p_{t-63:t}^{(i)})},
$$

and the final MACD input feature

$$
Y_t^{(i)}(S,L)=\frac{q_t^{(i)}(S,L)}{\operatorname{std}(q_{t-252:t}^{(i)}(S,L))}.
$$

The three MACD indicators fed to the LSTM are \(Y_t^{(i)}(8,24)\), \(Y_t^{(i)}(16,28)\), and \(Y_t^{(i)}(32,96)\).

### 4.3 CPD outputs (2 features)

For a chosen changepoint lookback window \(l\), append:

- severity \(\nu_t^{(i)}\),
- location \(\gamma_t^{(i)}\).

### 4.4 Feature dimension

Assuming one scalar per listed component, the per-timestep feature vector has:

- 5 normalized return features
- 3 MACD features
- 2 CPD features

Total:

$$
10 \text{ features per timestep}
$$

This feature count is a direct implication of the paper, even though the paper never says “10-dimensional input” explicitly.

---

## 5. CPD module specification

## 5.1 Windowing

For CPD, the paper does not use the full history.  
For each end time \(T\), it considers a rolling return window:

$$
\{r_t^{(i)}\}_{t=T-l}^{T}
$$

where \(l\) is the CPD lookback window (LBW).

The paper’s candidate LBW values are:

$$
l \in \{10, 21, 63, 126, 252\}
$$

Interpretation:

- 10 days,
- 21 days,
- 63 days,
- 126 days,
- 252 days.

The paper says these were chosen to line up with return timescales, except 10 days, which was chosen as the shortest practical daily-scale CPD window.

## 5.2 Window standardization

Within each CPD window, returns are standardized as:

$$
\hat r_t^{(i)} =
\frac{
 r_t^{(i)} - E_T[r_t^{(i)}]
}{
\sqrt{\mathrm{Var}_T[r_t^{(i)}]}
}
$$

So each CPD fit uses a local window with:

- zero-centered mean,
- unit variance.

The paper says this is done:

- to assume zero mean over the window,
- and to make CPD outputs more consistent across windows/assets.

## 5.3 Baseline GP (no changepoint)

The baseline GP regression model is:

$$
\hat r_t^{(i)} = f(t) + \epsilon_t,
\quad
f \sim GP(0, k_\xi),
\quad
\epsilon_t \sim \mathcal N(0,\sigma_n^2)
$$

The baseline kernel is **Matérn 3/2**:

$$
k(x, x') =
\sigma_h^2
\left(
1 + \frac{\sqrt{3}|x-x'|}{\lambda}
\right)
\exp\left(
-\frac{\sqrt{3}|x-x'|}{\lambda}
\right)
$$

Baseline kernel hyperparameters:

$$
\xi_M = (\lambda, \sigma_h, \sigma_n)
$$

with:

- \(\lambda\): input scale,
- \(\sigma_h\): output scale,
- \(\sigma_n\): observation noise scale.

Covariance matrix:

$$
K(x,x)
$$

and

$$
V = K + \sigma_n^2 I
$$

The paper fits these by minimizing the negative log marginal likelihood:

$$
\mathrm{nlml}_\xi
=
\min_\xi
\left(
\frac{1}{2}\hat r^T V^{-1}\hat r
+
\frac{1}{2}\log|V|
+
\frac{l+1}{2}\log 2\pi
\right)
$$

Optimization details:

- framework: **GPflow**
- optimizer: **L-BFGS-B**
- via: `scipy.optimize.minimize`

## 5.4 Conceptual changepoint model

The paper first introduces a conceptual region-switching kernel with a single changepoint \(c\), assuming:

- one covariance function before \(c\),
- another after \(c\),
- observations before \(c\) become uninformative about observations after \(c\).

This conceptual model assumes:

- the lookback window contains **exactly one changepoint**.

That single-changepoint assumption is important and explicit.

## 5.5 Actual implemented changepoint kernel

The paper says directly fitting the discrete changepoint \(c\) is inefficient, so the implemented kernel is a **sigmoid-approximated changepoint kernel**.

Define:

$$
\sigma(x) = \frac{1}{1 + e^{-s(x-c)}}
$$

with:

- \(c \in (t-l, t)\): changepoint location,
- \(s > 0\): steepness.

The implemented kernel is:

$$
k_{\xi_C}(x,x')
=
k_{\xi_1}(x,x')\,\sigma(x)\sigma(x')
+
k_{\xi_2}(x,x')\,(1-\sigma(x))(1-\sigma(x'))
$$

The paper denotes the second weighting term with \(\bar\sigma(x,x')\), but functionally it is the complement weighting for the post/pre regions.

Full changepoint-kernel parameter set:

$$
\xi_C = \{\xi_1,\xi_2,c,s,\sigma_n\}
$$

where each of \(\xi_1, \xi_2\) parameterizes a Matérn 3/2 kernel.

Implementation details:

- GPflow class: `gpflow.kernels.ChangePoints`
- additional constraint: force \(c \in (t-l, t)\), because GPflow does **not** enforce this by default.

## 5.6 CPD outputs

The paper defines:

### Severity score

$$
\nu_t^{(i)}
=
1 - \frac{1}{1 + e^{-(\mathrm{nlml}_{\xi_C}-\mathrm{nlml}_{\xi_M})}}
$$

### Normalized changepoint location

$$
\gamma_t^{(i)}
=
\frac{c-(t-l)}{l}
$$

The paper says both are normalized to improve LSTM stability/performance.

### Important notation issue

In the printed equation, the symbol appears as `nlmn` in some renderings, but the earlier equation defines `nlml`.  
This looks like a typographical/typesetting inconsistency.  
For implementation, treat Eq. (10) as using the same negative log marginal likelihood from Eq. (7), but mark this as a **paper notation inconsistency** rather than an invented correction.

## 5.7 Per-timestep GP fitting procedure

For each asset-time pair and chosen \(l\):

1. Build the standardized return window.
2. Fit the baseline Matérn 3/2 GP.
3. Initialize the changepoint GP.
4. Fit the changepoint GP.
5. Compute \(\nu_t^{(i)}\) and \(\gamma_t^{(i)}\).

The paper’s initialization procedure is explicit:

### Baseline Matérn GP initialization

For **every timestep**:

- reinitialize **all** Matérn hyperparameters to **1**.

The paper explicitly says this was more stable than warm-starting from the previous timestep.

### Changepoint GP initialization

Initialize:

- \(c = t - \frac{l}{2}\)
- \(s = 1\)

Initialize all other changepoint-kernel parameters from the already fitted baseline Matérn GP, with:

- \(k_{\xi_1}\) and \(k_{\xi_2}\) both starting from the same baseline values.

### Failure handling

If the changepoint fit fails:

- retry after reinitializing **all** changepoint-kernel parameters to **1**,
- except keep \(c = t - \frac{l}{2}\).

If it still fails:

- fill \(\nu_t^{(i)}\) and \(\gamma_t^{(i)}\) using the **previous timestep’s outputs**,
- while “incrementing the changepoint location by an additional step.”

### Ambiguity in fallback rule

Because \(\gamma_t^{(i)}\) is already a **normalized** location, the paper’s “increment by one step” instruction is not fully operationally specified in normalized coordinates.  
So the precise fallback computation for \(\gamma_t^{(i)}\) is **Not fully specified in paper; Section 14 provides the normative closure used by this spec**.

---

## 6. LSTM DMN specification

## 6.1 Core architecture

The paper’s deep-learning model is:

- an **LSTM**,
- followed by a **time-distributed fully connected layer**,
- with **tanh activation**.

The dense layer maps to:

$$
X_t^{(i)} \in (-1,1)
$$

So the model outputs a **continuous position size**, not a discrete label and not a discrete long/short signal.

## 6.2 Sequence formulation

The paper defines:

$$
X_{T-\tau+1:T}^{(i)} = g(u_{T-\tau+1:T}^{(i)};\theta)
$$

where:

- \(u_{T-\tau+1:T}^{(i)}\): input feature sequence,
- \(X_{T-\tau+1:T}^{(i)}\): output position sequence,
- \(\tau\): sequence length,
- \(\theta\): trainable network parameters.

For online prediction, the paper says only the **final** element

$$
X_T^{(i)}
$$

is used for trading.

## 6.3 Sequence length

Appendix B sets:

$$
\tau = 63
$$

for **all experiments**.

## 6.4 Stateful vs stateless

The model uses a **stateless LSTM**:

- final hidden/cell state from one batch is **not** reused as the initial state for the next batch.

## 6.5 Sequence construction for training

For train/validation:

- sequences are **non-overlapping**,
- not sliding-window sequences.

The paper says this was chosen to help reduce overfitting.

Within each sequence:

- chronological order is preserved.

Across sequences:

- sequence order is shuffled each epoch.

### Not specified in paper

The paper does not specify:

- how leftover samples shorter than 63 are handled,
- whether sequences are built independently per asset before mixing,
- the exact batching layout across assets.

These details are not fully specified by the paper itself. Section 14 normatively closes the handling of leftover fragments shorter than 63; the remaining batching/layout details stay unspecified unless explicitly closed elsewhere in this spec.

## 6.6 Dropout

The paper applies dropout regularization:

- to **LSTM inputs**
- and **LSTM outputs**

Dropout-rate search grid:

$$
\{0.1, 0.2, 0.3, 0.4, 0.5\}
$$

#### Final specification

**Status: paper-implied but unresolved, resolved by justified replication choice.**

Let \(d\) denote the Dropout Rate hyperparameter. During training only, apply dropout with rate \(d\) at two sites: (1) on the inputs to the single LSTM layer and (2) on the sequence outputs of that LSTM immediately before the final time-distributed dense output layer. At each site, sample one dropout mask per sequence and reuse that same mask across all timesteps within the sequence. Do not apply recurrent-state dropout. Disable dropout at validation and inference time.

## 6.7 Hidden size

Hyperparameter search includes:

$$
\{5, 10, 20, 40, 80, 160\}
$$

The table calls this **Hidden Layer Size**.

#### Final specification

**Status: paper-implied but unresolved, resolved by justified replication choice.**

Hidden Layer Size denotes the number of units in the single LSTM layer, i.e. the dimensionality \(H\) of the LSTM hidden state \(h_t\) and cell state \(c_t\). The searched values \(\{5,10,20,40,80,160\}\) therefore parameterize \(H\) directly.

## 6.8 Derived tensor shapes

Directly implied by the paper:

- per timestep feature count: **10**
- sequence length: **63**
- output per timestep: **1 scalar position**

So a natural sequence shape is:

- input: `[batch, 63, 10]`
- output: `[batch, 63, 1]`

This is a direct inference, not an explicit sentence in the paper.

---

## 7. Return definition and loss function

## 7.1 Realized return used by the strategy

The paper defines per-asset next-step realized return:

$$
R_{t+1}^{(i)} = X_t^{(i)} \frac{\sigma_{tgt}}{\sigma_t^{(i)}} r_{t+1}^{(i)}
$$

Portfolio average:

$$
R_{t+1}^{TSMOM} = \frac{1}{N}\sum_{i=1}^{N} R_{t+1}^{(i)}
$$

with:

- \(N\): number of assets,
- \(\sigma_{tgt} = 15\%\) annualized target volatility,
- \(\sigma_t^{(i)}\): 60-day EWM volatility estimate.

This means:

- output positions are volatility-scaled before contributing to realized return,
- assets are equally averaged at the portfolio layer.

## 7.2 Sharpe loss

The training loss is:

$$
\mathcal{L}_{sharpe}(\theta)
=
-
\frac{
\sqrt{252}\,E_\Omega[R_t^{(i)}]
}{
\sqrt{\mathrm{Var}_\Omega[R_t^{(i)}]}
}
$$

where:

$$
\Omega = \{(i,t)\mid i \in \{1,\dots,N\},\ t \in \{T-\tau+1,\dots,T\}\}
$$

Important implications:

- the loss is computed over **all asset-time pairs** in the relevant sequence span,
- training is directly on a **risk-adjusted trading objective**,
- there is **no supervised label target** like next-return sign or price class.

The paper states gradients are obtained by automatic differentiation and optimized by backpropagation.

## 7.3 Cross-asset sharing implication

Because the loss is defined jointly over the set \(\Omega\) of asset-time pairs, the paper strongly implies a **single shared network across assets**, rather than one separate LSTM per asset.

This is an inference from the objective and training description; the paper does not spell it out in one sentence.

---

## 8. Training procedure

## 8.1 Optimizer and implementation stack

Training details:

- training method: minibatch SGD
- optimizer: **Adam**
- neural-network framework: **Keras API in TensorFlow**

GP fitting stack:

- **GPflow**
- backed by TensorFlow
- optimization via SciPy L-BFGS-B.

## 8.2 Training budget and stopping

Training budget:

- maximum **300 epochs**

Early stopping:

- patience **25 epochs**
- stop if validation loss does not decrease over that patience window.

## 8.3 Hyperparameter search

The paper uses:

- **50 iterations** of random grid search in an outer loop.

Search grid:

| Hyperparameter    | Search grid                            |
| ----------------- | -------------------------------------- |
| Dropout Rate      | 0.1, 0.2, 0.3, 0.4, 0.5                |
| Hidden Layer Size | 5, 10, 20, 40, 80, 160                 |
| Minibatch Size    | 64, 128, 256                           |
| Learning Rate     | \(10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\) |
| Max Gradient Norm | \(10^{-2}, 10^{0}, 10^{2}\)            |
| CPD LBW Length    | 10, 21, 63, 126, 252                   |

The paper says CPD LBW length can be:

- fixed,
- or treated as a structural hyperparameter.

### Not specified in paper

The paper does not say:

- whether random search samples with or without replacement,
- how ties are broken,
- the random seed,
- or the final selected hyperparameter values for each window.

#### Final specification for gradient clipping

**Status: paper-implied but unresolved, resolved by justified replication choice.**

Interpret Max Gradient Norm as a global gradient-norm clipping threshold \(G\). For each optimization step, compute the full set of parameter gradients for the minibatch, compute their joint L2 norm \(\lVert g\rVert_2\), and, if \(\lVert g\rVert_2>G\), rescale all gradients by the common factor \(G/\lVert g\rVert_2\) before applying the Adam update. Do not use elementwise clipping and do not clip each tensor independently.

## 8.4 Model selection

The paper selects the model with the **lowest validation loss**.

For the “optimized LBW” CPD model:

- the full experiment is run for each CPD LBW choice,
- then the model with the best validation loss is used.

---

## 9. Online inference procedure

After training, the paper’s online procedure is:

1. For the most recent data, compute the CPD outputs \(\nu_t^{(i)}, \gamma_t^{(i)}\).
2. Build the latest LSTM input sequence.
3. Run the LSTM.
4. Take the **final** position \(X_T^{(i)}\).
5. Hold that position for the next day.

This is a strictly causal, online setup.  
A bidirectional LSTM would contradict this online use case, even though the paper never explicitly writes “unidirectional.”

---

## 10. Paper backtest protocol (needed only if you want the paper-style experiment, not just the model)

Paper experiment setup:

- dataset: **50 liquid continuous futures**
- period: **1990–2020**
- out-of-sample test windows:
  - 1995–2000
  - 2000–2005
  - 2005–2010
  - 2010–2015
  - 2015–2020
- expanding-window retraining:
  - 1990–1995 train/validation, test 1995–2000
  - then expand train/validation by 5 years each step
- only use an asset if enough validation data exists for at least **one LSTM sequence**
- results are averaged across test windows.

### Extra evaluation notes

The paper reports:

- raw signal output,
- and a second evaluation with **additional volatility rescaling to 15%**.

### Not specified in paper

The paper does not fully spell out the exact formula for that extra portfolio-level rescaling layer in the result tables.  
Section 14 provides the normative evaluation formula used by this spec for that additional Exhibit 4 rescaling.

---

## 11. What is **not** part of the final model

These are easy implementation traps and should be excluded:

### 11.1 No hard regime labels

“Bull,” “Bear,” “Correction,” and “Rebound” are explanatory language only.  
They are **not** model inputs and **not** supervised labels.

### 11.2 No changepoint-threshold trading rule

Exhibit 5 uses severity thresholds such as \(\nu_t^{(i)} \ge 0.9995\) and \(\nu_t^{(i)} \ge 0.995\) to visualize changepoints.  
Those thresholds are for **plotting/illustration**, not for the production trading rule.

### 11.3 No multiple parallel CPD modules

The paper explicitly says that feeding multiple CPD modules with different LBWs in parallel into the LSTM degraded performance and is not the final design.

### 11.4 No transaction-cost-adjusted loss by default

Main experiments do **not** include transaction costs in the objective.  
The turnover-adjusted loss is discussed only as an optional adjustment for higher-cost settings.

---

## 12. Derived implementation requirements that follow from the paper

These are not all written explicitly, but they follow directly from the paper’s equations and horizons:

### 12.1 Daily frequency assumption is structural

The paper hardcodes daily-market conventions:

- 21-day month,
- 63-day quarter,
- 126-day half-year,
- 252/256-day year,
- \(\sqrt{252}\) annualization.

So the architecture is structurally a **daily-bar** model.

### 12.2 Minimum usable history for a full sequence

With:

- annual return input horizon = 256 days,
- sequence length = 63,

the earliest timestep in a full 63-step sequence needs data going back roughly:

$$
256 + 62 = 318
$$

daily steps before the sequence endpoint.

That is a **derived** requirement, not an explicit paper sentence.

### 12.3 CPD is fit independently at each timestep

Because the paper reinitializes GP parameters to 1 for each timestep and says this is more stable than borrowing previous parameters, the CPD fits are effectively **independent rolling refits**, not persistent stateful GP updates.

---

## 13. Paper-to-project mapping for your FTMO setup

This section is **project overlay**, not paper fact.

Your provided project docs define:

- the FTMO asset universe to use,
- and a directory structure with many timeframes, including daily `D`, on OHLC files.

### 13.1 Closest faithful mapping to the paper

Because the paper is written on **daily closing prices**, the closest faithful FTMO mapping is:

- use the **D** timeframe
- use the **close** column only

Using intraday bars would require redefining:

- all day-based return horizons,
- 60-day volatility,
- 252-day annualization,
- 252-day winsor half-life,
- and the CPD LBW semantics.

That would be an **adaptation**, not a direct replication.

### 13.2 Universe mismatch to flag now

The paper’s experiment uses **50 liquid continuous futures**, while the FTMO universe is a different multi-asset CFD/FX/crypto/equity universe.  
So a future FTMO implementation can be **paper-faithful in model logic** but cannot be a literal reproduction of the paper’s asset universe unless an explicit universe-mapping rule is later defined.

That is a project-level deviation, not a model-level deviation.

---

## 14. Explicit unknowns / unresolved items

| Topic | Status |
| --- | --- |
| Exact MACD formula and normalization | **Closed —** For each pair \((S,L)\in\{(8,24),(16,28),(32,96)\}\), define \(m_t^{(i)}(x)\) as the exponentially weighted moving average of daily close prices with time-scale \(x\), where \(HL(x)=\log(0.5)/\log(1-1/x)\). Define \(MACD_t^{(i)}(S,L)=m_t^{(i)}(S)-m_t^{(i)}(L)\), \(q_t^{(i)}(S,L)=MACD_t^{(i)}(S,L)/\operatorname{std}(p_{t-63:t}^{(i)})\), and \(Y_t^{(i)}(S,L)=q_t^{(i)}(S,L)/\operatorname{std}(q_{t-252:t}^{(i)}(S,L))\). The three MACD indicators fed to the LSTM are \(Y_t^{(i)}(8,24)\), \(Y_t^{(i)}(16,28)\), and \(Y_t^{(i)}(32,96)\). |
| Whether winsorization applies to prices, returns, or features | **Closed —** Apply winsorization at the feature level, not to raw prices. After constructing the five normalized-return features and the three MACD features for each asset-date pair, causally cap/floor each of those eight feature series to its trailing EWM mean \(\pm 5\) trailing EWM standard deviations, using a 252-day half-life. Do not winsorize raw close prices. Do not winsorize the bounded CPD features \(\nu_t^{(i)}\) and \(\gamma_t^{(i)}\). |
| Exact LSTM layer count | **Closed —** Use exactly one unidirectional, stateless LSTM layer. No additional LSTM layer and no other recurrent layer is present. |
| Exact meaning of “Hidden Layer Size” | **Closed —** Hidden Layer Size denotes the number of units in the single LSTM layer, i.e. the dimensionality \(H\) of the LSTM hidden state \(h_t\) and cell state \(c_t\). The searched values \(\{5,10,20,40,80,160\}\) parameterize \(H\) directly. |
| Any additional dense hidden layers | **Closed —** No additional dense hidden layer shall be inserted between the LSTM and the final output head. The single LSTM layer is followed directly by one time-distributed dense output layer of size 1 with tanh activation. |
| Exact dropout implementation details | **Closed —** Let \(d\) denote the Dropout Rate hyperparameter. During training only, apply dropout with rate \(d\) at two sites: (1) on the inputs to the single LSTM layer and (2) on the sequence outputs of that LSTM immediately before the final time-distributed dense output layer. At each site, sample one dropout mask per sequence and reuse that same mask across all timesteps within the sequence. Do not apply recurrent-state dropout. Disable dropout at validation and inference time. |
| Exact gradient clipping implementation for Max Gradient Norm | **Closed —** Interpret Max Gradient Norm as a global gradient-norm clipping threshold \(G\). For each optimization step, compute the full set of parameter gradients for the minibatch, compute their joint L2 norm \(\lVert g\rVert_2\), and, if \(\lVert g\rVert_2>G\), rescale all gradients by the common factor \(G/\lVert g\rVert_2\) before applying the Adam update. Do not use elementwise clipping and do not clip each tensor independently. |
| Exact GP optimizer bounds/tolerances/restarts | **Closed —** Use GPflow’s SciPy L-BFGS-B optimizer with no user-specified overrides to optimizer bounds, tolerances, or generic restart count. Impose only the paper-specified changepoint-location constraint \(c \in (t-l,t)\). Perform one optimization run per current initialization. If the first Changepoint-kernel fit fails, perform exactly one paper-specified reinitialization-and-refit attempt; do not perform any further optimizer restarts or multistart searches. |
| Exact operational fallback rule for normalized \(\gamma_t\) after GP failure | **Closed —** If the Changepoint-kernel fit still fails after the paper’s second initialization attempt, set \(\nu_t^{(i)} := \nu_{t-1}^{(i)}\) and \(\gamma_t^{(i)} := \gamma_{t-1}^{(i)}\). The paper’s required one-step increment applies to the latent raw changepoint location \(c\); under Eq. (10), that increment leaves the normalized feature \(\gamma_t^{(i)}\) unchanged after the CPD window also advances by one step. |
| Handling of leftover sequence fragments shorter than 63 | **Closed —** For training and validation, partition each per-asset chronological split into contiguous non-overlapping sequences of exactly \(\tau=63\) timesteps. Discard any terminal fragment shorter than 63. Do not pad, overlap, wrap, stitch across boundaries, or create variable-length sequences. |
| Missing-data imputation/cleaning policy beyond paper’s asset-level comments | **Closed —** No missing-data imputation shall be performed. Do not forward-fill, back-fill, interpolate, or zero-fill missing closes, returns, features, or CPD outputs. Compute returns, features, CPD outputs, and next-step targets only where all required upstream observations exist. Drop any timestep with missing required inputs or target values, do not bridge across missing gaps when forming sequences, and discard any 63-step sequence that contains such a gap. Retain an asset in a given expanding window only if, after lookback requirements and missing-data filtering, the validation split still contains at least one full 63-step sequence. |
| Exact formula for the additional rescaling used in Exhibit 4 | **Closed —** For Exhibit 4 evaluation only, apply an additional constant volatility rescaling at the test-window level. For strategy \(s\) in test window \(w\), let \(R_{t,w}^{(s)}\) be the raw daily portfolio return series and \(\hat{\sigma}_{w}^{(s)}=\sqrt{252}\,\operatorname{std}(R_{t,w}^{(s)})\) its realized annualized volatility. Define \(k_{w}^{(s)}=0.15/\hat{\sigma}_{w}^{(s)}\) and \(\tilde R_{t,w}^{(s)}=k_{w}^{(s)}R_{t,w}^{(s)}\). Compute Exhibit 4 metrics on \(\tilde R_{t,w}^{(s)}\), then average metrics across windows exactly as in the raw-signal evaluation. |
| Exact best hyperparameter values per expanding window | **Closed —** The paper does not report numeric winning hyperparameter vectors for the individual expanding windows, and the spec shall not invent them. For each expanding window, define the selected hyperparameter vector as the configuration from that window’s paper-specified 50-iteration random search that achieved the minimum validation loss in that window. For the optimized-LBW model, define the selected CPD LBW as the LBW choice whose corresponding searched model achieved the minimum validation loss in that window. |

---

## 15. Final replication invariants

A later implementation that claims to follow this spec must preserve all of the following unless an explicit deviation is declared:

1. **Daily close-based arithmetic returns** as the primitive series.
2. **60-day EWM volatility** for scaling.
3. Input features are exactly:
   - 5 normalized return horizons,
   - 3 MACDs,
   - 2 CPD outputs.
4. **One CPD module only**, using a single LBW at a time.
5. **One stateless unidirectional LSTM layer**.
6. **Time-distributed tanh output** for position sizing.
7. **Sharpe-ratio loss** on volatility-scaled future returns.
8. **Non-overlapping 63-step sequences** for training/validation.
9. **Chronological 90/10 split** per asset.
10. **Expanding-window evaluation** if reproducing the paper’s backtest.

Anything that changes these items is not a strict replication of the paper’s LSTM+CPD model.

