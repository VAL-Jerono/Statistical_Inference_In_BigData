# DSA 8301: Statistical Inference for Big Data
### A Complete Guide — From First Principles to Mastery

**Strathmore University | Institute of Mathematical Sciences | 2025/2026**
**Lecturer: Prof. Jacob Ong'ala | jongala@strathmore.edu**

---

> This document is a living map of the entire course. Every formula here is not decoration — it is the grammar of statistical reasoning. Read it the way you would read a story: sequentially. The concepts build on each other, and where they connect, the connection is made explicit. By the end, you will not just know statistics. You will think in it.

---

## Table of Contents

1. [Parametric Statistical Models](#topic-1-parametric-statistical-models)
2. [Statistical Estimation and Sampling Distributions](#topic-2-statistical-estimation-and-sampling-distributions)
3. [Properties of Estimators](#topic-3-properties-of-estimators)
4. [Asymptotic Properties of Estimators](#topic-4-asymptotic-properties-of-estimators)
5. [Maximum Likelihood Estimation](#topic-5-maximum-likelihood-estimation)
6. [Statistical Hypothesis Testing](#topic-6-statistical-hypothesis-testing)
7. [Likelihood Inference and Applications](#topic-7-likelihood-inference-and-applications)
8. [Large Sample Theory](#topic-8-large-sample-theory)
9. [Nonparametric Statistical Methods](#topic-9-nonparametric-statistical-methods)
10. [Specialized Topics and Computational Applications](#topic-10-specialized-topics-and-computational-applications)
11. [Quick Reference: Formula Sheet](#quick-reference-formula-sheet)
12. [How the Topics Connect: The Bigger Picture](#how-the-topics-connect-the-bigger-picture)

---

## Preface: What Is Statistical Inference?

Before any formula, a story.

Imagine you are an actuary in Nairobi, and an insurer asks: *"What is the true average claim amount of our policyholders?"* You cannot observe every policyholder — past, present, and future. You have only a sample. From that sample, you must reason about the unseen whole. That reasoning — disciplined, principled, quantifiable — is **statistical inference**.

Statistical inference is the science of learning about a population from a sample, while honestly accounting for uncertainty. It has three major schools of thought, and this course touches all three:

| School | Core Question | Key Tool |
|---|---|---|
| **Frequentist** | What does the data say if the experiment were repeated infinitely? | Confidence intervals, p-values |
| **Likelihood** | For what parameter value does this data make most sense? | Likelihood function, MLE |
| **Bayesian** | What do we believe about the parameter, after seeing data? | Posterior distribution |

All three are frameworks for the same fundamental problem. They differ in *what they assume* and *what they report*. You will encounter all three in this course.

---

## Topic 1: Parametric Statistical Models

### 1.1 The Foundation: What Is a Statistical Model?

Every act of statistical inference begins with a model. A model is a simplified, mathematically precise description of how data might have been generated. We are essentially saying: *"I believe the world worked like this when it produced these numbers."*

A **parametric statistical model** narrows that belief. It says: *"I believe the data came from a specific family of distributions, indexed by a finite number of unknown constants called parameters."*

**Formal Definition.**

Let:
- $\mathcal{X}$ = the **sample space** — the set of all values the data could possibly take
- $\Theta \subseteq \mathbb{R}^p$ = the **parameter space** — the set of all plausible parameter values
- $\theta = (\theta_1, \theta_2, \ldots, \theta_p)$ = the **parameter vector** — the unknown quantities we want to learn

A parametric statistical model is a family of probability distributions:

$$\mathcal{M} = \{ f(x \mid \theta) : \theta \in \Theta \}$$

where $f(x \mid \theta)$ is either a **probability mass function (pmf)** if $\mathcal{X}$ is discrete, or a **probability density function (pdf)** if $\mathcal{X}$ is continuous.

The critical feature: the dimension $p$ of $\theta$ is **fixed** and does not grow with sample size $n$. This separates parametric models from nonparametric ones.

---

### 1.2 The Three Components of Every Parametric Model

Think of a parametric model as a machine with three interchangeable parts:

**1. Sample Space $\mathcal{X}$** — What can the data look like?
- If modeling a coin flip: $\mathcal{X} = \{0, 1\}$
- If modeling a person's height: $\mathcal{X} = \mathbb{R}^+$
- If modeling insurance claim count: $\mathcal{X} = \{0, 1, 2, 3, \ldots\}$

**2. Parameter Space $\Theta$** — What are the admissible parameter values?
- Bernoulli model: $\Theta = (0, 1)$
- Normal model: $\Theta = \{(\mu, \sigma^2) : \mu \in \mathbb{R},\ \sigma^2 > 0\}$
- Poisson model: $\Theta = (0, \infty)$

**3. Family of Distributions** — A separate probability distribution for each $\theta \in \Theta$. Choosing $\theta$ pins down a single distribution.

---

### 1.3 The Major Parametric Families

#### Discrete Distributions

**Bernoulli Distribution**

The simplest model: one trial, two outcomes. If $X \sim \text{Bernoulli}(p)$:

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\},\ p \in (0,1)$$

$$E[X] = p \qquad \text{Var}(X) = p(1-p)$$

*This is the atomic unit from which much of statistics is built. Logistic regression, binary classification, survival endpoints — all reduce to Bernoulli trials at their core.*

---

**Binomial Distribution**

Run $n$ independent Bernoulli trials, each with success probability $p$. Count the successes. If $X \sim \text{Binomial}(n, p)$:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

$$E[X] = np \qquad \text{Var}(X) = np(1-p)$$

*Connection: $\text{Binomial}(1, p) \equiv \text{Bernoulli}(p)$. The Binomial is just repeated Bernoulli.*

---

**Poisson Distribution**

Model the count of rare, independent events in a fixed interval. If $X \sim \text{Poisson}(\lambda)$:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots,\ \lambda > 0$$

$$E[X] = \lambda \qquad \text{Var}(X) = \lambda$$

Note: mean equals variance. This is a diagnostic — real count data with variance $\gg$ mean signals **overdispersion**, suggesting the Poisson may be misspecified.

*Real-world applications: insurance claim frequency, hospital arrivals, network packet counts, road accidents per km.*

---

#### Continuous Distributions

**Normal (Gaussian) Distribution**

The central object of classical statistics. If $X \sim N(\mu, \sigma^2)$:

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad x \in \mathbb{R}$$

$$E[X] = \mu \qquad \text{Var}(X) = \sigma^2$$

$\mu$ is the **location parameter** (shifts the bell left/right). $\sigma^2$ is the **scale parameter** (widens or narrows the bell).

*Why is the Normal everywhere? The Central Limit Theorem (Topic 4) — sums and averages of almost any random variables converge to Normal as sample size grows.*

---

**Exponential Distribution**

Model the waiting time until an event. If $X \sim \text{Exponential}(\lambda)$:

$$f(x) = \lambda e^{-\lambda x}, \quad x > 0,\ \lambda > 0$$

$$E[X] = \frac{1}{\lambda} \qquad \text{Var}(X) = \frac{1}{\lambda^2}$$

**The Memoryless Property** — the defining feature of the Exponential:

$$P(X > s + t \mid X > s) = P(X > t), \quad s, t > 0$$

This says: if you have already waited $s$ units of time for the event, the remaining waiting time has the exact same distribution as if you just started. The past gives you no information. *This is why the Exponential is used for failure times of components that do not "wear out".*

---

**Gamma Distribution**

Generalize the Exponential to model waiting times for the $\alpha$-th event. If $X \sim \text{Gamma}(\alpha, \beta)$:

$$f(x) = \frac{1}{\Gamma(\alpha)\beta^\alpha} x^{\alpha-1} e^{-x/\beta}, \quad x > 0,\ \alpha > 0,\ \beta > 0$$

$$E[X] = \alpha\beta \qquad \text{Var}(X) = \alpha\beta^2$$

where $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t}\,dt$ is the **Gamma function** (a generalized factorial: $\Gamma(n) = (n-1)!$ for positive integers).

**Special Cases of the Gamma — a unifying table:**

| Condition | Resulting Distribution | Interpretation |
|---|---|---|
| $\alpha = 1$ | $\text{Exponential}(\beta)$ | Wait for the 1st event |
| $\alpha = k \in \mathbb{N}$ | $\text{Erlang}(k, \beta)$ | Wait for the $k$-th event |
| $\alpha = \nu/2,\ \beta = 2$ | $\chi^2_\nu$ | Sum of $\nu$ squared standard normals |
| $\alpha \to \infty$ | $\approx N(\alpha\beta, \alpha\beta^2)$ | CLT approximation |

*The Gamma is a parent distribution. It swallows the Exponential and Chi-square as special cases, and connects to the Normal in the limit.*

---

**Chi-Square Distribution**

If $Z_1, Z_2, \ldots, Z_\nu \overset{iid}{\sim} N(0,1)$, then:

$$Q = \sum_{i=1}^{\nu} Z_i^2 \sim \chi^2_\nu$$

$$E[Q] = \nu \qquad \text{Var}(Q) = 2\nu$$

*The Chi-square distribution appears every time you are dealing with sums of squared Normal quantities — which is precisely what happens in variance estimation, goodness-of-fit tests, and likelihood ratio tests.*

---

### 1.4 The Exponential Family: A Unifying Structure

Many distributions above share a hidden algebraic structure. Recognizing it unlocks powerful results.

**General Form.** A distribution belongs to the **exponential family** if its density/mass function can be written as:

$$f(x \mid \theta) = h(x) \exp\!\Big(\eta(\theta)^\top T(x) - A(\theta)\Big)$$

where:
- $T(x)$ = **sufficient statistic** — the only function of data relevant to inference about $\theta$
- $\eta(\theta)$ = **natural (canonical) parameter** — the reparameterized form of $\theta$
- $A(\theta)$ = **log-partition function** — ensures the density integrates to 1; formally $A(\theta) = \log \int h(x) \exp(\eta(\theta)^\top T(x))\,dx$
- $h(x)$ = **base measure** — the part of the density that does not involve $\theta$

**Examples worked through:**

*Bernoulli:*
$$f(x \mid p) = p^x(1-p)^{1-x} = \exp\!\left(x \log\frac{p}{1-p} + \log(1-p)\right)$$
So: $T(x) = x$, $\eta(p) = \log\frac{p}{1-p}$ (the log-odds), $A(p) = -\log(1-p)$

*Poisson:*
$$f(x \mid \lambda) = \frac{\lambda^x e^{-\lambda}}{x!} = \frac{1}{x!}\exp(x\log\lambda - \lambda)$$
So: $T(x) = x$, $\eta(\lambda) = \log\lambda$, $A(\lambda) = \lambda$

*Normal (known $\sigma^2$):*
$$T(x) = x, \quad \eta(\mu) = \frac{\mu}{\sigma^2}, \quad A(\mu) = \frac{\mu^2}{2\sigma^2}$$

**Why the exponential family matters:**

1. **Sufficient statistics have fixed dimension** regardless of sample size $n$
2. **Conjugate priors** exist in Bayesian analysis, giving closed-form posteriors
3. **Generalized Linear Models (GLMs)** — logistic regression (Bernoulli), Poisson regression, linear regression (Normal) are all GLMs because their response distributions belong to this family
4. **Simple log-likelihood structure:** $\ell(\theta) = \eta(\theta)^\top \sum_i T(x_i) - nA(\theta)$
5. **Mean and variance from $A(\theta)$:** $E[T(X)] = \nabla_\eta A$, $\text{Var}[T(X)] = \nabla^2_\eta A$

---

### 1.5 Model Assumptions and Identifiability

#### Core Assumptions

**Distributional Assumption:** Data follow a specific distribution family.
$$X \sim f(x \mid \theta)$$

**Independence (i.i.d.) Assumption:** Observations are independent and identically distributed.
$$X_1, X_2, \ldots, X_n \overset{iid}{\sim} f(x \mid \theta)$$

This gives a product likelihood: $L(\theta) = \prod_{i=1}^n f(x_i \mid \theta)$ — crucial for every result in Topics 2-7.

Violations arise in: time series (today depends on yesterday), spatial data (neighboring plots are similar), clustered observations (patients in the same hospital).

**Homogeneity Assumption:** All observations share the same $\theta$ — the data-generating conditions are identical for each unit. Violations require hierarchical or mixture models.

**Regularity Conditions** (needed for MLE theory):
- $\Theta$ is an open set
- $f(x \mid \theta)$ is differentiable in $\theta$
- Differentiation and integration can be interchanged (Leibniz rule applies)

#### Identifiability: The Crucial Requirement

**Definition.** A model $\mathcal{M} = \{f(x \mid \theta) : \theta \in \Theta\}$ is **identifiable** if:

$$f(x \mid \theta_1) = f(x \mid \theta_2) \text{ for all } x \implies \theta_1 = \theta_2$$

In plain English: different parameter values must produce different distributions. If two $\theta$ values produce the same distribution, they are statistically indistinguishable — no amount of data can tell them apart.

**Identifiable example:** $X \sim N(\mu, \sigma^2)$. If $N(\mu_1, \sigma_1^2) = N(\mu_2, \sigma_2^2)$, then $\mu_1 = \mu_2$ and $\sigma_1^2 = \sigma_2^2$. Identifiable.

**Non-identifiable example:** $X \sim N(\theta_1 + \theta_2, 1)$. The distribution depends only on the sum $\theta_1 + \theta_2$. Infinitely many pairs $(\theta_1, \theta_2)$ give the same distribution: $(2,1)$, $(1,2)$, $(0,3)$. The likelihood surface is a ridge, not a peak.

**Consequences of non-identifiability:**
- MLEs are not unique
- Likelihood has flat regions — numerical optimizers fail or wander
- Confidence intervals become unbounded
- Parameters lose interpretability (label-switching in mixture models)

**Strategies to restore identifiability:**
- Reduce the number of parameters
- Impose constraints (e.g., fix $\theta_1 = 0$ in the sum model)
- Reparameterize (model $\mu = \theta_1 + \theta_2$ directly)
- Use informative priors (Bayesian approach)

---

## Topic 2: Statistical Estimation and Sampling Distributions

### 2.1 The Estimation Problem

You have data $x_1, x_2, \ldots, x_n$ drawn from $f(x \mid \theta)$. The parameter $\theta$ is unknown. **Estimation** is the act of using data to produce a guess for $\theta$.

An **estimator** $\hat\theta$ is a function of the data:

$$\hat\theta = g(X_1, X_2, \ldots, X_n)$$

Note the capital letters: before data is collected, $X_i$ are random variables, so $\hat\theta$ is also a random variable. After data is collected and we substitute actual values $x_i$, we get an **estimate** — a single number.

This distinction — *estimator* (random) vs *estimate* (fixed) — is foundational.

---

### 2.2 Point Estimation

A **point estimator** produces a single number as the guess for $\theta$.

**Example estimators for $\mu$ in a Normal model:**
- Sample mean: $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$
- Sample median: the middle value
- Trimmed mean: the mean after removing extreme values

All three are valid point estimators. Topic 3 gives us tools to compare them.

---

### 2.3 Method of Moments (MoM)

A general technique for deriving estimators, older than MLE and simpler to compute.

**The idea:** Population moments are functions of $\theta$. Replace population moments with sample moments. Solve for $\theta$.

**$k$-th population moment:**
$$\mu_k = E[X^k] = \int x^k f(x \mid \theta)\,dx$$

**$k$-th sample moment:**
$$\hat\mu_k = \frac{1}{n}\sum_{i=1}^n X_i^k$$

For a model with $p$ parameters, set up $p$ equations:
$$\mu_k(\theta) = \hat\mu_k, \quad k = 1, 2, \ldots, p$$
and solve for $\theta$.

**Example:** $X \sim \text{Exponential}(\lambda)$.
- Population moment: $E[X] = 1/\lambda$
- Sample moment: $\bar{X} = \frac{1}{n}\sum X_i$
- MoM estimator: $\hat\lambda_{MoM} = 1/\bar{X}$

**Example:** $X \sim N(\mu, \sigma^2)$ (two parameters).
- $E[X] = \mu \Rightarrow \hat\mu = \bar{X}$
- $E[X^2] = \mu^2 + \sigma^2 \Rightarrow \hat\sigma^2 = \frac{1}{n}\sum X_i^2 - \bar{X}^2$

Note: MoM gives the biased estimator $\hat\sigma^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$. MLE and UMVUE give $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$. The difference matters — see Topic 3.

---

### 2.4 The Sampling Distribution

Every estimator $\hat\theta = g(X_1, \ldots, X_n)$ is a random variable. Its probability distribution is called its **sampling distribution**.

The sampling distribution answers: *"If I repeated this experiment many times and computed $\hat\theta$ each time, what would the histogram of $\hat\theta$ values look like?"*

**The most important sampling distribution result:**

**Theorem.** Let $X_1, \ldots, X_n \overset{iid}{\sim} N(\mu, \sigma^2)$. Then:

$$\bar{X} \sim N\!\left(\mu, \frac{\sigma^2}{n}\right)$$

exactly, for all $n$.

$$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0,1)$$

$$T = \frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}$$

(uses $S^2$ instead of $\sigma^2$, introduces one degree of freedom of uncertainty, gives the $t$-distribution)

$$\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$$

These three results — $Z$, $T$, $\chi^2$ — are the workhorses of classical hypothesis testing in Topics 6 and 7.

---

### 2.5 Standard Error

The **standard error (SE)** of an estimator is the standard deviation of its sampling distribution:

$$\text{SE}(\hat\theta) = \sqrt{\text{Var}(\hat\theta)}$$

For the sample mean: $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$.

The SE quantifies how much the estimator varies across repeated samples. It is the formal measure of **estimation precision**. Narrower SE = more information in the data.

---

## Topic 3: Properties of Estimators

### 3.1 How Do We Choose Between Estimators?

If the sample mean, median, and trimmed mean are all valid estimators of $\mu$, which one should we use? We need criteria. Topic 3 provides four major criteria: **bias**, **consistency**, **efficiency**, and **sufficiency**.

---

### 3.2 Bias

**Definition.** The **bias** of estimator $\hat\theta$ for parameter $\theta$ is:

$$\text{Bias}(\hat\theta) = E[\hat\theta] - \theta$$

An estimator is **unbiased** if $\text{Bias}(\hat\theta) = 0$, i.e., $E[\hat\theta] = \theta$ for all $\theta \in \Theta$.

Unbiasedness means: on average, over repeated experiments, the estimator hits the target. It does not mean it hits the target in any single experiment.

**Example — sample mean is unbiased:**
$$E[\bar{X}] = E\!\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n}(n\mu) = \mu$$

**Example — biased vs unbiased variance estimator:**
$$\hat\sigma^2 = \frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2 \quad \Rightarrow \quad E[\hat\sigma^2] = \frac{n-1}{n}\sigma^2 \neq \sigma^2 \quad \text{(biased)}$$

$$S^2 = \frac{1}{n-1}\sum_{i=1}^n(X_i - \bar{X})^2 \quad \Rightarrow \quad E[S^2] = \sigma^2 \quad \text{(unbiased)}$$

The correction factor $\frac{1}{n-1}$ instead of $\frac{1}{n}$ fixes the bias. The intuition: we lose one degree of freedom because $\bar{X}$ is estimated from the same data.

**Mean Squared Error (MSE)** — combines bias and variance:

$$\text{MSE}(\hat\theta) = E[(\hat\theta - \theta)^2] = \text{Var}(\hat\theta) + [\text{Bias}(\hat\theta)]^2$$

This is the **bias-variance tradeoff** in its purest form. A slightly biased estimator with lower variance can have lower MSE than an unbiased one with high variance. This tradeoff reappears in regularization, ridge regression, and machine learning shrinkage methods.

---

### 3.3 Consistency

**Definition.** An estimator $\hat\theta_n$ is **consistent** for $\theta$ if, as $n \to \infty$:

$$\hat\theta_n \overset{P}{\to} \theta$$

More precisely: for every $\epsilon > 0$,

$$\lim_{n \to \infty} P(|\hat\theta_n - \theta| > \epsilon) = 0$$

Consistency means: given enough data, the estimator will be arbitrarily close to the truth. It is a minimum requirement — an inconsistent estimator is useless regardless of sample size.

**Sufficient condition for consistency:**

$$\text{Bias}(\hat\theta_n) \to 0 \quad \text{and} \quad \text{Var}(\hat\theta_n) \to 0 \quad \text{as } n \to \infty$$

**Example:** $\bar{X}$ for $\mu$:
- $\text{Bias}(\bar{X}) = 0$ for all $n$
- $\text{Var}(\bar{X}) = \sigma^2/n \to 0$ as $n \to \infty$
- Therefore $\bar{X}$ is consistent (also follows directly from the Law of Large Numbers)

*Connection forward: The Law of Large Numbers (Topic 4) is essentially the consistency of the sample mean. The Central Limit Theorem tells us the rate at which this convergence occurs.*

---

### 3.4 Efficiency and the Cramér-Rao Lower Bound

Unbiasedness and consistency are necessary but not enough. Among all unbiased estimators, we want the one with the smallest variance — the most **efficient** one.

**Fisher Information**

The **Fisher information** $\mathcal{I}(\theta)$ measures how much information the data carries about the parameter $\theta$:

$$\mathcal{I}(\theta) = E\!\left[\left(\frac{\partial}{\partial\theta}\log f(X \mid \theta)\right)^2\right] = -E\!\left[\frac{\partial^2}{\partial\theta^2}\log f(X \mid \theta)\right]$$

The quantity $\frac{\partial}{\partial\theta}\log f(X \mid \theta)$ is called the **score function**. The Fisher information is its variance.

For a sample of $n$ i.i.d. observations, the total Fisher information is:

$$\mathcal{I}_n(\theta) = n \cdot \mathcal{I}(\theta)$$

**The Cramér-Rao Lower Bound (CRLB)**

**Theorem.** Under regularity conditions, for any unbiased estimator $\hat\theta$ of $\theta$:

$$\text{Var}(\hat\theta) \geq \frac{1}{\mathcal{I}_n(\theta)} = \frac{1}{n\mathcal{I}(\theta)}$$

This is the **Cramér-Rao Lower Bound**. No unbiased estimator can have smaller variance than this. An estimator that achieves the CRLB is called **efficient**.

*The MLE (Topic 5) achieves the CRLB asymptotically — this is one of the deepest results in the course.*

**Example:** $X \sim N(\mu, \sigma^2)$, $\sigma^2$ known, estimating $\mu$.

Score function: $\frac{\partial}{\partial\mu}\log f(x \mid \mu) = \frac{x - \mu}{\sigma^2}$

Fisher information: $\mathcal{I}(\mu) = E\!\left[\frac{(X-\mu)^2}{\sigma^4}\right] = \frac{\sigma^2}{\sigma^4} = \frac{1}{\sigma^2}$

CRLB: $\text{Var}(\hat\mu) \geq \frac{1}{n/\sigma^2} = \frac{\sigma^2}{n}$

And indeed $\text{Var}(\bar{X}) = \sigma^2/n$ — the sample mean achieves the CRLB. It is the UMVUE (Uniformly Minimum Variance Unbiased Estimator).

---

### 3.5 Sufficiency

**Definition.** A statistic $T(X_1, \ldots, X_n)$ is **sufficient** for $\theta$ if the conditional distribution of the data given $T$ does not depend on $\theta$:

$$f(x_1, \ldots, x_n \mid T(x_1, \ldots, x_n) = t,\ \theta) = f(x_1, \ldots, x_n \mid T = t)$$

**Intuition:** Once you know $T$, no additional information about $\theta$ can be extracted from the raw data. $T$ is a lossless compression of the data for the purpose of inference.

**The Fisher-Neyman Factorization Theorem** — the practical tool for finding sufficient statistics:

$T(\mathbf{X})$ is sufficient for $\theta$ if and only if the likelihood can be factored as:

$$f(x_1, \ldots, x_n \mid \theta) = g(T(\mathbf{x}),\ \theta) \cdot h(x_1, \ldots, x_n)$$

where $g$ depends on the data only through $T$, and $h$ does not depend on $\theta$.

**Example:** $X_1, \ldots, X_n \overset{iid}{\sim} \text{Poisson}(\lambda)$.

$$\prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} = \frac{\lambda^{\sum x_i} e^{-n\lambda}}{\prod x_i!}$$

Here $g = \lambda^{\sum x_i} e^{-n\lambda}$ depends on data only through $T = \sum x_i$, and $h = 1/\prod x_i!$ does not involve $\lambda$. So $T = \sum_{i=1}^n X_i$ is sufficient for $\lambda$.

*Note the connection to the exponential family (Topic 1): for any exponential family distribution, the sufficient statistic is $T(x)$ from the canonical form $h(x)\exp(\eta(\theta)^\top T(x) - A(\theta))$. The exponential family was designed to make sufficiency automatic.*

**Rao-Blackwell Theorem:** If $\hat\theta$ is unbiased and $T$ is sufficient, then:

$$\tilde\theta = E[\hat\theta \mid T]$$

is also unbiased and satisfies $\text{Var}(\tilde\theta) \leq \text{Var}(\hat\theta)$. Conditioning on a sufficient statistic never hurts and often improves an estimator.

---

### 3.6 Robustness

An estimator is **robust** if it performs well even when model assumptions are mildly violated (e.g., data has outliers or is slightly non-Normal).

The sample mean has zero breakdown point — a single extreme outlier can drag it arbitrarily far. The sample median has a 50% breakdown point — you can corrupt up to half the data before it breaks. The trimmed mean lies in between.

In big data contexts with messy, real-world data, robustness matters enormously.

---

## Topic 4: Asymptotic Properties of Estimators

### 4.1 Why Asymptotics?

For small samples, exact sampling distributions are available for specific models (Normal, Binomial). But real data is messy — distributions are unknown, sample sizes vary, and exact results are impossible. **Asymptotic theory** gives us approximate results that become exact as $n \to \infty$ and are accurate enough for large $n$.

This is not approximation as a last resort. It is a principled, mathematically rigorous framework.

---

### 4.2 Modes of Convergence

These are the different ways a sequence of random variables $X_n$ can approach a limit $X$. Understanding the hierarchy is essential.

**Convergence in Probability** ($X_n \overset{P}{\to} X$):

$$\forall \epsilon > 0: \quad \lim_{n\to\infty} P(|X_n - X| > \epsilon) = 0$$

The probability that $X_n$ deviates from $X$ by any fixed amount goes to zero. This is what consistency means.

**Almost Sure Convergence** ($X_n \overset{a.s.}{\to} X$):

$$P\!\left(\lim_{n\to\infty} X_n = X\right) = 1$$

Stronger than convergence in probability. The sequence converges with probability 1. This is what the Strong Law of Large Numbers gives.

**Convergence in Distribution** ($X_n \overset{d}{\to} X$):

$$\lim_{n\to\infty} F_{X_n}(x) = F_X(x) \quad \text{at all continuity points of } F_X$$

The CDFs converge. The random variables themselves need not converge — only their distributions. This is the weakest form and the one delivered by the Central Limit Theorem.

**Hierarchy:**

$$\text{Almost sure} \implies \text{In probability} \implies \text{In distribution}$$

The implications are one-directional. Convergence in distribution does not imply convergence in probability.

---

### 4.3 The Law of Large Numbers (LLN)

Two versions, both saying the sample mean converges to the population mean.

**Weak LLN (Convergence in Probability):**

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \overset{P}{\to} \mu \quad \text{as } n \to \infty$$

**Strong LLN (Almost Sure Convergence):**

$$\bar{X}_n \overset{a.s.}{\to} \mu \quad \text{as } n \to \infty$$

Conditions: $X_i$ i.i.d. with $E|X_i| < \infty$.

The LLN is the mathematical justification for the entire empirical approach to science: collect enough data, and averages stabilize to their true values.

*Connection to consistency (Topic 3): the LLN directly proves that $\bar{X}$ is a consistent estimator of $\mu$.*

---

### 4.4 The Central Limit Theorem (CLT)

The most important theorem in statistics.

**Theorem (Classical CLT).** Let $X_1, X_2, \ldots$ be i.i.d. with mean $\mu$ and finite variance $\sigma^2$. Then:

$$\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \overset{d}{\to} N(0, 1) \quad \text{as } n \to \infty$$

Equivalently:

$$\sqrt{n}(\bar{X}_n - \mu) \overset{d}{\to} N(0, \sigma^2)$$

Or: for large $n$:
$$\bar{X}_n \overset{\text{approx}}{\sim} N\!\left(\mu, \frac{\sigma^2}{n}\right)$$

**What makes this remarkable:** The $X_i$ need not be Normal. They can be Bernoulli, Poisson, Uniform, anything — as long as the mean and variance exist. The sum's distribution converges to Normal regardless.

**The Delta Method** — extending the CLT to functions of the sample mean:

If $\sqrt{n}(\bar{X}_n - \mu) \overset{d}{\to} N(0, \sigma^2)$ and $g$ is differentiable at $\mu$ with $g'(\mu) \neq 0$, then:

$$\sqrt{n}(g(\bar{X}_n) - g(\mu)) \overset{d}{\to} N(0,\ [g'(\mu)]^2 \sigma^2)$$

*Example:* $X_i \sim \text{Exponential}(\lambda)$. We know $\hat\lambda = 1/\bar{X}$. With $g(x) = 1/x$, $g'(x) = -1/x^2$:

$$\sqrt{n}(\hat\lambda - \lambda) \overset{d}{\to} N(0,\ \lambda^2 \cdot \lambda^2 \cdot 1/\lambda^2) = N(0, \lambda^2)$$

Wait, let me be precise: $\text{Var}(X) = 1/\lambda^2$, $g'(\mu) = -\lambda^2$, so $[g'(\mu)]^2 \sigma^2 = \lambda^4 / \lambda^2 = \lambda^2$. So $\hat\lambda \overset{approx}{\sim} N(\lambda, \lambda^2/n)$.

The Delta Method is the key technique for finding the asymptotic distribution of any smooth function of sample moments.

---

### 4.5 Asymptotic Normality and Efficiency

An estimator $\hat\theta_n$ is **asymptotically normal** if:

$$\sqrt{n}(\hat\theta_n - \theta) \overset{d}{\to} N(0, V(\theta))$$

where $V(\theta)$ is the **asymptotic variance**.

The **asymptotic efficiency** of $\hat\theta_n$ relative to the CRLB is:

$$e(\hat\theta_n) = \frac{1/\mathcal{I}(\theta)}{V(\theta)/n}$$

An estimator is **asymptotically efficient** if $e(\hat\theta_n) = 1$, i.e., it achieves the CRLB in the limit.

*Connection forward: the MLE (Topic 5) is asymptotically efficient under regularity conditions.*

---

## Topic 5: Maximum Likelihood Estimation

### 5.1 The Core Idea

Given data $x_1, \ldots, x_n$ and model $f(x \mid \theta)$, ask: *for which $\theta$ is this data most probable?*

This is the likelihood principle, and it produces the **Maximum Likelihood Estimator (MLE)**.

---

### 5.2 The Likelihood Function

**Definition.** The **likelihood function** is:

$$L(\theta \mid x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i \mid \theta)$$

(For i.i.d. data, the joint density/pmf is a product.)

Note carefully: $f(x_i \mid \theta)$ is a function of $x_i$ with $\theta$ fixed (it's a density). When we write $L(\theta \mid x_i)$, we hold the $x_i$ fixed and vary $\theta$. Same formula, completely different perspective.

**The log-likelihood** (almost always easier to work with):

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i \mid \theta)$$

Because $\log$ is monotone increasing, maximizing $L(\theta)$ and $\ell(\theta)$ give the same answer.

---

### 5.3 Computing the MLE

The MLE is:

$$\hat\theta_{MLE} = \arg\max_{\theta \in \Theta} \ell(\theta)$$

**Procedure:**

1. Write $\ell(\theta) = \sum_{i=1}^n \log f(x_i \mid \theta)$
2. Differentiate: $\frac{\partial\ell}{\partial\theta} = 0$ — this is the **score equation** (or likelihood equation)
3. Solve for $\theta$ — this gives candidate(s)
4. Check second-order condition: $\frac{\partial^2\ell}{\partial\theta^2} < 0$ to confirm maximum (not minimum or saddle point)

**Example: Normal distribution, both parameters unknown.**

Data $X_1, \ldots, X_n \overset{iid}{\sim} N(\mu, \sigma^2)$.

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2$$

Differentiating with respect to $\mu$:
$$\frac{\partial\ell}{\partial\mu} = \frac{1}{\sigma^2}\sum_{i=1}^n(x_i - \mu) = 0 \implies \hat\mu = \bar{x}$$

Differentiating with respect to $\sigma^2$:
$$\frac{\partial\ell}{\partial\sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n(x_i - \mu)^2 = 0 \implies \hat\sigma^2 = \frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2$$

Note: $\hat\sigma^2_{MLE} = \frac{1}{n}\sum(x_i - \bar{x})^2$ is **biased** (denominator $n$, not $n-1$). The MLE is not always unbiased.

**Example: Poisson distribution.**

$X_1, \ldots, X_n \overset{iid}{\sim} \text{Poisson}(\lambda)$.

$$\ell(\lambda) = \sum_{i=1}^n (x_i \log\lambda - \lambda - \log x_i!) = \bar{x}\cdot n\log\lambda - n\lambda + C$$

$$\frac{d\ell}{d\lambda} = \frac{n\bar{x}}{\lambda} - n = 0 \implies \hat\lambda = \bar{x}$$

The MLE of the Poisson rate is the sample mean. Intuitive: the mean of the Poisson is $\lambda$, so you estimate $\lambda$ by the sample mean.

---

### 5.4 Fisher Information (Revisited)

For a single observation:

$$\mathcal{I}(\theta) = -E\!\left[\frac{\partial^2}{\partial\theta^2}\log f(X \mid \theta)\right]$$

For the curvature interpretation: high Fisher information means the log-likelihood is sharply peaked around the truth — the data are very informative. Low Fisher information means the log-likelihood is flat — the data are not very discriminating.

**Fisher information for the Normal ($\sigma^2$ known):**
$$\mathcal{I}(\mu) = \frac{1}{\sigma^2}$$

**Fisher information for the Poisson:**
$$\mathcal{I}(\lambda) = \frac{1}{\lambda}$$

---

### 5.5 Asymptotic Properties of the MLE

This is the crown jewel. Under regularity conditions, as $n \to \infty$:

**1. Consistency:**
$$\hat\theta_{MLE} \overset{P}{\to} \theta_0$$

where $\theta_0$ is the true parameter.

**2. Asymptotic Normality:**
$$\sqrt{n}(\hat\theta_{MLE} - \theta_0) \overset{d}{\to} N\!\left(0,\ \frac{1}{\mathcal{I}(\theta_0)}\right)$$

Equivalently:
$$\hat\theta_{MLE} \overset{approx}{\sim} N\!\left(\theta_0,\ \frac{1}{n\mathcal{I}(\theta_0)}\right)$$

**3. Asymptotic Efficiency:**

The MLE achieves the Cramér-Rao Lower Bound asymptotically. No consistent, asymptotically Normal estimator can have smaller asymptotic variance.

**4. Invariance:**

If $\hat\theta$ is the MLE of $\theta$, then for any function $g$:
$$\widehat{g(\theta)} = g(\hat\theta)$$

*Example:* If $\hat\lambda = \bar{X}$ is the MLE of $\lambda$ in the Poisson model, then $\hat\sigma = 1/\hat\lambda = 1/\bar{X}$ is automatically the MLE of the mean $\sigma = 1/\lambda$. No additional optimization needed.

*Connection to Topic 3: The asymptotic normality result lets us construct large-sample confidence intervals and tests using normal critical values. Connection to Topic 6: The likelihood ratio test statistic (Wilks' theorem) is derived directly from comparing log-likelihoods.*

---

### 5.6 Profile Likelihood for Nuisance Parameters

When $\theta = (\psi, \lambda)$ where $\psi$ is of interest and $\lambda$ is a nuisance parameter:

$$\ell_P(\psi) = \max_\lambda \ell(\psi, \lambda)$$

This **profile likelihood** eliminates $\lambda$ by substituting its MLE conditional on $\psi$. It can then be treated like a one-dimensional log-likelihood for $\psi$.

---

## Topic 6: Statistical Hypothesis Testing

### 6.1 The Structure of a Hypothesis Test

Testing is the second major form of inference (after estimation). Where estimation asks "what is $\theta$?", testing asks "is $\theta$ consistent with some claimed value?"

**The setup:**

- **Null hypothesis** $H_0$: the claim to be tested (usually a statement of no effect or a specific value)
- **Alternative hypothesis** $H_1$ (or $H_a$): what we accept if $H_0$ is rejected

**Types of alternatives:**

| Test Type | $H_1$ | Decision |
|---|---|---|
| Two-sided | $\theta \neq \theta_0$ | Reject if $\hat\theta$ is far from $\theta_0$ in either direction |
| Right-sided | $\theta > \theta_0$ | Reject if $\hat\theta$ is much larger than $\theta_0$ |
| Left-sided | $\theta < \theta_0$ | Reject if $\hat\theta$ is much smaller than $\theta_0$ |

**Test statistic:** A function of the data $T(X_1, \ldots, X_n)$ whose sampling distribution under $H_0$ is known.

**Rejection region (critical region):** The set of values of $T$ for which we reject $H_0$.

---

### 6.2 Type I and Type II Errors

|  | $H_0$ True | $H_0$ False |
|---|---|---|
| **Reject $H_0$** | Type I Error ($\alpha$) | Correct (Power) |
| **Fail to Reject $H_0$** | Correct | Type II Error ($\beta$) |

**Type I Error (False Positive):** Reject $H_0$ when it is true.

$$\alpha = P(\text{Reject } H_0 \mid H_0 \text{ true}) = P(T \in \mathcal{R} \mid H_0)$$

$\alpha$ is called the **significance level**. Convention: $\alpha = 0.05$ or $\alpha = 0.01$.

**Type II Error (False Negative):** Fail to reject $H_0$ when it is false.

$$\beta = P(\text{Fail to reject } H_0 \mid H_1 \text{ true})$$

**Power:** The probability of correctly rejecting a false $H_0$.

$$\text{Power} = 1 - \beta = P(\text{Reject } H_0 \mid H_1 \text{ true})$$

**The fundamental tradeoff:** For fixed $n$, decreasing $\alpha$ increases $\beta$ (increases Type II error, reduces power). To decrease both simultaneously, increase $n$.

**Power function:** $\pi(\theta) = P(\text{Reject } H_0 \mid \theta)$ for all $\theta$.
- Under $H_0$: $\pi(\theta_0) = \alpha$ (should be small)
- Under $H_1$: $\pi(\theta_1) = \text{Power}$ (should be large)

---

### 6.3 The p-value

**Definition.** The **p-value** is the probability of observing a test statistic as extreme as, or more extreme than, the observed value, *assuming $H_0$ is true*:

$$p = P(T \geq t_{obs} \mid H_0) \quad \text{(right-tailed)}$$

$$p = 2 \cdot P(T \geq |t_{obs}| \mid H_0) \quad \text{(two-tailed, symmetric distribution)}$$

Decision rule: reject $H_0$ if $p < \alpha$.

**Critical interpretation:** The p-value is NOT the probability that $H_0$ is true. It is not the probability that results occurred by chance. It is purely a measure of how consistent the data are with $H_0$. Small $p$ = data are inconsistent with $H_0$.

---

### 6.4 The Neyman-Pearson Lemma

For simple hypotheses ($H_0: \theta = \theta_0$ vs $H_1: \theta = \theta_1$), this lemma identifies the most powerful test.

**Theorem (Neyman-Pearson).** The **most powerful** level-$\alpha$ test for $H_0: \theta = \theta_0$ vs $H_1: \theta = \theta_1$ rejects $H_0$ when:

$$\Lambda = \frac{L(\theta_1 \mid \mathbf{x})}{L(\theta_0 \mid \mathbf{x})} > k$$

where $k$ is chosen so that $P(\Lambda > k \mid H_0) = \alpha$.

This is the **likelihood ratio statistic**. The NP Lemma says: among all tests at level $\alpha$, the likelihood ratio test maximizes power against the specific alternative $\theta_1$.

**Intuition:** If the data look much more likely under $H_1$ than under $H_0$ (i.e., $\Lambda$ is large), we reject $H_0$ in favor of $H_1$.

---

### 6.5 The Generalized Likelihood Ratio Test

For composite hypotheses (parameter sets rather than single values):

$$H_0: \theta \in \Theta_0 \quad \text{vs} \quad H_1: \theta \in \Theta \setminus \Theta_0$$

**The likelihood ratio statistic:**

$$\Lambda(\mathbf{x}) = \frac{\sup_{\theta \in \Theta_0} L(\theta \mid \mathbf{x})}{\sup_{\theta \in \Theta} L(\theta \mid \mathbf{x})} = \frac{L(\hat\theta_0)}{L(\hat\theta_{MLE})}$$

where $\hat\theta_0$ is the constrained MLE (maximizing over $\Theta_0$) and $\hat\theta_{MLE}$ is the unconstrained MLE.

Note: $0 < \Lambda \leq 1$. Small $\Lambda$ means the null model fits much worse than the unconstrained model — evidence against $H_0$.

**Wilks' Theorem** — the fundamental asymptotic result:

$$-2\log\Lambda(\mathbf{x}) \overset{d}{\to} \chi^2_r \quad \text{as } n \to \infty \text{ under } H_0$$

where $r = \dim(\Theta) - \dim(\Theta_0)$ is the number of restrictions imposed by $H_0$.

This means: reject $H_0$ if $-2\log\Lambda > \chi^2_{r,\alpha}$ (the $\alpha$ upper quantile of $\chi^2_r$).

*Connection: Wilks' theorem is why the $\chi^2$ distribution appears constantly in statistical tests — it is the limiting distribution of log-likelihood ratio statistics. The Chi-square from Topic 1 (sum of squared normals) connects to the Chi-square from hypothesis testing through this theorem.*

---

### 6.6 Common Tests and Their Statistics

**One-sample Z-test** ($\sigma^2$ known, $H_0: \mu = \mu_0$):

$$Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} \sim N(0,1) \text{ under } H_0$$

**One-sample t-test** ($\sigma^2$ unknown, $H_0: \mu = \mu_0$):

$$T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t_{n-1} \text{ under } H_0$$

**Two-sample t-test** (comparing means, $H_0: \mu_1 = \mu_2$):

$$T = \frac{\bar{X}_1 - \bar{X}_2}{S_p\sqrt{1/n_1 + 1/n_2}} \sim t_{n_1+n_2-2} \text{ under } H_0$$

where $S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$ is the pooled variance.

**Chi-square test for variance** ($H_0: \sigma^2 = \sigma_0^2$):

$$\chi^2 = \frac{(n-1)S^2}{\sigma_0^2} \sim \chi^2_{n-1} \text{ under } H_0$$

---

### 6.7 Multiple Testing

When conducting $m$ simultaneous tests each at level $\alpha$, the probability of at least one false rejection (the **family-wise error rate**) is:

$$\text{FWER} = 1 - (1-\alpha)^m$$

For $m = 100$ tests at $\alpha = 0.05$: $\text{FWER} = 1 - 0.95^{100} \approx 0.994$. Nearly certain to make a false rejection.

**Bonferroni correction:** Test each hypothesis at level $\alpha/m$ to control FWER at $\alpha$.

**Benjamini-Hochberg procedure:** Controls the **False Discovery Rate (FDR)** = expected proportion of false rejections among all rejections — less conservative, preferred in genomics, big data.

---

## Topic 7: Likelihood Inference and Applications

### 7.1 Likelihood-Based Confidence Intervals

**Wald Interval** — based on asymptotic normality of MLE:

$$\hat\theta \pm z_{\alpha/2} \cdot \widehat{\text{SE}}(\hat\theta) = \hat\theta \pm z_{\alpha/2} / \sqrt{n\hat{\mathcal{I}}(\hat\theta)}$$

**Profile Likelihood Interval** — based on inverting the likelihood ratio test:

$$\left\{\theta : -2[\ell(\theta) - \ell(\hat\theta)] \leq \chi^2_{1,\alpha}\right\}$$

This is the set of $\theta$ values not rejected at level $\alpha$ by the likelihood ratio test. It is generally preferred over the Wald interval because it is invariant to reparameterization and can be asymmetric (which is more honest when the distribution of $\hat\theta$ is skewed).

**Score (Rao) Interval** — based on the score function at $\theta_0$.

---

### 7.2 Likelihood Inference in Linear Regression

**The model:** For $i = 1, \ldots, n$:

$$Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad \varepsilon_i \overset{iid}{\sim} N(0, \sigma^2)$$

This is a parametric model with $\theta = (\beta_0, \beta_1, \sigma^2)$.

**The likelihood:**

$$L(\beta_0, \beta_1, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y_i - \beta_0 - \beta_1 x_i)^2}{2\sigma^2}\right)$$

**Log-likelihood:**

$$\ell(\beta_0, \beta_1, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \beta_0 - \beta_1 x_i)^2$$

Maximizing over $(\beta_0, \beta_1)$ for fixed $\sigma^2$: maximizing $\ell$ is equivalent to **minimizing** $\sum(y_i - \beta_0 - \beta_1 x_i)^2$.

*This is the connection between MLE and Ordinary Least Squares (OLS): under Normal errors, MLE = OLS. They are the same thing.*

**MLE solutions:**

$$\hat\beta_1 = \frac{\sum_{i=1}^n(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n(x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}$$

$$\hat\beta_0 = \bar{y} - \hat\beta_1 \bar{x}$$

$$\hat\sigma^2 = \frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2 = \frac{\text{RSS}}{n}$$

**Hypothesis test for slope** ($H_0: \beta_1 = 0$, no linear relationship):

$$T = \frac{\hat\beta_1}{\text{SE}(\hat\beta_1)} \sim t_{n-2} \text{ under } H_0$$

where $\text{SE}(\hat\beta_1) = \hat\sigma / \sqrt{S_{xx}}$.

**Confidence interval for $\beta_1$:**

$$\hat\beta_1 \pm t_{n-2, \alpha/2} \cdot \text{SE}(\hat\beta_1)$$

---

### 7.3 Likelihood Inference for Contingency Tables

A **contingency table** cross-tabulates two categorical variables. The simplest case: a $2 \times 2$ table.

|  | Category B=1 | Category B=0 | Total |
|---|---|---|---|
| **A=1** | $n_{11}$ | $n_{10}$ | $n_{1\cdot}$ |
| **A=0** | $n_{01}$ | $n_{00}$ | $n_{0\cdot}$ |
| **Total** | $n_{\cdot 1}$ | $n_{\cdot 0}$ | $n$ |

**Test of independence** ($H_0$: A and B are independent):

Under independence, expected counts are:

$$E_{ij} = \frac{n_{i\cdot} \cdot n_{\cdot j}}{n}$$

**Pearson Chi-square statistic:**

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \overset{d}{\to} \chi^2_{(r-1)(c-1)}$$

where $r$ and $c$ are the number of rows and columns.

**Likelihood ratio (G-test) statistic:**

$$G^2 = 2\sum_{i,j} O_{ij}\log\!\left(\frac{O_{ij}}{E_{ij}}\right) \overset{d}{\to} \chi^2_{(r-1)(c-1)}$$

Both statistics are asymptotically equivalent and converge to the same $\chi^2$ distribution under $H_0$. The G-test is the likelihood ratio test from Topic 6 applied to categorical data. This is Wilks' theorem in action.

---

## Topic 8: Large Sample Theory

### 8.1 Finite Sample vs Asymptotic Inference

**Finite-sample inference** gives exact results for specific models (usually Normal-based). Exact $t$-tests, exact F-tests, exact chi-square tests for variance — these are all finite-sample.

**Asymptotic inference** gives approximate results valid for large $n$. These are broader: they apply to any smooth model, not just Normal.

In big data ($n$ large), asymptotic approximations are often excellent. But "large" is context-dependent — for some problems $n=30$ suffices; for others, $n=1000$ is still small.

---

### 8.2 Asymptotic Relative Efficiency (ARE)

When comparing two consistent, asymptotically Normal estimators $\hat\theta_1$ and $\hat\theta_2$ with asymptotic variances $V_1$ and $V_2$:

$$\text{ARE}(\hat\theta_1, \hat\theta_2) = \frac{V_2}{V_1}$$

ARE $> 1$: $\hat\theta_1$ is more efficient. ARE $< 1$: $\hat\theta_2$ is more efficient.

**Example:** For a Normal population:
- $\text{ARE}(\text{mean}, \text{median}) = \pi/2 \approx 1.57$

The sample mean is about 57% more efficient than the sample median for Normal data. But for heavy-tailed distributions (Cauchy), the median is more efficient. This is where robustness (Topic 3) and efficiency make contact: efficiency is model-specific.

---

### 8.3 Slutsky's Theorem

Essential for combining asymptotic results:

If $X_n \overset{d}{\to} X$ and $Y_n \overset{P}{\to} c$ (a constant), then:

$$X_n + Y_n \overset{d}{\to} X + c$$
$$X_n Y_n \overset{d}{\to} cX$$
$$X_n / Y_n \overset{d}{\to} X/c \quad (c \neq 0)$$

*Example application:* In a $t$-test, $Z_n = \sqrt{n}(\bar{X} - \mu)/\sigma \overset{d}{\to} N(0,1)$ and $S/\sigma \overset{P}{\to} 1$, so $T_n = Z_n \cdot (\sigma/S) \overset{d}{\to} N(0,1)$ by Slutsky. This justifies using $z$-critical values instead of $t$-critical values for large $n$.

---

### 8.4 Continuous Mapping Theorem

If $X_n \overset{d}{\to} X$ and $g$ is continuous, then $g(X_n) \overset{d}{\to} g(X)$.

*Example:* If $Z_n \overset{d}{\to} N(0,1)$, then $Z_n^2 \overset{d}{\to} \chi^2_1$.

This is why the $\chi^2$ distribution for sums of squared normals (Topic 1 special case of Gamma) connects to test statistics in hypothesis testing.

---

### 8.5 M-Estimators

**M-estimators** generalize MLEs. An M-estimator $\hat\theta$ solves:

$$\sum_{i=1}^n \psi(X_i, \theta) = 0$$

for some function $\psi$. For MLE: $\psi(x, \theta) = \frac{\partial}{\partial\theta}\log f(x \mid \theta)$ (the score function).

M-estimators are asymptotically normal under general conditions:

$$\sqrt{n}(\hat\theta - \theta_0) \overset{d}{\to} N\!\left(0,\ \frac{E[\psi^2]}{[E[\partial\psi/\partial\theta]]^2}\right)$$

This unifies regression, quantile estimation, and robust estimation under one framework.

---

## Topic 9: Nonparametric Statistical Methods

### 9.1 When Parametric Assumptions Break Down

Everything so far has assumed a specific distributional form (Normal, Poisson, etc.). In practice:
- The true distribution may be unknown
- Data may be severely skewed or heavy-tailed
- Sample sizes may be small, making CLT approximations inaccurate
- We may want results that hold regardless of the distribution

**Nonparametric methods** make minimal distributional assumptions. They sacrifice some efficiency (when parametric assumptions are correct) in exchange for robustness (when they are not).

---

### 9.2 The Empirical Distribution Function (EDF)

The nonparametric estimator of the true CDF $F$ is the **empirical distribution function**:

$$\hat{F}_n(x) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(X_i \leq x)$$

At any point $x$, $\hat{F}_n(x)$ is the proportion of observations $\leq x$.

**Glivenko-Cantelli Theorem:** $\sup_x |\hat{F}_n(x) - F(x)| \overset{a.s.}{\to} 0$ — the EDF converges uniformly to the true CDF.

**Donsker's Theorem:** $\sqrt{n}(\hat{F}_n - F)$ converges to a Brownian bridge process — the functional CLT.

---

### 9.3 The Sign Test

**Setting:** $H_0: M = M_0$ where $M$ is the population median. No distributional assumption.

Under $H_0$, exactly half of observations should exceed $M_0$. Define:

$$B = \#\{i : X_i > M_0\}$$

Under $H_0$: $B \sim \text{Binomial}(n, 0.5)$.

The sign test is exact — the p-value uses the exact Binomial distribution, not an approximation.

---

### 9.4 The Wilcoxon Signed-Rank Test

A more powerful version of the sign test that uses the magnitude of deviations, not just their sign.

**Procedure:**
1. Compute $D_i = X_i - M_0$, discard zeros
2. Rank $|D_i|$ from smallest (rank 1) to largest
3. Compute $W^+ = \sum_{D_i > 0} R_i$ where $R_i$ is the rank of $|D_i|$

Under $H_0$ (and assuming symmetric distribution): $E[W^+] = n(n+1)/4$, $\text{Var}(W^+) = n(n+1)(2n+1)/24$.

For large $n$: $\frac{W^+ - E[W^+]}{\sqrt{\text{Var}(W^+)}} \overset{d}{\to} N(0,1)$.

**ARE vs $t$-test:**
- Normal data: $\text{ARE}(\text{Wilcoxon}, t) = 3/\pi \approx 0.955$ — only 4.5% less efficient
- Logistic data: ARE $= \pi^2/9 \approx 1.097$ — more efficient than $t$
- Heavy-tailed data: can be much larger — far more efficient than $t$

The Wilcoxon test loses very little when the $t$-test assumptions hold, but gains enormously when they do not.

---

### 9.5 The Wilcoxon Rank-Sum Test (Mann-Whitney)

Two independent samples, $H_0$: same distribution (same median/location).

Let $X_1, \ldots, X_m$ and $Y_1, \ldots, Y_n$ be the two samples. Pool and rank all $m+n$ observations. $W$ = sum of ranks from the first sample.

Under $H_0$: $E[W] = m(m+n+1)/2$, $\text{Var}[W] = mn(m+n+1)/12$.

Large-sample approximation: $Z = \frac{W - E[W]}{\sqrt{\text{Var}[W]}} \overset{d}{\to} N(0,1)$.

---

### 9.6 Exact Confidence Intervals from Rank Tests

By **test-inversion**: a $100(1-\alpha)\%$ confidence interval for a parameter consists of all values $\theta_0$ not rejected by the corresponding hypothesis test at level $\alpha$.

Inverting the sign test gives a confidence interval for the median based on order statistics — exact, no distributional assumption.

Inverting the Wilcoxon signed-rank test gives the **Hodges-Lehmann estimator** of location:

$$\hat{M} = \text{median}\!\left\{\frac{X_i + X_j}{2} : i \leq j\right\}$$

The associated confidence interval is based on these pairwise averages (Walsh averages).

---

### 9.7 The Kolmogorov-Smirnov Test

**Goodness-of-fit test:** $H_0: F = F_0$ (data come from specified distribution $F_0$).

**Test statistic:**

$$D_n = \sup_x |\hat{F}_n(x) - F_0(x)|$$

Under $H_0$ and for continuous $F_0$: $P(\sqrt{n} D_n > t) \to 2\sum_{k=1}^\infty (-1)^{k+1} e^{-2k^2 t^2}$ — the Kolmogorov distribution.

The KS test is **distribution-free**: the null distribution of $D_n$ does not depend on $F_0$ (when $F_0$ is fully specified and continuous). This is the exact coverage property.

---

### 9.8 The Kruskal-Wallis Test

Nonparametric analog of one-way ANOVA. $k$ independent groups, $H_0$: all groups have the same location.

$$H = \frac{12}{N(N+1)}\sum_{i=1}^k \frac{R_i^2}{n_i} - 3(N+1)$$

where $N = \sum n_i$, $R_i$ = sum of ranks in group $i$.

Under $H_0$: $H \overset{d}{\to} \chi^2_{k-1}$ for large samples.

---

## Topic 10: Specialized Topics and Computational Applications

### 10.1 Dichotomous Data: One-Sample and Two-Sample Location Problems

**One-sample proportion test.** $X \sim \text{Binomial}(n, p)$, $H_0: p = p_0$.

**Exact test (small samples):** Use exact Binomial probabilities.

$$p\text{-value} = P(X \geq x_{obs} \mid p = p_0) = \sum_{k=x_{obs}}^n \binom{n}{k} p_0^k (1-p_0)^{n-k}$$

**Large-sample Z-test:**

$$Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}} \sim N(0,1) \text{ under } H_0$$

**Two-sample proportion test.** $H_0: p_1 = p_2$.

$$Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}}$$

where $\hat{p} = (n_1\hat{p}_1 + n_2\hat{p}_2)/(n_1 + n_2)$ is the pooled proportion.

**Odds ratio and logistic regression.** The log-odds $\log\frac{p}{1-p}$ is the natural scale for Bernoulli data, which connects directly to the natural parameter $\eta(p) = \log\frac{p}{1-p}$ from the exponential family (Topic 1). Logistic regression models the log-odds as a linear function of covariates.

---

### 10.2 Point Estimation Revisited: UMVUE

A **Uniformly Minimum Variance Unbiased Estimator (UMVUE)** achieves the minimum variance among all unbiased estimators, for every $\theta \in \Theta$.

**The Lehmann-Scheffé Theorem:** If $T$ is a **complete sufficient statistic** and $g(T)$ is unbiased for $\theta$, then $g(T)$ is the UMVUE.

A statistic $T$ is **complete** if:

$$E_\theta[h(T)] = 0 \text{ for all } \theta \implies P_\theta(h(T) = 0) = 1 \text{ for all } \theta$$

For exponential family models, the natural sufficient statistic is complete. This means MLE-based estimators in exponential families are automatically UMVUEs.

The complete roadmap: Exponential family $\Rightarrow$ Sufficient statistic $T$ $\Rightarrow$ Complete $\Rightarrow$ Function of $T$ that is unbiased $\Rightarrow$ UMVUE.

---

### 10.3 Density Estimation

When you want to estimate the density $f$ without specifying a parametric family — a fully nonparametric approach.

**Histogram (naive density estimator):**

Partition the support into bins of width $h$. Estimate density as:

$$\hat{f}(x) = \frac{\text{count in bin containing } x}{n \cdot h}$$

**Kernel Density Estimator (KDE):**

$$\hat{f}(x) = \frac{1}{nh}\sum_{i=1}^n K\!\left(\frac{x - X_i}{h}\right)$$

where $K$ is a **kernel function** — a symmetric density (e.g., Gaussian, Epanechnikov).

**Gaussian kernel:** $K(u) = \frac{1}{\sqrt{2\pi}}e^{-u^2/2}$

**Epanechnikov kernel** (optimal in MSE sense): $K(u) = \frac{3}{4}(1 - u^2)\mathbf{1}(|u| \leq 1)$

**Bandwidth selection** — the key tuning parameter:

$$h^* = \left(\frac{4\hat\sigma^5}{3n}\right)^{1/5}$$

This is **Silverman's rule of thumb** (assumes underlying density is Normal). Too small $h$: wiggly, high variance. Too large $h$: smooth, high bias. Optimal $h$ balances the two.

**Bias-variance tradeoff for KDE:**

$$\text{MSE}[\hat{f}(x)] \approx \frac{h^4}{4}\left[\mu_2(K) f''(x)\right]^2 + \frac{R(K)}{nh}$$

where $\mu_2(K) = \int u^2 K(u)\,du$ and $R(K) = \int K(u)^2\,du$.

The optimal bandwidth minimizes this: $h^* \propto n^{-1/5}$. Note the rate: density estimation converges at $n^{-2/5}$, slower than the parametric $n^{-1/2}$ rate. This is the **curse of dimensionality** — nonparametric estimation requires more data.

---

### 10.4 Confidence Sets

A **confidence set** $C(\mathbf{X})$ is a set (not necessarily an interval) such that:

$$P(\theta \in C(\mathbf{X})) \geq 1 - \alpha \quad \text{for all } \theta \in \Theta$$

For a scalar parameter: confidence interval. For a vector parameter: confidence ellipse (or region).

For the Normal model with unknown $(\mu, \sigma^2)$, the joint confidence region at level $\alpha$ is:

$$C(\bar{x}, s^2) = \left\{(\mu, \sigma^2) : \frac{(\bar{x} - \mu)^2}{s^2/n} \leq F_{1,n-1,\alpha},\ \frac{(n-1)s^2}{\sigma^2} \in \left[\chi^2_{n-1,1-\alpha/2},\ \chi^2_{n-1,\alpha/2}\right]\right\}$$

The likelihood ratio approach gives an elegant all-in-one confidence region:

$$C_{LR} = \left\{\theta : -2[\ell(\theta) - \ell(\hat\theta)] \leq \chi^2_{p,\alpha}\right\}$$

This is the set of $\theta$ values within the "$\chi^2_{p,\alpha}/2$ log-likelihood units" of the MLE.

---

### 10.5 Bayesian Inference: The Full Picture

While frequentist inference treats $\theta$ as fixed and unknown, **Bayesian inference** treats $\theta$ as a random variable with a **prior distribution** $\pi(\theta)$.

**Bayes' Theorem:**

$$\pi(\theta \mid \mathbf{x}) = \frac{L(\theta \mid \mathbf{x}) \cdot \pi(\theta)}{\int L(\theta \mid \mathbf{x}) \pi(\theta)\,d\theta} \propto L(\theta \mid \mathbf{x}) \cdot \pi(\theta)$$

The **posterior** $\pi(\theta \mid \mathbf{x})$ combines:
- The **likelihood** $L(\theta \mid \mathbf{x})$ — what the data say
- The **prior** $\pi(\theta)$ — what we believed before seeing data

Point estimates from the posterior:
- **Posterior mean:** $E[\theta \mid \mathbf{x}]$ — minimizes posterior expected squared error
- **Posterior median:** minimizes posterior expected absolute error
- **MAP (Maximum a posteriori):** $\arg\max_\theta \pi(\theta \mid \mathbf{x})$ — closest to MLE when prior is flat

**Bayesian interval:** A **credible interval** $[a, b]$ satisfies $P(\theta \in [a,b] \mid \mathbf{x}) = 1 - \alpha$. This is the interpretation people mistakenly assign to frequentist confidence intervals.

**Conjugate priors** (from the exponential family, Topic 1):

| Likelihood | Prior | Posterior |
|---|---|---|
| Binomial($n$, $p$) | Beta($\alpha$, $\beta$) | Beta($\alpha + x$, $\beta + n - x$) |
| Poisson($\lambda$) | Gamma($\alpha$, $\beta$) | Gamma($\alpha + \sum x_i$, $\beta/(n\beta+1)$) |
| Normal($\mu$, $\sigma^2$) | Normal($\mu_0$, $\tau^2$) | Normal (formula below) |

For Normal with known $\sigma^2$:

$$\mu \mid \mathbf{x} \sim N\!\left(\frac{\mu_0/\tau^2 + n\bar{x}/\sigma^2}{1/\tau^2 + n/\sigma^2},\ \frac{1}{1/\tau^2 + n/\sigma^2}\right)$$

The posterior mean is a **weighted average** of the prior mean and the sample mean, with weights inversely proportional to their variances. As $n \to \infty$, the data overwhelm the prior — Bayesian and frequentist results agree in large samples.

---

### 10.6 Bootstrap Resampling

The **bootstrap** is a computational method for estimating the sampling distribution of any estimator without analytical derivation.

**Algorithm (Nonparametric Bootstrap):**

1. From data $\mathbf{x} = (x_1, \ldots, x_n)$, draw $B$ bootstrap samples $\mathbf{x}^{*(1)}, \ldots, \mathbf{x}^{*(B)}$, each of size $n$ **with replacement**
2. Compute $\hat\theta^{*(b)} = g(\mathbf{x}^{*(b)})$ for each $b$
3. The distribution of $\{\hat\theta^{*(b)}\}_{b=1}^B$ approximates the sampling distribution of $\hat\theta$

**Bootstrap standard error:** $\widehat{\text{SE}}_B = \text{sd}(\hat\theta^{*(1)}, \ldots, \hat\theta^{*(B)})$

**Bootstrap confidence interval (percentile method):** The $[\alpha/2, 1-\alpha/2]$ quantiles of $\{\hat\theta^{*(b)}\}$.

**Why it works:** The empirical distribution $\hat{F}_n$ is a consistent estimator of $F$ (Glivenko-Cantelli). Sampling from $\hat{F}_n$ is sampling from the best available approximation to $F$. The sampling distribution of $\hat\theta^*$ around $\hat\theta$ approximates the sampling distribution of $\hat\theta$ around $\theta_0$.

The bootstrap is especially powerful for: complex statistics (ratios, correlations, eigenvalues), small samples, and cases where analytical standard errors are unavailable.

---

## Quick Reference: Formula Sheet

### Distributions

| Distribution | PMF/PDF | Mean | Variance |
|---|---|---|---|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| Binomial($n,p$) | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Poisson($\lambda$) | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ |
| Normal($\mu,\sigma^2$) | $(2\pi\sigma^2)^{-1/2}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ |
| Gamma($\alpha,\beta$) | $\frac{x^{\alpha-1}e^{-x/\beta}}{\Gamma(\alpha)\beta^\alpha}$ | $\alpha\beta$ | $\alpha\beta^2$ |
| $\chi^2_\nu$ | Gamma($\nu/2, 2$) | $\nu$ | $2\nu$ |

### Estimation

| Formula | Description |
|---|---|
| $\text{Bias}(\hat\theta) = E[\hat\theta] - \theta$ | Bias of an estimator |
| $\text{MSE}(\hat\theta) = \text{Var}(\hat\theta) + \text{Bias}^2$ | Mean squared error |
| $\text{Var}(\hat\theta) \geq 1/(n\mathcal{I}(\theta))$ | Cramér-Rao Lower Bound |
| $\mathcal{I}(\theta) = -E[\partial^2\ell/\partial\theta^2]$ | Fisher information |
| $\bar{X} \sim N(\mu, \sigma^2/n)$ | Sampling distribution of mean (Normal data) |
| $S^2 = \frac{1}{n-1}\sum(X_i-\bar{X})^2$ | Unbiased sample variance |
| $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ | Distribution of sample variance |

### Asymptotic Results

| Formula | Description |
|---|---|
| $\bar{X}_n \overset{P}{\to} \mu$ | Weak Law of Large Numbers |
| $\bar{X}_n \overset{a.s.}{\to} \mu$ | Strong Law of Large Numbers |
| $\sqrt{n}(\bar{X}-\mu)/\sigma \overset{d}{\to} N(0,1)$ | Central Limit Theorem |
| $\sqrt{n}(g(\bar{X})-g(\mu)) \overset{d}{\to} N(0,[g'(\mu)]^2\sigma^2)$ | Delta Method |
| $\hat\theta_{MLE} \overset{approx}{\sim} N(\theta, 1/(n\mathcal{I}(\theta)))$ | Asymptotic normality of MLE |

### Hypothesis Testing

| Formula | Description |
|---|---|
| $Z = (\bar{X}-\mu_0)/(\sigma/\sqrt{n})$ | Z-test statistic |
| $T = (\bar{X}-\mu_0)/(S/\sqrt{n}) \sim t_{n-1}$ | t-test statistic |
| $\chi^2 = (n-1)S^2/\sigma_0^2 \sim \chi^2_{n-1}$ | Chi-square test for variance |
| $\Lambda = L(\hat\theta_0)/L(\hat\theta_{MLE})$ | Likelihood ratio statistic |
| $-2\log\Lambda \overset{d}{\to} \chi^2_r$ | Wilks' theorem |
| $\chi^2 = \sum (O-E)^2/E$ | Pearson chi-square |

### Nonparametric

| Statistic | Description |
|---|---|
| $\hat{F}_n(x) = \frac{1}{n}\sum\mathbf{1}(X_i \leq x)$ | Empirical CDF |
| $B = \#\{X_i > M_0\} \sim \text{Bin}(n, 0.5)$ | Sign test statistic |
| $W^+ = \sum_{D_i>0} R_i$ | Wilcoxon signed-rank |
| $D_n = \sup_x |\hat{F}_n - F_0|$ | KS test statistic |
| $\hat{f}(x) = \frac{1}{nh}\sum K\!\left(\frac{x-X_i}{h}\right)$ | Kernel density estimator |

---

## How the Topics Connect: The Bigger Picture

This course is not a collection of isolated techniques. It is a coherent argument, built one brick at a time.

**The story:**

You begin with a **parametric model** (Topic 1) — a mathematical description of your data-generating mechanism. The model has unknown parameters, and the data belong to the exponential family, which guarantees sufficient statistics.

To learn about those parameters, you **estimate** them (Topic 2). The estimator is a function of the data — itself a random variable with a sampling distribution. Method of moments gives you a starting point.

You evaluate your estimators using **bias, consistency, efficiency, and sufficiency** (Topic 3). The Cramér-Rao bound tells you how good an estimator can theoretically be. The Fisher information — introduced here — will return in Topic 5.

When exact sampling distributions are unavailable (which is most of the time), you rely on **asymptotic theory** (Topic 4): the LLN assures consistency, the CLT assures approximate normality, and the Delta Method extends both to functions of estimators.

The **MLE** (Topic 5) brings everything together: it uses the full parametric model to derive the optimal estimator. Under regularity, it is consistent, asymptotically normal with variance equal to the CRLB, and invariant. The log-likelihood is the central object — it reappears immediately in testing.

**Hypothesis testing** (Topic 6) uses the sampling distribution of test statistics (from Topic 4) and likelihood ratios (from Topic 5). The Neyman-Pearson lemma, Wilks' theorem, and all classical tests emerge from this framework.

**Likelihood inference and applications** (Topic 7) show the same likelihood machinery applied to regression and categorical data. Linear regression under Normal errors is MLE. Chi-square tests for contingency tables are likelihood ratio tests.

**Large sample theory** (Topic 8) ties up the asymptotic loose ends: ARE, Slutsky's theorem, and the machinery for comparing estimators and tests in the large-$n$ regime.

When parametric assumptions cannot be trusted, **nonparametric methods** (Topic 9) step in. They sacrifice efficiency but gain robustness. The EDF replaces the parametric family. Rank tests replace $t$-tests. KDE replaces parametric density specification.

Finally, **specialized topics** (Topic 10) connect everything to computation: Bootstrap resampling approximates sampling distributions without derivation. Bayesian inference reframes the entire enterprise — parameters have distributions, likelihood and prior combine to give a posterior.

**The through-line:**

```
Data → Parametric Model → Likelihood → MLE → Asymptotic Normality
                                  ↓                      ↓
                           Hypothesis Tests        Confidence Intervals
                                  ↓
                          (When assumptions fail)
                                  ↓
                       Nonparametric Methods
```

Every formula in this guide is a step along that path. Learn them not as isolated objects but as language — the formal language in which uncertainty speaks.

---

*DSA 8301 — Statistical Inference for Big Data*
*Strathmore University, Institute of Mathematical Sciences, 2025/2026*
*Prof. Jacob Ong'ala | jongala@strathmore.edu*

*Document compiled for: MSc Data Science and Analytics, Year 1, Semester III*