# Statistical Mechanics

## [Boltzmann Law: Physics to Computing](https://www.edx.org/learn/engineering/purdue-university-boltzmann-law-physics-to-computing)

After completing this course, you will be able to:

- Explain the _law of equilibrium_ and _entropy_.
- Analyze and design simple _Boltzmann machines_.
- Analyze and design simple _quantum circuits_.

### 1. The Boltzmann Law

Upon completion of this week, you will be able to:

- differentiate between _Boltzmann law_ and Boltzmann _approximation_
- describe _entropy_, _free energy_
- explain the _self-consistent field method_

#### 1.1. The State Space

In a material like a molecule or a solid, electrons can occupy different **energy levels** $\varepsilon_i, i \in \{1, \dots, n\}$ according to the **exclusion principle** that each level can accomodate at most one electron so that the lower levels are filled first.

The dividing line between energy levels is called the **electrochemical potential** $\mu$.

At higher temperatures $\mu$ becomes more diffuse. The **Fermi function** describes the probability that a given energy level $\varepsilon$ is occupied by an electron at a given temperature $T$, with the Boltzmann constant $k$ and electrochemical potential $\mu$:

$$
f(\varepsilon) = \frac{1}{1 + e^{\frac{\varepsilon - \mu}{kT}}},
\qquad
f(x)  = \frac{1}{1 + e^x}, \, x = \frac{\varepsilon - \mu}{kT}
$$

The Fermi function $f$ is equal to $\frac{1}{2}$ when $\varepsilon = \mu$.

Each energy level represents a one-electron space that can either be occupied or unoccupied. The **state space** of the system is the set of all possible configurations of the system, of which there are $2^n$ for $n$ one-electron energy levels.

With $8$ one-electron energy levels, the number of levels in state space is $2^8 = 256$.

The **Boltzmann law** defines _equilibrium in statistical mechanics_: it states that the probability of a system being in a particular state is proportional to the exponential of the negative of the difference of that state's energy and number of electrons (with partition function $Z = \sum_i e^{- \frac{E_i - \mu N_i}{kT}}$ which serves as a normalisation constant so the probabilities of all possible states sum to $1$, thus "partitioning" the probability space to be able to calculate individual probabilities):

$$p_i = \frac{1}{Z} e^{-\frac{E_i - \mu N_i}{kT}}$$

_Derivation of the Fermi function_ (for one level): there are two states, $0$ and $1$, for each position. The $0$ state has $0$ electrons and energy $0$, the $1$ state has $1$ electron and energy $\varepsilon_1$. By the law of total probability, $p_0 + p_1 = \frac{1}{Z} e^{-\frac{0}{kT}} + \frac{1}{Z} e^{-\frac{\epsilon_1 - \mu}{kT}} = \frac{1}{Z} + \frac{1}{Z} e^{-x_1} = \frac{1 + e^{-x_1}}{Z} = 1$. It follows that $Z = 1 + e^{-x_1}$. Thus, $p_1 = \frac{e^{-x_1}}{1 + e^{-x_1}} = \frac{1}{1 + e^{x_1}} = f(\varepsilon), \, p_0 = \frac{1}{1 + e^{-x_1}} = \frac{1 + e^{-x_1} - e^{-x_1}}{1 + e^{-x_1}} = 1 - \frac{e^{-x_1}}{1 + e^{-x_1}} = 1 - f(\varepsilon)$.

_Two one-electron energy levels_ without interactions: $2^2$ states, $Z = 1 + e^{-x_1} + e^{-x_2} + e^{-x_1}e^{-x_2} = (1 + e^{-x_1}) (1 + e^{-x_2})$

| $i$ | $N$ | $E$                             | $\frac{E - \mu N}{kT}$                                         | $p_i$                                    |
| --- | --- | ------------------------------- | -------------------------------------------------------------- | ---------------------------------------- |
| 11  | 2   | $\varepsilon_1 + \varepsilon_2$ | $\frac{\varepsilon_1 + \varepsilon_2 - 2 \mu}{kT} = x_1 + x_2$ | $\frac{1}{Z} e^{-x_1}e^{-x_2} = f_1 f_2$ |
| 10  | 1   | $\varepsilon_2$                 | $x_2$                                                          | $\frac{1}{Z} e^{-x_2} = (1 - f_1) f_2$   |
| 01  | 1   | $\varepsilon_1$                 | $x_1$                                                          | $\frac{1}{Z} e^{-x_1} = (1 - f_2) f_1$   |
| 00  | 0   | $0$                             | $0$                                                            | $\frac{1}{Z} = (1 - f_1)(1 - f_2)$       |

Note: The Fermi function can be generalised to multiple energy levels in this way iff the levels are **non-interacting**; systems with interactions between electrons can only be solved with the general Boltzmann law as the interaction term prevents factorisation into separate Fermi functions: $x_1 + x_2 + \frac{U_0}{kT}$.

The **Boltzmann approximation** gives a valid approximation _to the Fermi function_ for $x_1 \gg 1$ (i.e. $\varepsilon_1 \gg \mu$): $f(\varepsilon_1) \approx e^{-x_1}$.

#### 1.2. The Boltzmann Law

The Boltzmann law applies to systems that exchange either only energy, or only particles, or both with a reservoir. It does not apply to systems completely isolated from the reservoir.

Every state $i$ is a function of the number of electrons $N_i$ and amount of energy $E_i$. In order to change states, a non-isolated system continuously exchanges particles $\Delta N$ and energy $\Delta E$ with its surroundings, called the **reservoir**, which is a function of $\mu, T$.

- A **grand canonical ensemble** is an ensemble of systems which exchanges both energy and particles: $p_i = \frac{1}{Z} e^{-\frac{E_i - \mu N_i}{kT}}$.
- A **canonical ensemble** is an ensemble of systems which exchange only energy but no particles: $p_i = \frac{1}{Z} e^\frac{\mu N_i}{kT} e^{-\frac{E_i}{kT}} = \frac{1}{Z'} e^{-\frac{E_i}{kT}}$ with $Z' = \frac{Z}{e^\frac{\mu N_i}{kT}}$ with $N_i$ also a constant.

The Boltzmann law states that in equilibrium, _states with lower energy have higher probability of being occupied_. The system and reservoir have the same energy $E_0$ and the system can exchange energy $\varepsilon$ with the reservoir. The probability of the system having energy $E_0 + \varepsilon$ is $p_1 = \frac{1}{Z} e^{-\frac{E_0 + \varepsilon - \mu N}{kT}}$ and the probability of the system having energy $E_0$ is $p_2 = \frac{1}{Z} e^{-\frac{E_0 - \mu N}{kT}}$. The ratio of the probabilities shows that the probability of the system being in a higher energy state compared to a lower energy state decreases exponentially with the increase in energy $\varepsilon$:

$$\frac{p_1}{p_2} = \frac{W_1}{W_2} = \frac{W(E_0 + \varepsilon)}{W(E_0)} = \frac{e^{-\frac{E_0 + \varepsilon - \mu N}{kT}}}{e^{-\frac{E_0 - \mu N}{kT}}} = e^{-\frac{\varepsilon}{kT}}$$

- _When the system has lower energy, it is more likely to be in that state_; which is a reflection of the second law of thermodynamics where systems evolve towards states of higher entropy (or disorder) while maintaining energy conservation.
- _If the system has lower energy, the reservoir has higher energy_; the system and reservoir are in equilibrium when the system has the same energy as the reservoir.
- The weight of the states $W$ represents the _number of accessible states of the reservoir_ and _is a rapidly increasing function of energy_: $W_1 = W(E_0 + \varepsilon) > W_2 = W(E_0)$.

Entropy $S$ is proportional to the logarithm of the number of states available with energy $E$:

$$S(E) = k \ln{W(E)}$$

The temperature $T$ is the inverse of the rate of change of entropy with respect to energy:

$$\frac{1}{T} = \frac{dS}{dE}, \qquad \frac{dS}{dE}\bigg\vert_{E=E_0} \approx \frac{S(E_0 + \varepsilon) - S(E_0)}{\varepsilon}$$

The ratio of the probabilities of the system being in a higher energy state compared to a lower energy state can thus be shown to be equal to the ratios of the probabilities $p_i = \frac{1}{Z} e^{- \frac{E_i}{kT}}$ as predicted by the Boltzmann law for a canonical ensemble:

$$\frac{p_1}{p_2} = \frac{W_1}{W_2} = \frac{W(E_0 + \varepsilon)}{W(E_0)} = \frac{e^{\frac{S(E_0 + \varepsilon)}{k}}}{e^{\frac{S(E_0)}{k}}} = e^{\frac{S(E_0 + \varepsilon) - S(E_0)}{k}} = e^{\frac{\varepsilon}{kT}} = e^{\frac{E_2 - E_1}{kT}} = \frac{e^{\frac{-E_1}{kT}}}{e^{\frac{-E_2}{kT}}}$$

For a grand canonical ensemble, $p_i = \frac{1}{Z} e^{-\frac{E_i - \mu N_i}{kT}}$, where also particles are exchanged, $W$ is a function of both energy and the number of particles $N$:

$$\frac{p_1}{p_2} = \frac{W(E_0 + \varepsilon, N_0 + n)}{W(E_0, N_0)} = e^\frac{\varepsilon - \mu n}{kT}$$

The Boltzmann law applies to all systems irrespective of its details (such as interactions, superconduction) because it _reflects a property of the reservoir_, $W$, (not of the system, $p$):

$$S(E, N) = k \ln{W(E, N)}$$

$$
\frac{\partial S}{\partial E} \bigg\vert_N = \frac{1}{T},
\quad
\frac{\partial S}{\partial N} \bigg\vert_E = - \frac{\mu}{T}
$$

#### 1.3. Shannon Entropy

Consider a reservoir composed of two-electron levels with $n$ identical non-interacting units (where the tilde is used to distinguish from the system's equilibrium values $p_0, p_1$):

$$n \times \tilde{p}_0, \, n \times \tilde{p}_1, \quad \tilde{p}_0 + \tilde{p}_1 = 1, \quad N = n \tilde{p}_1, \, E = \varepsilon n \tilde{p}_1, \quad S = k \ln{W(\tilde{p}_1)}$$

Suppose $\tilde{p}_1 = 1$, then there is only one possible state $1111111\dots$; suppose $\tilde{p}_1 = 0$, then there is only one possible state $00000000\dots$; suppose $\tilde{p}_1 = 0.5$, then there are many possible states $0101010\dots, 1010101\dots, \dots$:

$$W = {}^n \mathrm{C}_{n\tilde{p}_1} = {}^n \mathrm{C}_{n\tilde{p}_0} = \frac{n!}{(n\tilde{p}_0)! (n\tilde{p}_1)!}$$

Consider a collection of $8$ one-electron levels which are all full, i.e. there is one possible state. The entropy is zero: $S = k \ln{1} = 0k = 0$.

That is, at $\tilde{p}_1 = 0$ and $\tilde{p}_1 = 1$ entropy $S = 0$, and it reaches its maximum value, $\ln{2}$ at $\tilde{p}_1 = 0.5$. For $n$ units, this curve is described by the following expression (which can be derived from the above using Stirling's approximation) in the limit as $n \rarr \infty$:

$$\frac{S}{n} \leq -k \ln{\max{W}} = -k (\tilde{p}_0 \ln{\tilde{p}_0} + \tilde{p}_1 \ln{\tilde{p}_1})$$

Note: in information theory, entropy is expressed in terms of the information content of a message with $n$ bits, which depends on the number of possible messages $\{100111\dots\}$: $H = - n (\tilde{p}_0 \log_2{\tilde{p}_0} + \tilde{p}_1 \log_2{\tilde{p}_1})$. It is a dimensionless number, while in thermodynamics entropy has a dimension.

The generalisation for $n$ units to multiple energy levels at one point in time follows:

$$
\frac{S}{n} = - k \sum_{i=0}^{d-1}{\tilde{p}_i \ln{\tilde{p}_i}}, \quad
\frac{E}{n} = \sum_{i=0}^{d-1}{\tilde{p}_i E_i}, \quad
\frac{N}{n} = \sum_{i=0}^{d-1}{\tilde{p}_i N_i}
$$

Consider a collection of $n$ units each with $3$ levels that are all equally probable. The entropy is $S = n ( -3k \frac{1}{3} \ln{\frac{1}{3}} ) = nk \ln{3}$.

Consider a collection of $8$ one-electron levels each with a probability of $0.5$ for being full (and for being empty, i.e. two states with the same probability). The entropy is $S = 8 ( -2k \frac{1}{2} \ln{\frac{1}{2}} ) = 8k \ln{2}$.

Averaging over time gives the entropy, energy and number of particles per unit (as a time or ensemble average):

$$
S = - k \sum_{i=0}^{d-1}{\tilde{p}_i \ln{\tilde{p}_i}}, \quad
E = \sum_{i=0}^{d-1}{\tilde{p}_i E_i}, \quad
N = \sum_{i=0}^{d-1}{\tilde{p}_i N_i}
$$

At equilibrium, i.e. when $\tilde{p}_i = p_i$: $dS = \frac{dE}{T} - \frac{\mu dN}{T}$. This can be shown with the concept of free energy.

#### 1.4. Free Energy

$$F = E - \mu N - TS = - kT \ln{Z} + kT \sum_i{\tilde{p}_i \big( \ln{\frac{\tilde{p}_i}{p_i}} \big)}$$

_Free energy is minimal at equilibrium_:

- At equilibrium, i.e. when $\tilde{p}_i = p_i$, $\sum_i{\tilde{p}_i \big( \ln{\frac{\tilde{p}_i}{p_i}} \big)} = 0$, $F = - kT \ln{Z}$m and $dF = 0$.
- Out of equilibrium, **Gibb's inequality** applies: $\sum_i{\tilde{p}_i \big( \ln{\frac{\tilde{p}_i}{p_i}} \big)} > 0$.

In statistics, this is referred to as the **KL divergence** which indicates _how far a distribution is from another_.

At equilibrium, small changes in the probabilities yield small changes in the entropy, energy and number of particles, but overall the free energy cannot change: $dF = dE - \mu dN - T \, dS = 0$; that is, $dS = \frac{dE}{T} - \frac{\mu dN}{T}$.

_Entropy drives flow_:

$$dF \leq 0 \qquad dF_\text{Reservoir} = 0$$

$$dS \geq \frac{dE}{T} - \frac{\mu dN}{T} \qquad dS_\text{Res} = \frac{dE_\text{Res}}{T_\text{Res}} - \frac{\mu dN_\text{Res}}{T_\text{Res}}$$

Note that $dE_\text{Res} = -dE, dN_\text{Res} = -dN$ since energy is always conserved in a closed system (i.e. the system and reservoir overall).

Overall _entropy increases_, and _heat flows from hot to cold_:

$$dS + dS_\text{Reservoir} = d(E - \mu N) \big( \frac{1}{T} - \frac{1}{T_{Res}} \big) \geq 0$$

- If the system is hot and the reservoir is cold then $\frac{1}{T} - \frac{1}{T_{Res}} < 0$, and it must be that $d(E - \mu N) < 0$ also.
  - If energy is exchanged but no electrons, then $dN = 0, dE < 0$; that is, energy flows from the hot system to the cold reservoir.
  - If electrons are exchanged, i.e. $dN \neq 0$, then $(\varepsilon - \mu) dN < 0$ and the flow depends on the energy level $\varepsilon$ (**thermoelectric current**):
    - If $\varepsilon > \mu$ then $dN < 0$ and the flow is from the hot system to the cold reservoir.
    - If $\varepsilon < \mu$ then $dN > 0$ and the flow is from the cold reservoir to the hot system.
- If the system is cold and the reservoir is hot then $\frac{1}{T} - \frac{1}{T_{Res}} > 0$, and it must be that $d(E - \mu N) > 0$ also.

Consider contacts 1 and 2 held at two different temperatures $T_1 > T_2$. If an energy $\Delta E$ is transferred from 1 to 2, the overall increase in entropy is $\Delta E ( \frac{1}{T_2} - \frac{1}{T_1} )$ with $\Delta E > 0$ so that the total change in entropy is greater than or equal to zero.

Consider an out-of-equilibrium system in contact with a reservoir at equilibrium. The system exchanges energy and particles with the reservoir as it comes to equilibrium. In this process

- the free energy of the system goes down, and
- the overall entropy of the system and reservoir goes up.

Consider contacts 1 and 2 held at the same temperature $T$, but at two different electrochemical potentials $\mu_1 > \mu_2$. If a number of electrons $\Delta N$ is transferred from 1 to 2, the overall increase in entropy is $\Delta N \big( \frac{\mu_1 - \mu_2}{T} \big)$.

#### 1.5. Self-Consistent Field

The Boltzmann law applies to the state space, which for $n$ one-electron levels contains $2^n$ states. This becomes impractical for calculations as $n$ becomes larger. The Fermi function is an alternative for non-interacting electrons, but not when there are interactions.

The Self-Consistent Field method makes use of the Fermi function and provides an approximation when there are interactions by solving the following two equations (where $\sigma$ is the sigmoid function):

$$f_r = \sigma(- \tilde{x}_r), \quad \tilde{x}_r = x_r + \sum_{q \neq r}{\frac{U_{rq}}{kT} f_q}$$

$$\text{where} \quad x_r = \frac{\varepsilon_r - \mu}{kT}, \quad \tilde{\varepsilon}_r = \varepsilon_r + \sum_{q \neq r}{U f_q}$$

$$\text{Photon energy} \, hv = E(n_r = 1) - E(n_r = 0)$$

Consider four one-electron energy levels each of energy $\varepsilon$ with an interaction for every pair of electrons. The total energy of the state $\{0111\}$ is given by the sum of the energies of the three occupied levels and the energies of their pairs (i.e. combinations): $3 \varepsilon + 3 U$.

### 2. Boltzmann Machines

Upon completion of this week, you will be able to:

- describe the time sampling method
- analyze state-space response of Boltzmann machines
- explain learning in Boltzmann machines

#### 2.1. Sampling

The relation between the Fermi function $f_r = \sigma(- \tilde{x}_r)$ and the electron number $n_r$, is that $P(n_r = 1) = f_r$ and $P(n_r = 0) = 1 - f_r$.

If you randomly sample a number between $0$ and $1$, $R_{0, 1}$, then define $\vartheta$ as a step function such that:

$$
n_r = \vartheta(f_r - R_{0, 1}) = \begin{cases}
  1 & \text{if } f_r - R_{0, 1} > 0 \\
  0 & \text{if } f_r - R_{0, 1} < 0
\end{cases}
$$

Thus, $P(n_r = 1) = P(f_r - R_{0, 1} > 0) = P(f_r > R_{0, 1}) = f_r$, e.g. if $f = 0.5$ then $n_r$ will be zero and one half of the time.

Combining the **time sampling and consistent field methods** yields a series of time samples of states of neurons or energy levels (following the Boltzmann distribution), a neural network consisting of two components:

- Binary stochastic neuron: $n_r = \vartheta(f_r - R_{0, 1}), \, f_r = \sigma(- \tilde{x}_r)$.
- Synapse: $\tilde{x}_r = x_r + \sum_q{w_{rq} n_q}$.

Note that the synapse function arises from interaction energy, $\tilde{x}_r = \frac{E - \mu N}{kT} = \frac{E\vert_{n_r=1} - E\vert_{n_r=0}}{kT} = \frac{\partial \tilde{E}}{\partial n_r} = x_r + \sum_q{w_{rq} n_q}$, which holds only for linear functions. Because $n_r$ is a binary variable $\in \{0, 1\}$, the functions in this system are always linear because $(n_r)^k = n_r \, \forall \, k$.

It follows from the definition of the derivative $\tilde{x}_r = \frac{\partial \tilde{E}}{\partial n_r}$ that a quadratic interaction $n_r n_q$ yields a linear synapse $n_q$, and a higher-order interaction, e.g. cubic $n_r n_q n_s$ yields a quadratic synapse $n_q n_s$.

If the energy of a system of three levels is given by $\tilde{E} = x_3 n_3 + n_1 n_2 n_3$, the synaptic output driving level 3 is given by $\tilde{x}_3 = x_3 + n_1 n_2$.

Time sampling reproduces the Boltzmann law _if updating is_ **sequential**, not if simultaneous.

The advantage of time sampling (and the consistent field method) is that it allows working in $n$ dimensions and still being able to analyse how the interactions in the $n$-dimensional space (the neurons) control or affect the response in the $2^n$ state space (the probability distribution, as defined by the Boltzmann law, over the number of possible configurations of the neurons).

#### 2.2. Orchestrating Interactions

In general, with a bias $x$ and symmetric weights $w_{rq} = w_{qr}$ zero on the diagonals $w_{ii} = 0 \, \forall \, i$, the energies can be written as:

$$
\begin{align*}
  \tilde{E} & = \big( \sum_r{n_r x_r} + \frac{1}{2} \sum_{r, q}{n_r w_{rq} n_q} \big) \times E_0 \\
  & = \big( \bm{n}^T \bm{x} + \frac{1}{2} \bm{n}^T \bm{w} \bm{n} \big) \times E_0
\end{align*}
$$

$\bm{x}, \bm{w} \in \mathbb{R}^n$ (dictated by physics) determine the $2^n$ state space response.

Consider biases and weights $\bm{x} = \begin{bmatrix}
  -1 \\ -1 \\ -1 \\ -1
\end{bmatrix}, \, \bm{w} = \begin{bmatrix}
  0 & 2 & 0 & 0 \\
  2 & 0 & 2 & 0 \\
  0 & 2 & 0 & 2 \\
  0 & 0 & 2 & 0 \\
\end{bmatrix}$. Then the states with the highest probability are given by $\{0101\}, \{1010\}, \{1001\}$. The synaptic output driving level 4 is given by $\tilde{x}_4 = -1 + 2 n_3$.

Going forward, the objective is to find the bias and weights $\bm{x}, \bm{w}$ that give a state space response dictated by / consisten with a given application, with $s$ a generic _binary variable_ ($0, 1$; on, off, empty, full; up, down; etc.; as opposed the number of electrons $n$).

#### 2.3. Optimisation

The energy of a system of $n$ classical spins (i.e. $s$ is a binary variable with two values $0$ and $1$) is given by

$$\tilde{E} = \big( \bm{s}^T \bm{x} + \frac{1}{2} \bm{s}^T \bm{w} \bm{s} \big) \times E_0$$

**Min-cut / max-cut problem**: how to divide into two equal groups so that they are as _weakly_ / _strongly_ connected as possible.

Define $C_{qr}$ as the connection matrix, $F_{qr} = \begin{cases}
  1 & \text{if } s_q \neq s_r \\
  0 & \text{if } s_q = s_r
\end{cases}$. Then

$$
\begin{align*}
  \tilde{E} & = \sum_{r, q}{C_{rq} R_{rq}} \\
  & = \sum_{r, q}{C_{rq} (s_r - s_q)^2} \\
  & = \sum_{r, q}{C_{rq} (s_r^2 + s_q^2 - 2 s_r s_q)} \\
  & = \sum_{r, q}{C_{rq} (s_r + s_q - 2 s_r s_q)} \quad \text{since } 0^2 = 0, 1^2 = 1 \\
  & = \sum_r{\Big( s_r \sum_q{C_{rq}} \Big)} + \sum_q{\Big( s_q \sum_r{C_{rq}} \Big)} - 2 \sum_{r, q}{C_{rq} s_r s_q} \\
  & = \sum_r{\Big( s_r \sum_q{C_{rq}} \Big)} + \sum_r{\Big( s_r \sum_q{C_{qr}} \Big)} - 2 \sum_{r, q}{C_{rq} s_r s_q} \\
  & = \sum_r{\Big( s_r \sum_q{( C_{rq} + C_{qr} )} \Big)} - 2 \sum_{r, q}{s_r C_{rq} s_q}
\end{align*}
$$

$$x_r = \sum_q{( C_{rq} + C_{qr} )}, \quad w_{rq} = -4 C_{rq}$$

Consider a graph with a connection matrix given by $C = \begin{bmatrix}
  0 & 1 & 0 & 0 \\
  1 & 0 & 0 & 0 \\
  0 & 0 & 0 & 1 \\
  0 & 0 & 1 & 0 \\
\end{bmatrix}$. Suppose we choose $w_{rq} = -4 C_{rq}, x_r = 2$. The highest probability peaks are $\{0000\}, \{0011\}, \{1100\}, \{1111\}$.

All zeros, and all ones, is also a solution. Therefore, an additional constraint has to be enforced that the **solution should contain equal number of zeros and ones** by adding the following term to the energy:

$$
\begin{align*}
  \Big( \sum_r{s_r - \frac{n}{2}} \Big)^2
  & = \frac{n^2}{4} - n \sum_r{s_r} + \sum_r{s_r} \sum_q{s_q} \\
  & = \frac{n^2}{4} - n \sum_r{s_r} + \sum_{r, q}{s_r s_q} \\
  & = \frac{n^2}{4} - n \sum_r{s_r} + \sum_{r}{s_r^2} + \sum_{r \neq q}{s_r s_q} \\
  & = \frac{n^2}{4} - n \sum_r{s_r} + \sum_{r}{s_r} + \sum_{r \neq q}{s_r s_q} \quad \text{since } 0^2 = 0, 1^2 = 1 \\
  & = \frac{n^2}{4} - (n - 1) \sum_{r}{s_r} + \sum_{r \neq q}{s_r s_q}
\end{align*}
$$

Note the constant $\frac{n^2}{4}$ makes no difference.

- Min-cut: $x_r = -K(n-1) + \sum_q{( C_{rq} + C_{qr} )}, \, w_{rq, \, r \neq q} = 2K - 4 C_{rq}$.
- Max-cut: $x_r = -K(n-1) - \sum_q{( C_{rq} + C_{qr} )}, \, w_{rq, \, r \neq q} = 2K + 4 C_{rq}$.

Depending on the value of $K$, the unwanted peaks will be partially or completely suppressed.

#### 2.4. Inference

The energy of a system of classical spins in bipolar notation ($m$ is a bipolar variable with values $-1$ and $+1$) is given by

$$\tilde{E} = \bm{m}^T \bm{h} + \frac{1}{2} \bm{m}^T J \bm{m}$$

In binary notation ($s$ is a bipolar variable with values $0$ and $1$),

$$\tilde{E} = \bm{s}^T \bm{x} + \frac{1}{2} \bm{s}^T \bm{w} \bm{s}$$

Suppose $\bm{h} = \begin{bmatrix}
  0 \\ 0
\end{bmatrix}, J = \begin{bmatrix}
  0 & +1 \\
  +1 & 0
\end{bmatrix}$. The following choice of $\bm{x}, \bm{w}$ should give the same values of energy for all configurations: $\bm{x} = \begin{bmatrix}
  -2 \\ -2
\end{bmatrix}, \bm{w} = \begin{bmatrix}
  0 & +4 \\
  +4 & 0
\end{bmatrix}$.

#### 2.5. Learning

### 3. The Transition Matrix

#### 3.1. Markov Chain Monte Carlo (MCMC)

#### 3.2. Gibbs Sampling

#### 3.3. Sequential vs. Simultaneous

#### 3.4. Bayesian Networks

#### 3.5. Feynman Paths

### 4. The Quantum Boltzmann Law

#### 4.1. Quantum Spins

#### 4.2. One Q-Bit systems

#### 4.3. Spin-Spin Interactions

#### 4.4. Two Q-Bit systems

#### 4.5. Quantum Annealing

### 5. The Quantum Transition Matrix

#### 5.1. Adiabatic Gated Computing

#### 5.2. Hadamard Gates

#### 5.3. Grover Search

#### 5.4. Shor's Algorithm

#### 5.5. Feynman Paths

## [Statistical Mechanics: Algorithms and Computations](https://www.coursera.org/learn/statistical-mechanics)
