# Reinforcement Learning

- **Model-Free**:
  - _Value-Based_:
    - Tabular Methods:
      - On-Policy: TD(0) [p. 119], SARSA [p. 129], MC Control (first-visit) [p. 92]
      - Off-Policy: Q-Learning [p. 131]
    - Semi-Gradient Methods:
      - On-Policy: TD(0) [p. 119], SARSA[p. 129]
      - Off-Policy: Q-Learning [p. 131]
    - Deep Methods:
      - On-Policy: DQN, Double DQN
  - _Policy-Based_:
    - Deep / Value Function Approximations:
      - On-Policy: REINFORCE [\*, p. 326], REINFORCE with Baseline [\*, p. 329]
  - _Value and Policy-Based_:
    - Actor-Critic:
      - Semi-Gradient Methods:
        - On-Policy: Actor-Critic with semi-gradient TD(0) [p. 332]
      - Deep Methods:
        - On-Policy: A3C (or A2C) [\*], TRPO [\*], PPO [\*]
        - Off-Policy: DDPG [\*], SAC [\*]
- **Model-Based**:
  - _Access to Model_:
    - Dynamic Programming: Exhaustive Search [p. 62], Policy Iteration [p. 80], Value Iteration [p. 82]
    - Planning: MCTS [p. 185]
    - Deep Methods: Alpha Zero
  - _Learn Model_:
    - Dyna: Dyna-Q [p. 161], Dyna-AC
    - Deep Methods: World Models, Dreamer

\* = Continuous Action Space is possible

- [1. Introduction to Planning and Reinforcement Learning](#1-introduction-to-planning-and-reinforcement-learning)
  - [1.1 Scope and Limitations](#11-scope-and-limitations)
- [2. Multi-Armed Bandits (MAB)](#2-multi-armed-bandits-mab)
  - [2.1. Exploration vs. Exploitation](#21-exploration-vs-exploitation)
  - [2.2. Rewards and regret](#22-rewards-and-regret)
  - [2.3. Greedy algorithms](#23-greedy-algorithms)
  - [2.4. Upper-Confidence Bounds (UCB)](#24-upper-confidence-bounds-ucb)

## 1. Introduction to Planning and Reinforcement Learning

Reinforcement learning: _Agent_ learning to achieve a goal, solve sequential decision problems, via repeated _interaction_ with the (dynamic) _environment_.

RL and planning:

- RL problem: efficiently learn a high-value policy by interacting with an MDP.
- Planning problem: given an MDP (we know all of its components, $(\mathcal{S}, \mathcal{A}, p)$), compute the optimal policy $\pi$.

**Reward hypothesis**: All goals can be described by the maximisation of the expected value of cumulative scalar rewards.

- Manage investment portfolio: reward?
- Make humanoid robot walk: reward?

TODO: read <https://www.sciencedirect.com/science/article/pii/S0004370221000862?via%3Dihub>

Elements of a RL System:

- Reward signal (given)
- Policy (learned)
- Value function (learned)
- Model (of the environment; learned)

### 1.1. Scope and Limitations

Machine Learning = improving performance with experience (data).

- In supervised learning, the training signal is _instructive_, i.e. "which is the correct action, independently of the one taken?"
- In RL, the training signal is **evaluative**, i.e. "how good is the action taken?", and can only be learned by trial and error.

Beyond Tic-Tac-Toe:

- RL works perfectly also with multiple agents or when there is no opponent at all (vs. nature): as long as you can encode the goal in a reward, there is no need for an opponent.
- RL works also for problems with a large number of states: Tic-Tac-Toe has $3^9 \approx 20k$ states, Go $3^{361} >$ the number of atoms in the universe.
- RL can also be used when we can't foresee the effect of our actions.
- RL works with any amount of prior knowledge we have on the problem.

Key challenges:

- _Unknown environment_: How do actions affect environment state and rewards?
- _Exloration-exploitation dilemma_.
- _Delayed rewards_: Which prior action(s)' long-term consequences led to the reward?

## 2. Multi-Armed Bandits (MAB)

Simplest RL problem: the **Multi-Armed Bandit Problem**:

- Given: a set of $k$ actions, $\mathcal{A}$, the agent can take, rewards from the environment distributed according to a **stationary** _probability distribution_, $p(r \mid a)$ number of rounds $T$.
- Repeat for $t$ in $T$ rounds:
  1. Algorithm selects arm $A_t \in \mathcal{A}$.
  2. Algorithm observes reward $R_t \in [0, 1]$.
- Goal: maximise expected long-term total reward.
  - Value of arm: expected reward given action taken $q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a] \, \big(= \sum_r{p(r \mid a) r} \big)$.
  - If we knew the action values with certainty, we would select the action with the maximal value. The true value $q_*$ is unknown to the agent, so compute _estimates_ $Q_t(a)$.

An action-value method requires two tasks:

1. How to _estimate the values_ of actions, i.e. how to compute $Q_t(a)$.
2. How to use the estimates to make an _action selection decision_, i.e. given $Q_t(a)$ how to select $A_t$.

Action-value _estimation_ methods:

- Sample average.
- Exponential recency-weighted average method.

Action _selection_ methods:

- Pure greedy.
- $\epsilon$-greedy.
- UCB.

**1. Sample-Average Estimation Method**: Estimate the value of each arm:

$$Q_t(a) \doteq \frac{\text{sum of the rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t} = \frac{\sum_{i=1}^{t-1}{R_i \mathbb{1}_{A_i = a}}}{\sum_{i=1}^{t-1}{\mathbb{1}_{A_i = a}}}$$

Same equation, different notation:

$$Q_t(a) = \frac{1}{N_t(a)} \sum_{\tau = 1}^{t-1}{R_\tau \cdot \mathbb{1}_{A_t = a}}$$

_Sample averages converge in the limit_: $\lim_{N_t(a) \rarr \infty}{Q_t(a)} = q_*(a)$.

**2. (Greedy) Action Selection Method**: Select the best action $A_t$ given action-value estimates $Q_t(a)$:

- Exploit: Greedy action selection $A_t = A_t^* \doteq \argmax_a{Q_t(a)}$.
- Explore: Random action selection $A_t \sim \text{Unif}(\mathcal{A})$.

**Incremental Learning Rule** - more computationally efficient: Sample average $Q_n = \frac{R_1 + R_2 + \dots + R_{n-1}}{n-1}$ (focusing on a single action) can be computed incrementally to avoid recomputing: $Q_{n+1} = Q_n + \frac{1}{n} [R_n - Q_n]$. (This is the standard form for update rules in RL: NewEstimate <- OldEstimate + StepSize[Target - OldEstimate].)

**Non-Stationary Problems**: Suppose the true _action values shift over time_ (like in the real world). Sample average alone is no longer appropriate (TODO: why?). We should weight more recent rewards higher!

Solution: **Exponential recency-weighted average method**: instead of using the average (from the sample average), track action values using a _constant step-size parameter_ $\alpha \in (0, 1]$: $Q_{n+1} \doteq Q_n + \alpha [R_n - Q_n]$ and unrolling $Q_{n+1} = \alpha R_n + (1 - \alpha) \alpha R_{n-1} + (1 - \alpha)^2 \alpha R_{n-2} + \dots$. Recent rewards are exponentially more important!

**Stochastic Approximation Convergence Conditions**: Estimates $Q_n$ will converge with probability $1$ to $q_*$ if $\sum_{n=1}^\infty{\alpha_n(a)} \rarr \infty$ and $\sum_{n=1}^\infty{\alpha_n^2(a) < \infty}$.

- Conditions hold: $\alpha_n = \frac{1}{n}$.
- Conditions don't hold: $\alpha_n = c, \alpha_n = \frac{1}{n^2}$.

### 2.1. Exploration vs. Exploitation

- Exploit: Pick best option so far.
- Explore: Learn more about other options.

**Exploration-Exploitation Dilemma**:

- Exploiting, i.e. selecting the maximal action value, is the right thing to do to maximise the expected reward at the current time step.
- Exploring is the right thing to do to maximise the expected reward _in the long run_ - but requires _time_.

Pursuing only one = failure. The more time you have, the more you can afford to explore - but there is no right answer.

### 2.2. Rewards and Regret

#### 2.2.1. The problem of the sparse reward

#### 2.2.2. Introduction to advanced exploration techniques: curiosity and empowerment in RL

#### 2.2.3. Introduction to curriculum learning to facilitate the learning of the goal

#### 2.2.4. Hierarchical RL to learn complex tasks

#### 2.2.5. The learning of Universal Value Functions and Hindsight Experience Replay (HER)

### 2.3. Greedy Algorithms

$\epsilon$**-Greedy Action Selection Algorithm** (greedy: $\epsilon = 0$; optimism = set $Q_1$ high, realism = $Q_1 = 0$):

- $Q_1(a), N_1(a) = 0, \, \forall a \in \mathcal{A}$
- For each round $t$ in $T$:
  - $A_t = \begin{cases}A_t^* & \text{Pr } 1 - \epsilon \\ \text{Unif}(\mathcal{A}) & \text{otherwise} \end{cases}$
  - Execute $A_t$, observe $R_t$.
  - Update $N_t(a), Q_t(a)$.

### 2.4. Upper Confidence Bounds (UCB)

Greedy action-selection selects actions that look best now but does not explore actions that could be better in the long run. $\epsilon$-greedy action selection tries also non-greedy actions but it explores _indiscriminately_.

**UCB Action-Selection Algorithm**: Takes into account both optimality, i.e. greedy, and uncertainty, i.e. not explored enough yet, of action values.

- $Q_1(a), N_1(a) = 0, \, \forall a \in \mathcal{A}$
- For each round $t$ in $T$:
  - $A_t = \begin{cases}\text{Unif}(\mathcal{A}) & \max_a{N_t(a) = 0} \\ \argmax_a{\bigg[Q_t(a) + c \sqrt{\frac{\ln{t}}{N_t(a)}}\bigg]} & \text{otherwise} \end{cases}$
  - Execute $A_t$, observe $R_t$.
  - Update $N_t(a), Q_t(a)$.

Note:

- The uncertainty $\sqrt{\frac{\ln{t}}{N_t(a)}}$ is
  - reduced each time we select $a$: $N_t(a) = N_{t-1}(a) + 1$.
  - increased each time another action is selected: same $N_t(a)$ but $\ln{t}$ increases.
- Estimate + (positive) uncertainty $\approx$ upper bound.
- $c$ is the confidence on this upper bound; $c = 0$ is equivalent to being greedy.

_When the uncertainty of an action $a$'s value is high, UCB considers $a$ to potentially have a higher true value than its current estimate_.

TODO: read UCB paper: P. Auer, N. Cesa-Bianchi, P. Fischer (2002). Finite-time analysis of the multi-armed bandit problem. Machine Learning, 47(2-3), 235-256.

## 3. Markov Decision Processes (MDPs)

_Bringing State Back_: The **Agent-Environment Interface**: Agent and environment interact at discrete time steps $t = 0, 1, 2, 3, \dots$:

- The best action for the agent may depend on the state.
- The actions may influence future states and rewards.

Markov Decision Process: the canonical way to model RL problems:

- Policy: $\pi$ is a strategy for assigning actions to states.
- Long term view of the quality of a policy: Value $v_\pi(s)$, Action-value $q_\pi(s, a)$ capture expected cumulative discounted reward
- Goal: Find a policy that maximises value.

The RL **environment** with an MDP is characterised completely by: transition function $p(s' \mid s, a)$, reward function $r(s, a)$ which constitute the _dynamics_ of the MDP.

Given an MDP, $(\mathcal{S}, \mathcal{A}, p, r)$, the quantities of interest are:

- Policy $\pi(a \mid s)$.
- Discount factor $\gamma \in [0, 1)$.
- Return $G_t = \sum_{k=0}^\infty{\gamma^k R_{t+1+k}}$.
- State-value $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$.
- Action-value $q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$.

Reward is enough, D. Silver et al., 2021: All of what we mean by _goals_ and purposes can be well thought of as the _maximisation_ of the _expected value_ of the _cumulative sum_ of a scalar signal, called _reward_.

Goals and **rewards**: Rewards indicate the goal / purpose, not the strategy; that is, rewards indicate the WHAT not the HOW. The agent otherwise could focus on the intermediate steps and forget about the real goal: winning.

A **policy** is the agent's strategy for assigning actions to states: $\pi: \mathcal{S} \rarr \mathcal{A}$ (can be stochastic too). The _goal_ is to find a policy that maximises expected cumulative reward. The _value_ $v_\pi(s)$ and _action-value_ $q_\pi(s, a)$ capture the expected cumulative discounted reward.

**Markov property**: Future states and rewards are independent of past states and actions, given the current state and action: state $S_t$ is a sufficient summary of interaction history.

### 3.2. Policies and value functions

An MDP is controlled with a **policy**: $\pi(a \mid s)$ is the probability of selecting $a$ when in state $s$ under policy $\pi$.

- Special case: deterministic policy $\pi(s) = a$, e.g. `high -> search, low -> recharge`.
- An MDP coupled with a fixed policy $\pi$ is a _Markov chain_.

The **state-value function** $v$ of a state $s$ under a policy $\pi$, $v_\pi$, is the expected return when starting in $s$ at timestep $t$ and following $\pi$ from there on:

$$v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\bigg[ \sum_{k=0}^\infty{ \gamma^k R_{t+k+1} \mid S_t = s } \bigg]$$

If we want to evaluate a policy, we compute its value functions!

Agent's goal is to learn a policy that maximises cumulative reward.

- Assuming _terminating episodes_, e.g. can enforce termination by setting number of allowed time steps.
- For non-terminating (infinite) episodes, can use **discount rate** $\gamma \in [0, 1)$, e.g. financial portfolio management, one cookie now or many later?
  - Low $\gamma$ is shortsighted.
  - High $\gamma$ is farsighted.
  - A reward received $k$ timesteps in the future is only worth $\gamma^{k-1}$ times what it would be worth if it was received immediately.

$$G_t \doteq \sum_{k=0}^\infty{\gamma^k R_{t+1+k}}$$

This sum is _finite_ for $\gamma < 1$ and _bounded_ for rewards $R_t \leq r_\text{max}$:

$$\sum_{k=0}^\infty{\gamma^k R_{t+1+k}} \leq r_\text{max} \sum_{k=0}^\infty{\gamma^k} = r_\text{max} \frac{1}{1 - \gamma}$$

And the definition also works for terminating episodes if terminal are **absorbing**: an absorbing state always transitions into itself and gives reward zero.

### 3.1. Bellman Equations

By virtue of the Markov property, the **state-value** and **action-value** functions can be written in recursive form (i.e. a look-ahead tree):

$$
v_\pi(s) \doteq \mathbb{E}[G_t \mid S_t = s]
= \sum_a{ \pi(a \mid s) \sum_{s', r}{ p(s', r \mid s, a) [r + \gamma v_\pi(s')] } }
$$

$$q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \sum_{s', r}{ p(s', r \mid s, a) [r + \gamma v_\pi(s')] }$$

Both equations represent the sum of the _immediate reward plus the discounted expectd future value_ ($\mathbb{E}_{s'}[v_\pi(s')] = \sum_{s' \in \mathcal{S}}{ p(s' \mid s, a) \cdot v_\pi(s') }$):

$$v_\pi(s) = \sum_{a \in \mathcal{A}}{\pi(a \mid s) r(s, a) + \gamma \sum_{s' \in \mathcal{S}}{ p(s' \mid s, a) \cdot v_\pi(s') }}$$

$$q_\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}}{ p(s' \mid s, a) \cdot v_\pi(s') }$$

### 3.3. Optimality

**Bellman optimality equations**: A policy $\pi$ is optimal if (can be expressed without reference to policy):

$$
v_\pi(s) = v_*(s) = \max_{\pi'}{v_{\pi'}(s)}
= \max_a{ \sum_{s', r}{ p(s', r \mid s, a) [r + \gamma v_*(s')] } }
$$

$$q_\pi(s, a) = q_*(s, a) = \max_{\pi'}{q_{\pi'}(s, a)} = \sum_{s', r}{ p(s', r \mid s, a) [r + \gamma \max_{a'}{q_*(s', a')}] }$$

By the Bellman Equation, this implies that for any optimal policy: $\forall \hat{\pi}, s: v_\pi(s) \geq v_{\hat{\pi}}(s)$.

What can be said about the value, $v_\pi(s)$ of a policy $\pi$ when $\gamma=0.5$ vs. $\gamma=0.9$? When are they equal, if ever?

- With $\gamma=0.5$, future rewards are discounted more aggressively. The agent focuses more on immediate rewards and less on long-term returns.
- With $\gamma=0.9$, the agent considers long-term rewards more significantly. A higher $\gamma$ generally leads to a larger $v_\pi(s)$, assuming rewards are positive.
- If all future rewards beyond the immediate step are zero, then $\gamma$ has no effect, and the value functions are equal: This happens in episodic tasks that end immediately.

The Bellman optimality equation for $v_\pi$ forms a system of $n = \lvert \mathcal{S} \rvert$ linear equations with $n$
variables (for finite MDPs): $v_\pi(s_1) = \dots, v_\pi(s_2) = \dots, \dots, v_\pi(s_n) = \dots$

- The optimal value function $v_\pi$ is the unique solution to the system.

The Bellman optimality equation for $v_∗$ forms a system of $n$ non-linear equations with $n$ variables (for finite MDPs):

- Equations are non-linear due to the $\max$ operator.
- The optimal value function $v_∗$ is the unique solution to the system.

Have we solved RL? We are making two assumptions: we have $p(s', r \mid s, a)$, we have engouh computational power. DP, for example, provides an approximate solution to the Bellman equations.

### 3.4. Partial and full observability

### 3.5. Ergodicity, Discounting and Average Reward

For finite MDPs and non-terminating episodes, any policy $\pi$ will produce an **ergodic** set of states $\hat{\mathcal{S}}$:

- Every state in $\hat{\mathcal{S}}$ is visited infinitely often.
- Steady-state distribution $P_\pi(s) = \lim_{t \rarr \infty}{ \text{Pr}\{ S_t = s \mid A_0, \dots, A_{t-1} \sim \pi \} }$.

Policy performance can be measured by **average reward**, independent of initial state $S_0$!

$$
r(\pi) \doteq \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^{h} \mathbb{E} [ R_t \mid S_0, A_0, ..., A_{t-1} \sim \pi]
= \sum_{s} P_{\pi}(s) \sum_{a} \pi(a | s) \sum_{s', r} p(s', r | s, a) r
$$

Maximising discounted return over the steady-state distribution is the same as maximising average reward - $\gamma$ has no effect on the maximisation!

$$\sum_{s} P_{\pi}(s) v_{\pi}(s) = \sum_{s} P_{\pi}(s) \sum_{a} \pi(a | s) \sum_{s',r} p(s', r | s, a) [r + \gamma v_{\pi}(s')]$$

$$= r(\pi) + \sum_{s} P_{\pi}(s) \sum_{a} \pi(a | s) \sum_{s',r} p(s', r | s, a) [\gamma v_{\pi}(s')]$$

$$= r(\pi) + \gamma \sum_{s'} P_{\pi}(s') v_{\pi}(s')$$

$$= r(\pi) + \gamma \left[ r(\pi) + \gamma \sum_{s'} P_{\pi}(s') v_{\pi}(s') \right]$$

$$= r(\pi) + \gamma r(\pi) + \gamma^2 r(\pi) + \gamma^3 r(\pi) + \cdots$$

$$= r(\pi) \frac{1}{1 - \gamma} \quad \Rightarrow \quad \gamma \text{ has no effect on maximisation!}$$

Focus on discounted returns:

- Most of current RL theory was developed for discounted returns.
- Discounted and average return settings both give the same limit results for $\gamma \rarr 1$, which is why people often use $\gamma \in [0.95, 0.99]$.
- Discounted returns work well for finite and infinite episodes.

TODO: read Tsitsiklis, J., Van Roy, B. (2002). On Average Versus Discounted Reward Temporal-Difference Learning. Machine Learning 49, 279-191.

## 4. Dynamic Programming

- Dynamic = problems of a sequential nature (e.g. time).
- Programming = finding optimal programs (i.e. policies).

DP is a general solution method for problems with _two properties_:

1. Optimal substructure (principle of optimality)
2. Overlapping subproblems

Are MDPs good candidates for DP? The value of a state $\textcolor{blue}{v_*(s)} = \max_a{\sum_{s', r}{p(s', r \mid s, a) [r + \gamma \textcolor{blue}{v_*(s')}]}}$ is a composition of the value of successor states, i.e. a substructure - value functions are caches of partial solutions!

We will focus on the planning problem. We have perfect knowledge of the model, i.e. $p(s', r \mid s, a)$.

_Solving_ an MDP: Use Bellman Equations to organise search for good policies.

- Given: an MDP $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$.
- Output: an optimal policy $\pi_* = \argmax_\pi{v_\pi(s), \, \forall s}$.

Interfacing policy evaluation (/ prediction) and policy improvement in an iterative mechanism:

![Dynamic Programming](./assets/dp.png)

Remember:

- The _state-value function_ $v_\pi(s)$ is the expected return when starting in $s$ and following $\pi$ afterwards: $v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s]$.
- The _action-value function_ $q_\pi(s, a)$ is the expected return when starting in $s$, taking action $a$ and following $\pi$ afterwards: $q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$.

### 4.1. Policy Iteration

1. Start from a random policy $\pi(a \mid s)$.
2. Policy Evaluation (E; using a DP algorithm): compute the $n = \lvert S \rvert$ state-value functions $v_\pi(s)$ for policy $\pi$.
   - Gaussian elimination has complexity $\mathcal{O}(n^3)$.
3. Policy Improvement (I; using the evaluations to find a better policy): make policy $\pi$ **greedy** w.r.t. $v_\pi$:

   $$\pi'(s) = \argmax_a{q_\pi(s, a)} = \argmax_a{ \sum_{s', r}{ p(s', r \mid s, a) [r + \gamma v_k(s')] } }$$

Process converges to optimal policy!

#### Iterative Policy Evaluation (/ Prediction)

1. Initialise $v_0(s) = 0$ (can be arbitrary value functions, but terminal states must be initialised to zero).
2. Perform _expected update_ $v_k \rarr v_{k+1}$ for each state $s$:

$$v_{k+1}(s) = \sum_a{ \pi(a \mid s) \sum_{s', r}{ p(s', r \mid s, a) [r + \gamma v_k(s')] } }$$

The sequence $\{v_k\}$ converges to $v_\pi$ as $k \rarr \infty$.

#### Policy Improvement

Given a value function $v_\pi(s)$ for $\pi(a \mid s)$, compute a policy $\pi'(a \mid s)$ that is "better" than $\pi(a \mid s)$.

#### Policy Improvement Theorem

Let $\pi, \pi'$ be deterministic policies s.t. $\forall s, \, \sum_a{ \pi'(a \mid s) q_\pi(s, a) = q_\pi(s, \pi'(s)) \geq v_\pi(s) = \sum_a{ \pi(a \mid s) q_\pi(s, a) } }$.

Then $\pi'$ must at least as good or **better** than $\pi$ (i.e. it will obtain greater or equal expected return in all states $s$):

$$\forall s: v_{\pi'}(s) \geq v_\pi(s)$$

_Proof for deterministic policies_:

$$v_{\pi}(s) \leq q_{\pi}(s, \pi'(s))$$

$$= \mathbb{E}_{\pi'} \big[ R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t = \pi'(s) \big]$$

$$= \mathbb{E}_{\pi'} \big[ R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s \big]$$

$$\leq \mathbb{E}_{\pi'} \big[ R_{t+1} + \gamma q_{\pi}(S_{t+1}, \pi'(S_{t+1})) \mid S_t = s \big] \quad \text{(by premise)}$$

$$= \mathbb{E}_{\pi'} \big[ R_{t+1} + \gamma \mathbb{E}_{\pi'} [ R_{t+2} + \gamma v_{\pi}(S_{t+2}) \mid S_{t+1}, A_{t+1} = \pi'(S_{t+1}) ] \mid S_t = s \big]$$

$$= \mathbb{E}_{\pi'} \big[ R_{t+1} + \gamma R_{t+2} + \gamma^2 v_{\pi}(S_{t+2}) \mid S_t = s \big]$$

$$\leq \mathbb{E}_{\pi'} \big[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 v_{\pi}(S_{t+3}) \mid S_t = s \big]$$

$$\vdots$$

$$\leq \mathbb{E}_{\pi'} \big[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \dots \mid S_t = s \big]$$

$$= v_{\pi'}(s)$$

_Picking the **greedy** action w.r.t._ $q_\pi(s, a)$ _in each state satisfies the condition of the theorem_:

$$q_\pi(s, \pi'(s)) = \max_a{q_\pi(s, a)} \geq \sum_a{\pi(s \mid a) q_\pi(s, a)} = v_\pi(s)$$

What if $\pi' = \pi$? $\pi'$ gives all probability to $\hat{a}$: $v_\pi(s) = \sum_a{\pi(s \mid a) q_\pi(s, a)} = q_\pi(s, \hat{a}) = \max_a{q_\pi(s, a)}$. This is the Bellman optimality equation: $v_\pi(s) = \sum_a{\pi(s \mid a) q_\pi(s, a)} = \max_a{q_\pi(s, a)} = \max_a{ \sum_{r, s'}{p(r, s' \mid a, s) [r + \gamma v_\pi(s')]} }$, i.e. the policy is optimal!

### 4.2. Value Iteration

Iterative policy evaluation uses Bellman equation as operator: $v_{k+1}(s) = \sum_a{ \pi(a \mid s) \sum_{s', r}{ p(s', r \mid s, a) [r + \gamma v_k(s')] } } \quad \forall s \in \mathcal{S}$. It may take many sweeps $v_k \rarr v_{k+1}$ to converge. Do we have to wait until convergence before policy improvement $\pi'(s) = \argmax_a{\sum_{r, s'}{p(r, s' \mid a, s) [r + \gamma v_\pi(s')]}}$?

Value iteration _combines_ one sweep of policy evaluation and policy improvement by using _Bellman optimality equation_ as (iterative) operator:

$$v_{k+1}(s) = \max_a{ \sum_{r, s'}{ p(r, s' \mid s, a) [r + \gamma v_k(s')] } } \quad \forall s \in \mathcal{S}$$

Sequence converges to optimal policy (can show that Bellman optimality operator is $\gamma$-contraction).

### 4.3. Asynchronous DP

Performiing policy evaluation and improvement for all states is prohibitive if the state space is large.

Asynchronous DP methods evaluate and improve policy on subset of states:

- Gives flexibility to choose best states to update, e.g. random states, recently visited states (real-time DP).
- Parallelisation (on multiple processors).
- Still guaranteed to converge to optimal policy if all states in $\mathcal{S}$ are updated infinitely many times in the limit.

### 4.4. Generalised Policy Iteration (GPI)

Policy iteration, Value iteration, Asynchronous DP vary in the granularity of the interleaving between evaluation and improvement.

But the idea of this interleaved process is general enough for describing most of RL.

**Bootstrapping**: new estimates are based on old estimates.

### 4.5. Efficiency of DP

TODO: read <https://arxiv.org/pdf/1302.4971>

## 5. Monte Carlo Methods

Planning model: given the model, find an optimal policy. DP solutions:

- Policy Iteration (Policy Evaluation, Bellman Expected Update, + Greedy Policy Improvement)
- Value Iteration (Bellman Optimal Update)
- Generalized Policy Iteration

DP methods iterate through policy evaluation and improvement until convergence to optimal value function $v_∗$ and policy $\pi_∗$:

- Policy evaluation via repeated application of Bellman operator.
- Requires complete knowledge of MDP model: $p(s', r \mid s, a)$.

Can we compute optimal policy without knowledge of complete model?

Often, we do not have a model of the environment. We want to learn from experience, i.e. interaction with the environment. Already saw something like this in bandits: did not have access to the reward distribution $p(r \mid a)$ and estimated the values from experience with the _sample-average method_: $Q_t(a) \doteq \frac{\text{sum of the rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t}$.

Monte Carlo methods:

- Solving reinforcement learning by averaging sample _returns_
  - Compare bandits: average rewards.
  - MC: average returns.
- We focus on episodic tasks
- We sample an episode and then change our estimates of values and policies
  - Compare bandits: estimate $Q(a)$.
  - MC: multiple states -> estimate $Q(s, a)$, i.e. one bandit problem per state.
- Episode-by-episode learning

Monte Carlo (MC) methods learn value function based on experience: entire episodes $E^i = \langle S_0^i, A_0^i, R_1^i, S_1^i, A_1^i, R_2^i, \dots, S_{T_i}^i \rangle$.

Two ways to obtain episodes:

- Real experience: generate episodes directly from “real world”.
- Simulated experience: use simulation model $\hat{p}$ to sample episodes — $\hat{p}(s, a)$ returns a pair $(s', r )$ with probability $p(s', r |s, a)$.

Same steps as for DP:

1. Prediction (Evaluation)
2. Control (i.e. optimal policy by GPI)

### 5.1. MC Policy Prediction (Evaluation)

The Bellman equation for policy evaluation is $v_\pi(s) = \sum_a{ \pi(a \mid s) \sum_{s', r}{ \textcolor{green}{p(s', r \mid s, a)} [r + \gamma v_\pi(s')] } }$. The model $\textcolor{green}{p(s', r \mid s, a)}$ is unknown.

$v_\pi(s)$ is the _expected cumulative future discounted reward starting from_ $s$ _and following_ $\pi$.

MC:

- Collect many episodes obtained by following $\pi$ and passing through $s$.
- Estimate the value function by averaging sample returns observed after visiting $s$:

$$v_\pi(x) \doteq \mathbb{E}\bigg[ \sum_{k=t}^{T-1}{\gamma^{k-t} R_{k+1}} \mid S_t = s \bigg] \approx \frac{1}{\lvert \varepsilon(s) \rvert} \sum_{t_i \in \varepsilon(s)}{ \sum_{k=t_i}^{T_i-1}{\gamma^{k-t_i} R_{k+1}^i} }$$

Two modalities:

- First-visit MC: only the first time of encountering $s$ in the episode is considered, i.e. $\varepsilon(s)$ contains _first_ $t_i$ for which $S_{t_i}^i = s$ in $E^i$.
- Every-visit MC: all occasions of encountering $s$ in the episode are considered, i.e. $\varepsilon(s)$ contains _all_ $t_i$ for which $S_{t_i}^i = s$ in $E^i$.

Both methods converge to $v_\pi(s)$ as $\lvert \varepsilon(s) \rvert \rarr \infty$.

**States in Blackjack**: Couldn't we just define states as $S_t = \{\text{Player cards}, \text{Dealer card}\}$?

- Tricky: states would have variable length (player cards)
- If we fix maximum number of player cards to 4, then there are $10^5 = 100,000$ possible states! (ignoring face cards and ordering)

Blackjack example uses _engineered state features_:

- Fixed length: St = (Player sum, Dealer card, Usable ace?)
- Player sum limited to range 12–21 because decision below 12 is trivial (always hit)
- Number of states: 10 ∗ 10 ∗ 2 = 200 → much smaller problem!
- Still has all relevant information

Can we solve Blackjack MDP with DP methods?

- Yes, in principle, because we know complete MDP (remember: need knowledge of complete MDP!)
- But computing $p(s', r |s, a)$ can be complicated! E.g. what is probability of $+1$ reward as function of Dealer's showing card?
- On other hand, easy to code a simulation model:
  - Use Dealer rule to sample cards until stick/bust, then compute reward
  - Reward outcome is distributed by $p(s', r |s, a)$
- MC can evaluate policy without knowledge of probabilities $p(s', r |s, a)$

DP vs. MC predictions:

- DP backup diagram: all possible transitions $(s, s')$, only 1 step into the future; bootstrap
  - still need to compute the values of all other states
- MC backup diagram: a single _sampled_ transition per state, entire episode trajectory; no bootstrap
  - sample only episodes starting from $s$, do not use $v(s')$

### 5.2. MC Estimation of Action Values

MC methods can learn $v_\pi(s) = \sum_a{\pi(a \mid s) q_\pi(s, a)}$ without knowledge of model $p(s', r |s, a)$, by simply collecting complete episodes and averaging returns.

But improving policy $\pi$ from $v_\pi$ requires model! When improving a policy, we want to create a new policy $\pi'$ that is greedy with respect to the current value function:

$$\pi'(s) = \arg\max_a{q_\pi(s, a)} = \arg\max_a \sum_{s', r} p(s', r|s, a) [r + \gamma v_\pi(s')]$$

This equation directly uses the transition model $p(s', r|s, a)$ to determine the expected return for each action. Without the model, we cannot compute this expectation.

Our goal is to improve the policy: but we can never know if another action would have been better if we have no value estimated for it in $s$. This is an **exploration** issue in evaluation action-values with Monte Carlo not encountered in DP.

Two solutions:

- Exploring starts: we ensure that we start from $(s,a)$ pairs instead than on states. All $(s,a)$ pair should have a non-zero probability to be the start.
- Stochastic policies: in any state, all actions should have a non-zero probability of being selected

Must estimate action values: $q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$ - using same MC methods to learn $q_\pi$, but visits are to $(s, a)$-pairs. Converges to $q_\pi$ if every $(s, a)$-pair visited infinitely many times in limit, e.g. **exploring starts**: every $(s, a)$-pair has non-zero probability of being starting pair of episode

### 5.3. MC Control (Improvement)

MC policy evaluation: estimate $q_\pi$ using MC method.

MC Policy improvement: given the value function $Q_{\pi_k}(s, a)$ for the given policy $\pi(a \mid s)$, find a policy $\pi_{k+1}(a \mid s)$ that is "better" than $\pi_k(a \mid s)$.

We have access already to the estimation of state-action
value function $Q_{\pi_k}(s, a)$ for the current policy $\pi_k(s)$. Then, the improved policy $\pi_{k+1}(s) = \arg\max_a{Q_{\pi_k}(s, a)}$ - _the greedy policy w.r.t._ $Q$

Greedy policy meets conditions for policy improvement, according to the _policy improvement theorem_:

$$
\begin{align*}
q_{\pi_k}(s, \pi_{k+1}(s)) & = q_{\pi_k}(s, \argmax_a{q_{\pi_k}(s, a)}) \\
& = \max_a{q_{\pi_k}(s, a)} \\
& \geq q_{\pi_k}(s, \pi_k(s)) \qquad \text{by definition of } \max_a{q_{\pi_k}} \\
& = v_{\pi_k}(s)
\end{align*}
$$

### 5.4. MC Control without Exploring Starts

Same idea as generalised policy iteration (GPI) in DP:

- Evaluation: update estimate $Q$ toward the true value function $q_\pi$ of $\pi$, $Q \rightsquigarrow q_\pi$.
- Improvement: improve the policy $\pi$ toward the greedy policy of estimate $Q$, $\pi \rightsquigarrow \text{greedy}(Q)$.

Two assumptions:

1. Perfect evaluation -> infinite episodes
2. Exploration -> exploring starts from all $(s,a)$ pairs

Assuming exploring starts and infinite MC iterations:

- In practice, we update only to a given performance threshold
- Or alternate between evaluation and improvement per episode

MC control with imperfect evaluation: relax the requirement that we find the exact value function for the current policy.

1. Close to the exact one but allowing some approximation (e.g. using bounds)
   - Reducing the number of episodes of the evaluation;
2. Not even close to the exact one; we just move in the direction of the optimal
   - In the extreme, just one episode!

**Blackjack Example with MC–ES**: Policy stick: if player sum
is 20 or 21, else hit; exploring starts: sample initial states uniformly randomly.

Convergence to $q_\pi$ requires that all $(s, a)$-pairs are visited infinitely many times. Exploring starts guarantee this, but impractical. (TODO: why?) Other approach: use soft policy such that $\pi(a \mid s) > 0$ for all $s, a$.

$q_\pi(s, \pi'(s)) = v_\pi(s)$ only when $\pi'$ and $\pi$ both optimal $\epsilon$-soft policies.

$\epsilon$-soft policies: the policy we get will always explore randomly in $\epsilon$ percent of the cases.

MC control without exploring starts:

- With exploring starts, we give a probability of selecting unexplored $(s, a)$ pairs
- Without exploring starts, the policy should give some probability of exploring $(s, a)$ pairs

In an $\epsilon$-greedy policy:

- With probability $\epsilon$, choose an action at random
- Otherwise, choose the greedy action

A policy is said to be an $\epsilon$-soft policy if, for all $s$ and for all $a$, any action has at least $\frac{\epsilon}{\lvert A(s) \rvert}$ to be picked:

$$\pi(a \mid s) \geq \frac{\epsilon}{\lvert A(s) \rvert}$$

$\epsilon$-greedy is the greediest of the $\epsilon$-soft policies.

### 5.5. Off-policy Prediction via Importance Sampling

Like exploring starts, soft policies ensure all $(s, a)$ are visited infinitely many times

- But policies restricted to be soft - optimal policy is usually deterministic!
- Could slowly reduce $\epsilon$, but not clear how fast

Other approach: off-policy learning

- Learn $q_\pi$ based on experience generated with behaviour policy $\mu \neq \pi$.
- Requires “coverage”: if $\pi(a \mid s) > 0$ then $\mu(a \mid s) > 0$, for all $s, a$, e.g. use soft policy $\mu$.
- $\pi$ can be deterministic - usually the greedy policy

Difference:

- On-policy TD control: Learn $q_\pi$ with experience generated using policy $\pi$.
  - _The target policy_ (to update) _is the same as the policy used to explore_ (to collect experiences / expisodes).
- Off-policy TD control: Learn $q_\pi$ with experience generated using policy $\mu \neq \pi$.

Monte Carlo dilemma: The learning nature of RL creates a problem:

1. We want to find the optimal policy
2. We need to behave non-optimally to explore all actions

Off-policy learning: use two different policies So the data we collect are from the behaviour policy:

- one to explore (behaviour policy)
- one to evaluate and improve (target policy)

Let’s focus on the prediction problem:

- Given a policy $\pi$
- Find its value function $v_\pi$ or $q_\pi$

The off policy prediction problem:

- Given a target policy $\pi$
- Given a behavioral policy $b$ or $\mu$
- Find value function $v_\pi$ or $q_\pi$ of $\pi$

We can compute an estimate on the returns $G_t$ that we get when following $b$, i.e. $v_b(s) = \mathbb{E}_b[G_t \mid S_t = s]$ which is $\neq v_\pi(s)$.

Coverage: $b$ should always give at least as little probabilities to the state action pairs in which $\pi(a \mid s) > 0$:

$$\pi(a \mid s) > 0 \Rarr b(a \mid s) > 0$$

Otherwise we could have no information of $(s, a)$ pairs to estimate $q_\pi(s, a)$.

If in a state $s$ it holds that $\pi(s) \neq b(s)$, then $b$ should be stochastic in $s$.

Tool: Importance sampling

- Given samples from $b$: $s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, \dots, r_n$
- Compute $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$

Given a policy $\pi$ and a starting state $S_t$, we can compute the _probability of a trajectory_ $A_t, S_{t+1}, A_{t+1}, \dots, S_T$ as:

$$
\begin{align*}
  P(A_t, S_{t+1}, A_{t+1}, \dots, S_T \mid S_t, \pi) & = \pi(A_t, S_t) p(S_{t+1} \mid S_t, A_t) \pi(A_{t+1} \mid S_{t+1}) \dots \\
  & = \prod_{k=t}^{T-1}{\pi(A_k \mid S_k) p(S_{k+1} \mid A_k, S_k)}
\end{align*}
$$

The relative probability of the samples (or trajectory) under the behavior and target policy is the importance-sampling ratio

Importance Sampling Ratio: For episodes generated from $\mu$ expected return $G_t$ at $t$ is $\mathbb{E}_\mu[G_t \mid S_t = s] = v_\mu(s) \neq v_\pi(s)$. Fix expectation with sampling importance ratio

$$
\rho_{t:T} \doteq \frac{\prod_{k=t}^{T-1}{\pi(A_k \mid S_k) p(S_{k+1}, R_{k+1} \mid S_k, A_k)}}{\prod_{k=t}^{T-1}{\mu(A_k \mid S_k) p(S_{k+1}, R_{k+1} \mid S_k, A_k)}}
= \prod_{k=t}^{T-1}{\frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)}}
$$

s.t.

$$
\begin{align*}
  \mathbb{E}_\mu[\rho_{t:T} G_t \mid S_t = s] & = \sum_{E: S_t = s}{\Bigg[ \prod_{k=t}^{T-1}{\mu(A_k \mid S_k) p(S_{k+1}, R_{k+1} \mid S_k, A_k)} \Bigg]} \rho_{t: T} G_t \\
  & = \sum_{E: S_t = s}{\Bigg[ \prod_{k=t}^{T-1}{\mu(A_k \mid S_k) p(S_{k+1}, R_{k+1} \mid S_k, A_k)} \Bigg]} \prod_{k=t}^{T-1}{\frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)}} G_t \\
  & = \sum_{E: S_t = s}{\Bigg[ \prod_{k=t}^{T-1}{\pi(A_k \mid S_k) p(S_{k+1}, R_{k+1} \mid S_k, A_k)} \Bigg]} G_t \\
  & = v_\pi(s)
\end{align*}
$$

Importance sampling helps us:

$$v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_b[\rho_{t:T-1} G_t \mid S_t = s]$$

- $t$ index is an _inter-episode_ index, e.g. first episode $t=0$ to $t=50$; Second episode $t=51$ to $t=80$
- $T(t)$ is the termination step after $t$, e.g. $T(32) = 50$ ($32$ is a timestep of the first episode which ends at $t=50$)
- $\mathcal{T}(s)$ is the set of timesteps we encountered s (inter-episodes; every-visit)

Ordinary vs. weighted importance sampling in MC off-policy prediction:

- Ordinary importance sampling:
  1. Sample several episodes using the behaviour policy $b$
  2. Compute the returns $G_t$ for each state $S_t = t$ and the ratio $\rho_{t:T(t)-1}$
  3. Compute the predicted value under $\pi$ for each $s$:
     $$V(s) = \frac{\sum_{t \in \mathcal{T}(s)}{\rho_{t:T(t)-1} G_t}}{\lvert \mathcal{T}(s) \rvert}$$
- Weighted importance sampling:
  1. Sample several episodes using the behaviour policy $b$
  2. Compute the returns $G_t$ for each state $S_t = t$ and the ratio $\rho_{t:T(t)-1}$
  3. Compute the predicted value under $\pi$ for each $s$:
     $$V(s) = \frac{\sum_{t \in \mathcal{T}(s)}{\rho_{t:T(t)-1} G_t}}{\sum_{t \in \mathcal{T}(s)}{\rho_{t:T(t)-1}}}$$

Example: $s_0, a_0, r_1, s_1, \dots, G_0, \mathcal{T}(s_0) = \{0\}, \rho_{t:T(t)-1} = 10$

- Ordinary: $V(s) = \frac{\sum_{t \in \mathcal{T}(s)}{\rho_{t:T(t)-1} G_t}}{\lvert \mathcal{T}(s) \rvert} = \frac{10 G_0}{1} = 10 G_0$
- Weighted: $V(s) = \frac{\sum_{t \in \mathcal{T}(s)}{\rho_{t:T(t)-1} G_t}}{\sum_{t \in \mathcal{T}(s)}{\rho_{t:T(t)-1}}} = \frac{10 G_0}{10} = G_0 = V_b(s)$

### 5.6. Incremental Implementation

### 5.7. Off-policy Control

### 5.8. Discounting-Aware Importance Sampling

### 5.9. Per-Decision Importance Sampling

## 6. Temporal-Difference Learning

Often, we do not have a model of the environment. We want to learn from experience through interaction with the real environment: Monte Carlo.

We would like to learn continuously, not only at the end of episodes.

| Method | Model-free? | Bootstrap? |
| ------ | ----------- | ---------- |
| DP     | No          | Yes        |
| MC     | Yes         | No         |
| TD     | Yes         | Yes        |

- TD learns directly from experience (like MC)
- TD is model-free, do not need $p(s', r \mid s, a)$ (like MC)
- TD learns from incomplete episodes (unlike MC)
- TD updates an estimate towards another estimate (bootstrapping; like DP)

**General iterative update rule**:

$$
\begin{align*}
  \text{NewEstimate} & \larr \text{OldEstimate} + \text{StepSize} [ \text{Target} - \text{OldEstimate} ] \\
  & \larr (1 - \alpha) \text{OldEstimate} + \alpha \text{Target}
\end{align*}
$$

In MC, we have seen the following incremental formulation of the update rule for the value function: $V(S_t) = \text{average}(\text{Returns}(S_t))$

- MC update: $V(S_t) \larr V(S_t) + \alpha [G_t - V(S_t)]$

Why should we wait the end of the episode? We must wait the end of the episode to compute $G_t$.

We would like to learn at each transition $S_t, A_t, R_{t+1}, S_{t+1}$ from the knowledge we collected from the environment $R_{t+1}$. Can we compute $G_t$ using $R_{t+1}$?

Recall the most important definition in RL: $G_t = R_{t+1} + \gamma G_{t+1}$. Still, we do not know $G_{t+1}$.

**Bootstrapping**: use the current estimate $G_{t+1} \approx V(S_{t+1})$.

- TD(0) update (with $\delta$-error): $V(S_t) \larr V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

Compare:

- MC: $V(S_t) \larr V(S_t) + \alpha [R_{t+1} + \gamma G_{t+1} - V(S_t)], \, G_t = R_{t+1} + \gamma G_{t+1}$
- TD(0): $V(S_t) \larr V(S_t) + \alpha [R_{t+1} + \gamma V_{t+1} - V(S_t)]$, with approximation $G_t \approx R_{t+1} + \gamma V_{t+1}$

**Markov reward process**: We are not interested in the decision part of the MDP, just in the reward process.

- Intuitively, the agent becomes part of the environment.
- We do not need to know how and who takes decisions.
- For evaluation, we use updates $V(S_t) \larr V(S_t) + \alpha (G_t - V(S_t))$. No policy!

In a Markov reward process, we observe just sequences of states and rewards and we can still evaluate how good a state is: $S_0, R_0, S_1, R_1, S_2, R_2, \dots$

Example: driving home

DP vs. MC vs. TD:

$$
\begin{align*}
  v_\pi(s) & \doteq \mathbb{E}_\pi[G_t \mid S_t = s] \\
  & = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
  & = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]
\end{align*}
$$

- MC approximates directly the definition: instead of real expectation, we use _samples_: $\mathbb{E} \rarr \frac{1}{N} \sum_\text{samples}$
- DP approximates the recursive definition: exact expectation but _bootstrapping old estimates_: $v_\pi(S_{t+1}) \rarr V(S_{t+1})$
- TD approximates both!

![Dynamic Programming vs. Monte-Carlo vs. Temporal-Difference](./assets/dp-mt-td.png)

### 6.1. TD Prediction

Advantages:

- TD full incremental / online
  - Long episodes – very slow learning with MC
  - Continuing tasks
- You do not need the final outcome

Bias / variance trade-off:

- MC update is unbiased:
  - The MC update builds estimates $V(s)$ of $G_t$ that are unbiased w.r.t. the true $v_\pi(s)$ (i.e. on average they are the same).
  - With enough experience $V(s)$ will converge to $v_\pi(s)$
- The TD _true_ update is unbiased: $V(S_t) \larr V(S_t) + \alpha (R_{t+1} + \gamma \textcolor{red}{v_\pi(S_{t+1})} - V(S_t))$
  - since $v_\pi(s)$ is the correct one, we are not introducing biases
  - just the Bellman equation
- The TD update is biased: $V(S_t) \larr V(S_t) + \alpha (R_{t+1} + \gamma \textcolor{red}{V(S_{t+1})} - V(S_t))$
  - our current estimate could be very wrong
  - we could introduce any kind of error in our estimate on $V(S_t)$
- MC has high variance: $V(S_t) \larr V(S_t) + \alpha (\textcolor{red}{R_{t+1} + \dots + \gamma^{T - t + 1} R_{t+T}} - V(S_t))$
- TD has lower variance: $V(S_t) \larr V(S_t) + \alpha (\textcolor{red}{R_{t+1} + \gamma V(S_{t+1})} - V(S_t))$

| MC                                                      | TD                                           |
| ------------------------------------------------------- | -------------------------------------------- |
| good convergence guarantees                             | usually more efficient                       |
| (also with function approximation)                      | $TD(0)$ converges despite bias               |
| not very sensitive to initial values (no bootstrapping) | (but not always with function approximation) |
| very simple                                             | more sensitive to initial value              |

Convergence = in the limit = infinite number of episodes.

We know that MC and TD are guaranteed to converge to the true $v_\pi(s)$.

Suppose we have a _limited amount of episodes_ on which we apply several times both MC and TD prediction. They will converge to some approximate value function. Is it the same?

- MC converges to the solution with minimum error (MSE $= \sum_{e_i}\sum_t{(g_t^k - V(s_t^k))^2}$): best fit to the current returns in the episodes.
- TD converges to the Markov model with the maximum-likelihood of explaining data
  - makes the assumption that there is a Markov model
  - computes the values of a hypothetical Markov model that maximises the likelihood of the data

Take-away:

- TD exploits the Markov property, and is more effective if the environment is Markov
- MC does not exploit the Markov property, and is more effective if the environment is non-Markov

**Convergence**: TD(0) converges to $v_\pi$ with probability 1 if

- all states visited infinitely often, and
- standard stochastic approximation conditions ($\alpha$-reduction): $\forall s: \sum_{t: S_t = s}{\alpha_t \rarr \infty} \text{ and } \sum_{t: S_t = s}{\alpha_t^2 < \infty}$.

Expected TD update moves $V(S_t)$ toward $v_\pi(S_t)$ by $\alpha$ ($\alpha$ used to control averaging in sampling updates):

$$
\begin{align*}
  V(S_t) & \larr \mathbb{E}_\pi\big[(1 - \alpha) V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1})] \big] \\
  & = (1 - \alpha) V(S_t) + \alpha \mathbb{E}_\pi [R_{t+1} + \gamma V(S_{t+1})] \\
  & = (1 - \alpha) V(S_t) + \alpha sum_a{\pi(a \mid S_t)} \sum_{s', r} {p(s', r \mid S_t, a) [r + \gamma V(s')]} \\
  & = (1 - \alpha) V(S_t) + \alpha v_\pi(S_t)
\end{align*}
$$

**Advantages**:

- Like MC: TD does not require full model $p(s', r \mid S_t, a)$, only experience.
- Unlike MC: TD can be fully incremental.
  - Learn before final return is known.
  - Less memory and computation.
- Both MC and TD converge to $v_\pi / q_\pi$ under certain assumptions, but TD is usually faster in practice!

**Optimality**:

**Control**: same idea as DP and MC: generalised policy iteration (GPI)

- again we have the exploration-exploitation dilemma
  - $\epsilon$-greedy policies
- and again we consider on-policy and off-policy
  - on-policy TD control: SARSA
  - off-policy TD control: Q-learning
- instead of only state value functions $V(S_t)$, need to reason with state-action pairs $Q(S_t, A_t)$

### 6.2. Sarsa: On-policy TD Control

On-policy TD control: learn $q_\pi$ and improve $\pi$ while following $\pi$

Sarsa updates: $Q(S_t, A_t) \larr Q(S_t, A_t) + \alpha [ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) ]$

- If $S_{t+1}$ is a terminal state, $Q(S_{t+1}, A_{t+1}) = 0$
- Ensure exploration by using $\epsilon$-soft policy $\pi$

Ingredients for control:

- Evaluation: TD evaluation of $Q_\pi(S_t, A_t)$
- Improvement: greedy policy $\pi'$ w.r.t. $Q_\pi(S_t, A_t)$
- Exploration / exploitation: $\epsilon$-greedy policy $\pi$

Convergence to $\pi_*$ with probability $1$ if

- all $(s, a)$ infinitely visited and standard $\alpha$-reduction, i.e. $\forall s, a: \sum_{t: S_t = s, A_t = a}{\alpha_t \rarr \infty}, \sum_{t: S_t = s, A_t = a}{\alpha_t^2 < \infty}$, and
- $\epsilon$ gradually goes to zero. (TODO: why?)

Sarsa learns a Q for the current (eps-greedy) policy The policy converges to the optimal (eps-greedy) policy. Can we learn directly the value of the optimal policy (while still exploring)?

### 6.3. Q-learning: Off-policy TD Control

On-policy TD control: learn $q_\pi$ and improve $\pi$ while following $\mu$

Q-learning updates: $Q(S_t, A_t) \larr Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a{Q(S_{t+1}, a)} - Q(S_t, A_t)]$

Difference:

- Sarsa: $\epsilon$-greedy $\hat{a} \sim \pi(a \mid s)$
- Q-Learning: $\hat{a} = \arg\max_a{Q(S_{t+1}, a)}$

$$Q(S_t, A_t) \larr Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, \hat{a}) - Q(S_t, A_t)]$$

Exploration: policy ??

Convergence to $\pi_*$ with probability $1$ if all $(s, a)$ infinitely visited and standard $\alpha$-reduction.

Why is there no importance sampling ratio? Because $a$ in $q_\pi(s, a)$ is no random variable. (Recall: for $q_\pi$, ratio defined as $\prod_{k=t+1}^{T-1}{\frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)}}$)

### 6.4. Expected Sarsa

$$Q(S_t, A_t) \larr Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \sum_a{\pi(a \mid s) Q(S_{t+1}, a)} - Q(S_t, A_t)]$$

### 6.5. Maximisation Bias and Double Learning

Double Q-Learning:

- store different functions $Q_1, Q_2$.
- at each update, select

  - one of the two for determining the maximum (e.g. Q1)
    $$A^* = \arg\max_a{Q_1(s, a)}$$
  - one for update (e.g. Q2)
    $$Q(s, a) = Q_2(s, A^*) = Q_2(s, \arg\max_a{Q_1(s, a)})$$
  - their roles can be alternated (e.g. half-half of the time)

    $$Q_1(S_t, A_t) \larr Q_1(S_t, A_t) + \alpha [R_{t+1} + \gamma Q_2(S_{t+1}, \arg\max_a{Q_2(S_{t+1}, a)}) - Q_1(S_t, A_t)]$$

    $$Q_2(S_t, A_t) \larr Q_2(S_t, A_t) + \alpha [R_{t+1} + \gamma Q_2(S_{t+1}, \arg\max_a{Q_1(S_{t+1}, a)}) - Q_2(S_t, A_t)]$$

### 6.6. Games, Afterstates, and Other Special Cases

## 7. n-Step bootstrapping

Different techniques focus on different problems. And show different advantages and disadvantages.

- MDPs: dynamics defined by $p(s', r \mid s, a)$
- Stochastic processes: interaction with the environment gives rise to a trajectory of reward-state-actions $S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, \dots$
  - Each of these variables are random variables.
  - The overall trajectory instantiates a **stochastic process** $S_0 \rightsquigarrow \pi(a \mid s) \rightsquigarrow A_0 \rightsquigarrow p(a', r \mid s, a) \rightsquigarrow R_1, S_1 \rightsquigarrow \pi(a \mid s) \dots$

When interacting with the environment, specific realisations of these random variables are observed. Any path from the root is a **sequential process**, a realisation of the trajectory. The **return** is the cumulative future (discounted) reward $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$. It is a function of the trajectory, i.e. it depends on the actual realisation of the trajectory. That is, $G_t$ is uncertain, it is a random variable. So, we reason about our expectation on this variable. The value of a state is the expectation of the return that can be collected from the state onward, $v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]$.

The goal is to change $\pi(a \mid s)$ to give maximum probability to the high reward branches. Computing the expected return for all the branches can be expensive and / or impossible because we do not know $p(s', r \mid s, a)$. In both cases, we use approximate techniques.

- DP uses the recursive structure to define an iterative algorithm of single layers on the tree.
- TD(0) samples just one iteration, it uses 1-step return:
  $$G_{t:t+1} \doteq R_{t+1} + \gamma V_t(S_{t+1})$$
  $$V(S_t) \larr V(S_t) + \alpha (\textcolor{orange}{R_{t+1}} + \gamma V(S_{t+1}) - V(S_t))$$
- MC samples multiple times entire trajectories all the way to terminal states, it uses full return from the entire episode:
  $$G_{t:\infty} \doteq \sum_{k=1}^\infty{\gamma^{k-1} R_{t+k}}$$
  $$V(S_t) \larr V(S_t) + \alpha (\textcolor{orange}{R_{t+1} + \dots + \gamma^{T-t+1} R_{t+T}} - V(S_t))$$

Can we unify the different viewpoints to get the best of the single approaches?

- n-step return uses n-step return as target:

  $$G_{t:t+n} \doteq \sum_{k=1}^n{\gamma^{k-1} R_{t+k}} + \gamma^n V_{t+n-1}(S_{t+n})$$

  $$V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha [G_{t:t+n} - V_{t+n-1}(S_t)]$$

**On/Off-Policy Learning with n-Step Returns**:

Can similarly define n-step TD policy learning:

$$G_{t:t+n} = \sum_{k=1}^n{\gamma^{k-1} R_{t+k}} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n})$$

$$Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha \rho_{t+1:t+n}[G_{t:t+n} - Q_{t+n-1}(S_t, A_t)]$$

with importance ratio $\rho_{t:h} \doteq \prod_{k=t}^{\min(h, T-1)}{\frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)}}$

## 8. Planning and learning with tabular methods

Different techniques focus on different problems. And show different advantages and disadvantages. Can we unify the different viewpoints to get the best of the single approaches?

- Integrate Learning and Planning
- Understand different uses of models
- Planning in heuristic search and rollout algorithms

**Planning**: any process which uses a _model_ of the environment to compute a _plan_ of action (policy) to achieve a specified _goal_. Dynamic programming is planning: uses a model $p(s', r \mid s, a)$.

**Model**: anything the agent can use to predict how environment will respond to actions.

- **Distribution model** (MDPs): description of all possibilities and their probabilities, i.e. maximum information of the environment
  - can always sample from a distribution model
  - harder to model
  - more error prone
    $$p(s', r \mid s, a) \, \forall \, s, a, s', r$$
  - Example of queries possible: "What is the distribution of walking times from the office to the car park?"
- **Simulation (sample) model** (MC): produces sample outcomes, i.e. a less immediate view of the behaviour of the environment; but easier to model
  $$(s', r) \sim \hat{p}(s, a) \text{ s.t. } Pr\{ \hat{p}(s, a) = (s', r) \} = p(s', r \mid s, a)$$
  - Example of queries possible: "I'm in the office. Give me some examples of where I can go by walking and how much time it takes."

### 8.1. Dyna: integrated planning, acting and learning

Planning $\neq$ learning:

- two model-agent loops
- inner loop: planning
  - agent interacts with its internal model of the environment
  - simulated experience
- outer loop: learning
  - agent interacts with the environment
  - real experience

But planning $\sim$ learning: techniques to use experience are similar - there is a common property of using experience to "build" (learn) value functions.

The two loops are not mutually exclusive; learning and planning can happen at the same time. **Dyna** is a framework for learning and planning (direct RL, model learning, planning), with two roles for _experience_:

- improving the model through learning and indirectly improving the policy through planning
- directly improving the policy

Paths to a policy:

- Model-free RL: $\text{Environmental interaction} \rarr \text{Experience} \rarr \text{Direct RL methods} \rarr \text{Value function} \rarr \text{Policy}$.
- Model-based RL: $\text{Environmental interaction} \rarr \text{Experience} \lrarr \text{Model learning via simulation} \lrarr \text{Model} \rarr \text{Direct planning} \rarr \text{Value function} \rarr \text{Policy}$.

Considerations:

- Models can provide additional information and thus increase efficiency and robustness.
- Models can be costly to obtain, to run, and to keep updated.
- Model-free approaches appear more interesting as they are more challenging, in particular when model learning is included.
- Both model-free and model-based approaches can have biases.

Dyna-Q:

- Dyna-Q - Model Learning:
  - Simply store previous interactions
  - Suppose deterministic environment
    - For every, SARS interaction: Map $S, A$ -> $R, S$
- Dyna-Q - Planning
- For every real interaction with the environment, assign a budget of $n$ simulated interactions with the internal model.

A model is an internal belief of the agent about how the world should work. If the world changes, the agent should take measures to put its belief into question:

- Computational curiosity
- Linked to exploration

Dyna-Q uses an _exploration bonus_ heuristic:

- Keeps track of time since each state-action pair was tried in real environment.
- Bonus reward is added for transitions caused by state-action pairs related to how long ago they were tried: $R + \kappa \sqrt{\tau}$.
- Incentive to re-visit “old” state-action pairs.

Dyna-Q uses model to reuse past experiences. **Rollout planning**:

- Use model to simulate (“rollout”) _future_ trajectories
- Each trajectory starts at current state $S_t$.
- Find best action $A_t$ for state $S_t$.

Rollout Planning Optimality:

- If model is _correct_ and under Q-learning conditions (all $(s, a)$ infinitely visited and standard $\alpha$-reduction), rollout planning learns _optimal_ policy.
- If model is incorrect, learned policy likely sub-optimal on real task. Can range from slightly sub-optimal to failing to solve real task.

Can we use rewards from rollouts more effectively? Backpropagate rewards!

### 8.2. Real time dynamic programming

### 8.3. Monte-Carlo tree search

General, efficient rollout planning with backward updating.

1. Selection
2. Expansion
3. Simulation
4. Backpropagation

Stores _partial_ $Q$ as recursive tree and _asymmetrically expands_ tree based on most promising actions: $Q(s, a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'}{Q(S_{t+1}, a')} \mid S_t = a, A_t = a]$

**Upper Confidence Bounds for Trees (UCT)** is a popular MCTS variant that is easy to use and often effective. It uses UCB action selection as the tree policy, with $\alpha = 1 / N(S, a)$ (where $N(S, a)$ is the number of times action $a$ was selected in state $S$).

Imagine you are given an MDP for a chess game against a specific opponent.

- Dyna-Q and dynamic programming are
  suitable for **offline planning**:
  - Use MDP to find best policy _before_ the actual chess game takes place (offline)
  - Use as much time as needed to find policy
  - Policy is _complete_: gives optimal action for all possible states
- Rollout planning (including MCTS) is
  suitable for **online planning**:
  - Use MDP to find best policy _during_ the actual chess game (online)
  - Limited compute time budget at each state (e.g. seconds/minutes in chess)
  - Policy usually incomplete: gives optimal action for current state

## 9. Approximate methods

Theory so far has assumed:

- Unlimited space: can store value function as table
- Unlimited data: many (infinite) visits to all state-action pairs

Curse of dimensionality: in practice, the number of states grows exponentially with number of state variables: if state described by $k$ variables with values in $\{1, \dots, n\}$, then $\mathcal{O}(n^k)$ states.

- Not enough memory to store value function as table
  - Tabular methods require storage proportional to $\vert S \vert$ for $v(s)$ or $\vert S \vert \vert \mathcal{A} \vert$ for $q(s, a)$.
  - Need **compact representation of value functions** (But sometimes can be enough to store only partial value function; e.g. MCTS)
- No data (or not enough data) to estimate return in each state
  - Many states may never be visited
  - Need to **generalise observations** to unknown state-action pairs

### 9.1. Value function approximation

Replace tabular value function with parameterised function:

$$\hat{v}(s, \bm{w}) \approx v_\pi(s), \quad \hat{q}(s, a, \bm{w}) \approx q_\pi(s, a)$$

- $\bm{w} \in \mathbb{R}^d$ is parameter (weight) vector, e.g. linear function, neural network, regression tree
- **compact**: number of parameters $d$ much smaller than $\vert S \vert$
- **generalises**: changing one parameter value may change value estimate of many estimates / actions

Learning a value function is a form _supervised learning_: examples are pairs of states and return estimates , $(S_t, U_t)$, e.g.

- MC: $U_t = G_t$
- TD(0): $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \bm{w}_t)$
- n-step TD: $U_t = R_{t+1} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n \hat{v}(S_{t+n}, \bm{w}_{t+n-1})$

_Desired_ properties in supervised learning method:

- Incremental updates: update $\bm{w}$ using only partial data, e.g. most recent $(S_t , U_t)$ or batch
- Ability to handle noisy targets, e.g. different MC updates $G_t$ for same state $S_t$
- Ability to handle non-stationary targets, e.g. changing target policy, bootstrapping

If $\hat{v}$ or $\hat{q}$ differentiable, then stochastic gradient descent is a suitable approach:

- Let $J(\bm{w})$ be a differentiable function of $\bm{w}$.
- Gradient $\nabla J(\bm{w}) = \bigg( \frac{\partial J(\bm{w})}{\partial w_1}, \dots, \frac{\partial J(\bm{w})}{\partial w_d} \bigg)^T$
- $\bm{w}_{t+1} = \bm{w}_t - \frac{1}{2} \alpha \nabla J(\bm{w}_t)$
- Convergence requires standard $\alpha$-reduction.

**Objective**: find parameter vector $\bm{w}$ by minimising mean-squared error between approximate value $\hat{v}(s, \bm{w})$ and true value $v_\pi(s)$

$$J(\bm{w}) = \mathbb{E}_\pi\Big[ \big( v_\pi(s) - \hat{v}(s, \bm{w}) \big)^2 \Big]$$

- Gradient descent finds local minimum:
  $$\bm{w}_{t+1} = \bm{w}_t - \frac{1}{2} \alpha \nabla J(\bm{w}_t) = \bm{w}_t + \alpha \mathbb{E}_\pi\Big[ \big( v_\pi(s) - \hat{v}(s, \bm{w}_t) \big) \nabla \hat{v}(s, \bm{w}_t) \Big]$$
- Stochastic gradient descent samples the gradient:
  $$\bm{w}_{t+1} = \bm{w}_t + \alpha \big[ U_t - \hat{v}(S_t, \bm{w}_t) \big] \nabla \hat{v}(S_t, \bm{w}_t)$$
  - $\bm{w}_t$ will converge to local optimum under standard $\alpha$-reduction and if $U_t$ is an unbiased estimate $\mathbb{E}_\pi[U_t \mid S_t] = v_\pi(S_t)$
    - The MC update is unbiased, but the TD update is biased (TODO: why?)
  - Note this is not a true TD gradient because $U_t$ also depends on $\bm{w}$: $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \bm{w})$. Hence, we call it **semi-gradient TD**.

**Linear value function approximation**:

$$\hat{v}(s, \bm{w}) \doteq = \bm{w}^T \bm{x}(s) = \sum_{i=1}^d{w_i x_i(s)}$$

- $\bm{x}(s) = \begin{bmatrix} x_1(s), \dots, x_d(s) \end{bmatrix}^T$ is a feature vector of state $s$.
- Simple gradient: $\nabla \hat{v}(s, \bm{w}) = \begin{bmatrix} \frac{\partial \bm{w}^T \bm{x}}{\partial w_1}, \dots, \frac{\partial \bm{w}^T \bm{x}}{\partial w_d} \end{bmatrix}^T = \bm{x}(s)$.
- Gradient update: $\bm{w}_{t+1} = \bm{w}_t + \alpha \big[ U_t - \hat{v}(S_t, \bm{w}_t) \big] \bm{x}(S_t)$.

In the _linear case_, there is only _one optimum_!

- MC gradient updates converge to global optimum.
- TD gradient updates converge near global optimum (TD fixed point).

### 9.2. Gradient methods

### 9.3. On-policy and off-policy variants

## 10. Policy gradient methods

### 10.1. Policy approximation

Approximate Control in Episodic Tasks:

- Estimate state-action values: $\hat{q}(s, a, \bm{w}) \approx q_\pi(s, a)$.
- For linear approximation, features defined over states and actions: $\hat{q}(s, a, \bm{w}) \doteq \sum_{i=1}^d{w_i x_i(s, a)}$
- Stochastic gradient descent: $\bm{w}_{t+1} = \bm{w}_t + \alpha \big[ U_t - \hat{q}(S_t, A_t, \bm{w}_t) \big] \nabla \hat{q}(S_t, A_t, \bm{w}_t)$, e.g.
  - Sarsa: $U_t = R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \bm{w}_t)$
  - Q-learning: $U_t = R_{t+1} + \gamma \max_a{\hat{q}(S_{t+1}, a, \bm{w}_t)}$
  - Expected Sarsa: $U_t = R_{t+1} + \gamma \sum_a{ \pi(a \mid S_{t+1}) \hat{q}(S_{t+1}, a, \bm{w}_t) }$

Convergence to Global Optimum in Episodic Control:

| Algorithm                         | Tabular | Linear  | Nonlinear |
| --------------------------------- | ------- | ------- | --------- |
| MC control                        | yes     | chatter | no        |
| (semi-gradient) n-step Sarsa      | yes     | chatter | no        |
| (semi-gradient) n-step Q-learning | yes     | no      | no        |

Chatters near optimal solution because optimal policy may not be representable under value function approximation.

**Deadly triad**: Risk of divergence arises when the following three are combined:

1. Function approximation
2. Bootstrapping
3. Off-policy learning

Possible fixes:

- Use importance sampling to warp off-policy distribution into on-policy distribution
- Use gradient TD methods which follow true gradient of projected Bellman error (see book, p. 266)

### 10.2. Policy gradients

There are three kinds of RL algorithms:

- Policy-based ($\pi = \argmax_a{Q(s, a)}$): learn policy $\pi$ (e.g. gradient-based optimisation $\theta_{t+1} = \theta_t + \alpha \widehat{ \nabla J(\theta_t) }$) directly! Use it to act.
  - Softmax policy: $\pi(a \mid s, \theta) = \frac{e^{h(s, a, \theta)}}{\sum_{b \in \mathcal{A}}{e^{h(s, b, \theta)}}}$
  - Gaussian policy: $\pi(a \mid s, \theta) \sim \mathcal{N}\big( \mu(s, \theta), \sigma^2 \big)$
- Value-based ($Q$): Learn value, use to get policy.
- Model-based ($\hat{p}, \hat{r}$): Learn model, then _plan_ to get policy.

TODO: What is one advantage or disadvantage of any of the above three classes?

TODO: What is one setting you can think of where we should clearly use one of the above over the other two?

Definition: **Policy Optimisation Problem**:

- Given: $\pi(a \mid s, \theta)$, interaction with MDP $m$
- Find: optimal choice of $\theta$
- How to measure the quality of a given $\theta$? Episodic $J(\theta) = v_{\pi_\theta}(s_0)$

**Policy Gradient Theorem**: For any differentiable policy $\pi$, the policy gradient is $\nabla J(\theta) = \sum_s{d_\pi(s)} \sum_a{q_\pi(s, a) \nabla \pi(a \mid s, \theta)}$. $d_\pi(s)$ is the on-policy distribution under $\pi$:

- For start-state value: $d_\pi(s) = \sum_{t=0}^\infty{\gamma^t Pr\{ S_t = s \mid s_0, \pi \}}$
- For average reward: $d_\pi(s) = \lim_{t \rarr \infty}{Pr\{ S_t = s \mid \pi \}}$ (steady-state dist.)

Note: does not require derivative of environment dynamics $p(s', r \mid s, a)$.

Derivation:

$$
\begin{align*}
  \nabla J(\theta) & = \sum_s{d_\pi(s)} \sum_a{q_\pi(s, a) \nabla \pi(a \mid s, \theta)} \\
  & = \mathbb{E}_\pi\bigg[ \sum_a{q_\pi(S_t, a) \nabla \pi(a \mid S_t, \theta)} \bigg] \\
  & = \mathbb{E}_\pi\bigg[ \sum_a{\pi(a \mid S_t, \theta) q_\pi(S_t, a) \frac{\nabla \pi(a \mid S_t, \theta)}{\pi(a \mid S_t, \theta)}} \bigg] \\
  & = \mathbb{E}_\pi\bigg[ q_\pi(S_t, A_t) \frac{\nabla \pi(A_t \mid S_t, \theta)}{\pi(A_t \mid S_t, \theta)} \bigg] \\
  & = \mathbb{E}_\pi[ q_\pi(S_t, A_t) \nabla \ln{\pi(A_t \mid S_t, \theta)} ]
\end{align*}
$$

Policy gradient algorithms - general form: initialise $\theta_0$; for $t = 0, 1, \dots$, collect data using $\pi_{\theta_t}$ and iterate: $\theta_{t+1} = \theta_t + \alpha\big( q_\pi(S_t, A_t) \nabla \ln{\pi(A_t \mid S_t, \theta_t)} \big)$.

Need to approximate

- $q_\pi(S_t, A_t)$: Monte Carlo estimate of $G_t$ since $\mathbb{E}_\pi[G_t \mid S_t, A_t] = q_\pi(S_t, A_t)$
  - REINFORCE algorithm
- $\nabla \ln{\pi(A_t \mid S_t, \theta_t)}$
  - softmax $= x(s, a) - \sum_{a'}{\pi(a' \mid s, \theta) x(s, a')}$
  - Gaussian $= \frac{(a - \mu(s, \theta)) x(s)}{\sigma^2}$
  - Actor-critic algorithm

#### 10.2.1. What to do in continuous action spaces

#### 10.2.2. How probabilistic policies allow to apply the gradient method directly in the policy network

#### 10.2.3. The REINFORCE algorithm

Episodic algorithm

Mitigation for high variance: add a baseline (typically $\textcolor{orange}{b(S_t)} = \hat{v}(S_t)$) to reduce variance (note: baseline does not change expectation!)

$$\theta_{t+1} = \theta_t + \alpha\big( (q_\pi(S_t, A_t) - \textcolor{orange}{b(S_t)}) \nabla \ln{\pi(A_t \mid S_t, \theta_t)} \big)$$

#### 10.2.4. The Actor-Critic algorithms

$$\theta_{t+1} = \theta_t + \alpha\big( (\textcolor{orange}{R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)}) \nabla \ln{\pi(A_t \mid S_t, \theta_t)} \big)$$

- Actor ($\theta$): $\pi(A_t \mid S_t, \theta)$
- Critic: $\hat{v}(S_t, w)$

**Actor-Critic w/ TD(0)**.

RL Algorithms: Three Kinds

- Model-free:
  - Policy-based ($\pi$; Actor): learn policy, use to act
  - Value-based ($Q$; Critic): learn value, use to get policy
- Model-based ($\hat{p}, \hat{r}$): learn model, then _plan_ to get policy

#### 10.2.5. State-of-the-art algorithms in continuous action spaces: DDPG, TD3 and SAC

### 10.3. Actor Critic

## 11. Contemporary topics

### 11.1. [Deep Reinforcement learning](https://cs224r.stanford.edu/)

#### 11.1.1. Dealing with the deadly triad with the [DQN algorithm (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)

Deadly triad:

1. Function approximation
2. Bootstrapping
3. Off-policy learning

Recap: linear value function approximation:

- Linear representation of value: $\hat{v}(s, w) = w^T x(s) = \sum_{i=1}^d{w_i x_i(s)}$
- Gradient-based update: $w_{t+1} = w_t + \alpha [U_t - \hat{v}(S_t, w_t)] x(S_t)$
- State features: $x(s) = \langle x_1(s), x_2(s), \dots, x_d(s) \rangle$

Issue: Linearity! Linear is simple. But limited.

**Theorem**. Neural Networks are Universal Approximators (Informal): For any continuous function on the reals, $f: \mathbb{R} \rarr \mathbb{R}$, there exists a neural network that approximates that function.

Deep RL - The Algorithmic Journey:

- DQN (Mnih et al., 2013, 2015)
- TRPO (Schulman et al., 2015)
- Double Q-Learning (Van Hasselt, 2016)
- AlphaGo (Silver et al., 2016)
- PPO (Schulman et al., 2017)
- Rainbow (Hessel et al., 2018)
- DDPG (Lillicrap et al., 2018)
- Soft Actor-Critic (Harrnoja et al., 2018)

TODO: Given an RGB image of breakout as state, does the Markov property hold?

#### 11.1.2. [Application to the Atari games case (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)

#### 11.1.3. Evolutions of the DQN algorithm: Double DQN, Prioritized Experience Replay, multi-step learning and Distributional value functions

DQN Limitations:

1. Minimal exploration
2. Troubles with delayed, sparse reward
3. Only applicable to discrete action space
4. No convergence guarantees
5. Highly sensitive to hyper parameters

#### 11.1.4. Rainbow: the state-of-the-art algorithm in discrete action space

#### 11.1.5. Trust Region Policy Optimisation (TRPO)

- Actor ($\theta$): $\pi(A_t \mid S_t, \theta)$
- Critic ($w$): $\hat{v}(S_t, w)$

Problem: Big updates to $\pi_i$ yield instability!

TRPO main idea: Keep updates in “Trust Region”!

- Advantages:
  - Works with continuous action spaces
  - Trust region can yield stability
- Disadvantage:
  - Trust region implementation uses KL divergence:
    $$\mathbb{E}_{s \sim \pi_{\theta_t}} D_\text{KL}\big( \pi_{\theta_{t+1}}(\cdot \mid s) \Vert \pi_{\theta_t}(\cdot \mid s) \big) \leq \beta$$

#### 11.1.6. Proximal Policy Optimisation (PPO)

Uses **policy ratio** instead:

$$\frac{\pi_{\theta_{t+1}}(a \mid s)}{\pi_{\theta_t}(a \mid s)}$$

### 11.2. Multi-agent reinforcement learning (MARL)

A multi-agent system consists of:

- Environment: The environment is a physical or virtual world whose state evolves and is influenced by the agents' actions within the environment.
- Agents: An agent is an entity which receives information about the state of the environment and can choose actions to influence the state. => Agents are goal-directed, e.g. maximizing returns.

Applications: computer games, autonomous driving, multi-robot warehouses, automated trading.

Example: Level-Based Foraging: three agents (robots) with varying skill levels and shared goal to collect all items.

Goal: learn optimal policies for two or more agents interacting in a shared environment.

- Each agent chooses an action based on its policy ⇒ joint action
- Joint action affects environment state; agents get rewards + new observations

Reasons for MARL vs. SARL:

- Decomposing a large problem: train separate agent policies (more tractable than learning a single policy to control all agents)
- Decentralised decision making: There are many real-world scenarios where it is required for each agent to make decisions independently, e.g. autonomous driving is impractical for frequent long-distance data exchanges between a central agent and the vehicle.

New challenges arise in MARL:

- **Non-stationarity** caused by multiple learning agents
  - If multiple agents are learning, the environment becomes non-stationary from the perspective of individual agents.
  - Moving target: each agent is optimizing against changing policies of other agents.
- Optimality of policies and equilibrium selection
- **Multi-agent credit assignment**: which agent's actions contributed to received rewards?
- Scaling in number of agents

#### 11.2.1. Learning of behaviours in environments with several agents

#### 11.2.2. Learning of cooperative behaviours, Learning of competitive behaviours, and mixed cases

#### 11.2.3. State-of-the art algorithms

#### 11.2.4. The special case of games: The Alpha-Go case and the extension to Alpha-Zero

Hierarchy of games:

- Partially observable stochastic games: $n$ agents, $m$ (partially observed) states.
  - Stochastic games: $n$ agents, $m$ (fully observed) states
    - Repeated Normal-Form games: $n$ agents, $1$ state
    - Markov Decision Processes: $1$ agent, $m$ states

Normal-form games define a _single interaction_ between two or more agents, providing a simple kernel for more general games to build upon. Normal-form games are defined as a 3 tuple $(I, \{A_i\}_{i \in I} \{R_i\}_{i \in I})$:

- $I$ is a finite set of agents $I = \{1, \dots, n\}$
- For each agent $i \in I$:
  - $A_i$ is a finite set of actions.
  - $R_i$ is the reward function $R_i: A \rarr \mathbb{R}$ where $A = A_1 \times \dots \times A_n$ (set of **joint** actions).

In a normal-form game, there are _no time steps or states_. Agents choose an action and observe a reward.

The game proceeds as follows:

1. Each agent samples an action $a_i \in A_i$ with probability $\pi_i(a_i)$.
2. The resulting actions from all agents form a **joint action**, $a = (a_1, \dots, a_n)$.
3. Each agent receives a reward based on its **individual reward function** and the **joint
   action**, $r_i = \mathcal{R}_i(a)$.

**Classes of games**: Games can be classified based on the relationship between the agents' reward functions.

- In **zero-sum games**, the sum of the agents' reward is always zero, i.e. $\sum_{i \in I}{\mathcal{R}_i(a)} = 0, \forall a \in A$.
- In **common-reward games**, all agents receive the same reward $(R_i = R_j; \forall i, j \in I)$.
- In **general-sum games**, there are no restrictions on the relationship between reward functions.

Normal-from games with $2$ agents are also called **matrix games** because they can be represented using reward matrices.

**Repeated Normal-Form Games**: To extend normal-form games to _sequential_ multi-agent interaction, we can repeat the same game over $T$ timesteps.

- At each time step $t$ an agent $i$ samples an action
  $a_i^t$.
- The policy is now conditioned on a **joint-action**
  history $\pi_i(a_i^t \mid h^t)$ where $h^t = (a^0, \dots, a^{t-1})$.
- In special cases, $h^t$ contains n last joint actions. E.g. in a tit-for-tat strategy (Axelrod and
  Hamilton 1981), the policy is conditioned on $a^{t−1}$.

**Stochastic games**: e.g. level-based foraging

- Each state can be viewed as a non-repeated normal-form game
- Stochastic games can also be classified into: zero-sum, common-reward or general-sum
- The figure on the left shows a general-sum case

**Partially Observable Stochastic Games (POSG)**:

At the top of the game model hierarchy, the most _general_ model is the POSG

- POSGs represent complex decision processes with _incomplete information_
- Unlike in stochastic games, agents receive observations providing incomplete information about the state and agents’ actions
- POSGs apply to scenarios where agents have limited sensing capabilities, e.g. autonomous driving, strategic games (e.g. card games) with private, hidden information

The Observation Function: POSG can represent diverse observability conditions. For example:

- modeling noise by adding uncertainty in the possible observation
- to limit the visibility region of agents (see LBF example)

**Solution Concepts for Games**: MARL problem = game model (e.g. normal-form game, stochastic game, POSG) + solution concept (e.g. Nash equilibrium, social welfare)

Note: We must consider the _joint policy_ of all agents.

**Nash equilibrium**: mutual best response to general-sum games with two or more agents.

- No agent $i$ can improve its expected returns by changing its policy $\pi_i$ assuming other agents' policies remain fixed:
  $$\forall i, \pi_i' : U_i(\pi_i', \pi_{-i}) \leq U_i(\pi)$$
- Each agent's policy is a _best response_ to all other agents' policies.

Equilibrium solutions are standard solution concepts in MARL, but have _limitations_:

- **Sub-optimality**:
  - Nash equilibria do not always maximize expected returns
  - E.g. in Prisoner’s Dilemma, $(D,D)$ is Nash but $(C,C)$ yields higher returns
- **Non-uniqueness**:
  - There can be multiple (even infinitely many) equilibria, each with different expected returns
- **Incompleteness**:
  - Equilibria for sequential games don’t specify actions for _off-equilibrium paths_, i.e. paths not specified by equilibrium policy $Pr(\hat{h} \mid \pi) = 0$
  - If there is a temporary disturbance that leads to an off-equilibrium path, the equilibrium policy $\pi$ does not specify actions to return to a _on-equilibrium_ path

To address some of these limitations, we can add additional solution requirements such as **Pareto optimality**. A joint policy $\pi$ is Pareto-optimal if it is not Pareto-dominated by any other joint policy. A joint policy $\pi$ is Pareto-dominated by another policy $\pi'$ if $\forall i: U_i(\pi') \geq U_i(\pi) \text{ and } \exist i: U_i(\pi') > U_i(\pi)$.

A joint policy is Pareto-optimal if there is no other joint policy that improves the expected return for at least one agent without reducing the expected return for any other agent.

To further constrain the space of desirable solutions, we can consider social welfare and fairness concepts.

- A joint policy $\pi$ is **welfare-optimal** if $\pi \in \argmax_{\pi'}{W(\pi')}$ where welfare $W(\pi) = \sum_{i \in I}{U_i(\pi)}$.
- A joint policy $\pi$ is **fairness-optimal** if $\pi \in \argmax_{\pi'}{F(\pi')}$ where welfare $F(\pi) = \prod_{i \in I}{U_i(\pi), \, U_i(\pi) > 0 \, \forall i}$.

**Agent modelling & best response**: Game theory solutions are normative: they prescribe how agents should behave, e.g. minimax assumes worst-case opponent. What if agents don’t behave as prescribed by solution? e.g. minimax-Q was unable to exploit hand-built opponent in soccer example. Other approach: agent modeling with best response

- Learn models of other agents to predict their actions
- Compute optimal action (best response) against agent models

**The Multi-Agent Policy-Gradient Theorem**: In MARL, the expected returns of agent $i$ under its policy $\pi_i$ depends on the policies of all other agents $\pi_{-i}$ -> the multi-agent policy gradient theorem defines an expectation over the policies of all agents ($h_i$: individual observation history).

**Self-Play Monte Carlo Tree Search**: In zero-sum games with symmetrical roles and egocentric observations, agents can use the same policy to control both players -> learn a policy in _self-play_.

**Population-based training** is a generalisation of self-play to general-sum games:

- Maintain a _population of policies_ representing possible strategies of the agent
- Evolve populations so they become more effective against the populations of other agents
- We denote the population of policies for agent $i$ at generation $k$ as $\Pi_i^k$.

### 11.3. Shielding and safe reinforcement learning

### 11.4. Relational reinforcement learning and traditional planning

### 11.5. Towards life-long learning in agents

- Is RL a way to obtain a General Artificial Intelligence?
- Multi-task learning in RL, Transfer learning in RL and Meta-learning in RL.

## 12. Psychology

## 13. Neuroscience

## 14. Applications in game playing and beyond

## 15. Frontiers

### Beyond the Markov Property

Markov property holds for:

- RL problem (MDPs): learn to maximise reward by interaction.
- Planning problem (MDPs): compute the optimal policy given complete knowledge of $(\mathcal{S}, \mathcal{A}, p)$.

Does it hold for all RL problems?

#### POMDP: Partially Observable MDPs

- $\mathcal{S}$: a set of states.
- $\mathcal{A}$: a set of actions.
- $\mathcal{O}$: a set of _observations_.
- $p(s' \mid s, a)$: state transition function.
- $r(s, a)$: reward function.
- $\omega(o \mid s, a)$: _observation function_.

Definitions:

- RL problem (POMDPs):
  - Given: $(\mathcal{S}, \mathcal{A}, \mathcal{O})$, query access to $e$.
  - Repeat for $t = 0, 1, \dots$:
    1. Agent selects action $A_t \in \mathcal{A}$.
    2. Agent observes $O_t, R_t \sim e(o, r \mid s, a)$.
  - Goal: maximise total reward.
- Planning problem (POMDPs):
  - Given: a POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{O}, p, r, \omega)$.
  - Output: an optimal policy $\pi$.

Uncertainty about state!

Theorem. Planning in infinite horizon POMDPs is undecidable.

Theorem. Planning in finite horizon POMDPs, is PSPACE-complete.

Approximate learning in POMDPs:

- Belief state complexity can be independent of the environment.
- Value function can be function of _belief state_:
  $$b(s) = p(s \mid o, a, b')$$

Theorem. Weakly Revealing POMDPs: Learning difficulty in POMDPs scales as a function of $\beta$.

#### AIXI

No constraints on agent.

#### The Big World Hypothesis

"In many decision-making problems the agent is orders of magnitude smaller than the environment” - Javeed, Sutton (2024)

- Bounded rationality.
- All models are wrong, some are useful.
- Open-endedness.

### Rewards: RL and the Brain

"Part of the appeal of reinforcement learning is that it is in a sense the whole AI problem in a microcosm." - Sutton, 1992

**The Reward Hypothesis**: "...all of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward)" - Sutton (2004), Littman (2017)

**The Reward Is Enough Hypothesis**: "Intelligence, and its associated abilities, can be understood as subserving the maximisation of reward by an agent acting in its environment" - Silver, Singh, Precup, Sutton (2021)

Formalising the reward hypothesis in finite MDPs:

- Expression Question: Which signal can be used as a mechanism for expressing a given task?
  - The Reward Hypothesis (formalised): Given any task $\mathcal{T}$ and any environment $E$ there is a reward function that realises $\mathcal{T}$ in $E$.
    - Assumption: All environments are finite Controlled Markov Processes (CMPs): $E = (\mathcal{S}, \mathcal{A}, T, \gamma, s_0)$.
- Task Question: What is a task? $\mathcal{T} \in \{ \Pi_G, L_\Pi, L_{\tau, N} \}$
  - Set of acceptable policies (SOAP): $\Pi_G \subseteq \Pi$
    - Example: Reach the goal in less than 10 steps in expectation.
  - Policy ordering (PO): $L_\Pi$
    - Example: I prefer you reach the goal in 5 steps, else within 10, else don't bother.
  - Trajectory ordering (TO): $L_{\tau, N}$
    - Example: I prefer safely reaching the goal and avoiding lava at all costs.

**Markov Reward Is Limited**:

Theorem: For each of SOAP, PO, and TO, there exist $(E, \mathcal{T})$ pairs for which no reward function realises $\mathcal{T}$ in $E$.

What Kind of SOAPs Cannot Be Expressed?

- "always go in the same direction"
- XOR gate

Theorem: The Reward Design problem can be solved in polynomial time, for any finite $E$, and any $\mathcal{T}$.

### Inverse RL

- Given: An environment and behaviour.
- Output: A reward function that _explains_ the behaviour.

_If we can observe what an agent does, can we infer what reward function it maximizes?_

- Given: An controlled Markov process, $(\mathcal{S}, \mathcal{A}, p)$, and policy $\pi$.
- Output: A _reward function_ that _explains_ the policy $\pi = \argmax_{\pi' \in \Pi}{v_r^{\pi'}(s_0)}$.

There is a fundamental limitation to Inverse RL: Every policy is optimal w.r.t. the zero reward function! (and constant)

- Solution 1 to unidentifiability: add a regulariser $\pi = \argmax_{\pi' \in \Pi}{v_r^{\pi'}(s_0)} + \omega(r)$
- Solution 2 to unidentifiability: intervention

### Reward Shaping

Theorem. A shaping function preserves the optimal policy if and only if it is a potential-based shaping function.

$$f(s, a, s') = \gamma \phi(s') - \phi(s)$$

### RLHF

1. Collect demonstration data, and train a supervised policy.
   - A prompt is sampled from our prompt dataset.
   - A labeller demonstrates the desired output behaviour.
   - This data is used to fine-tune GPT-3 with supervised learning.
2. Collect comparison data, and train a reward model.
   - A prompt and several model outputs are sampled.
   - A labeller ranks the outputs from best to worst.
   - This data is used to train our reward model.
3. Optimise a policy against the reward model using reinforcement learning.
   - A new prompt is sampled from the dataset.
   - The policy generates an output.
   - The reward model calculates a reward for the output.
   - The reward is used to update the policy using PPO.

## References

Abbeel, P. (2014). CS188 Intro to AI [Course materials]. UC Berkeley. Retrieved from <https://ai.berkeley.edu/home.html>

Finn, C. (2025). CS 224R Deep Reinforcement Learning [Course materials]. Stanford University. Retrieved from <http://cs224r.stanford.edu/>

Malan, D., & Yu, B. (2024). CS50's Introduction to Artificial Intelligence with Python [Course materials]. Harvard OpenCourseWare. Retrieved from <https://cs50.harvard.edu/ai/2024/notes/3/>

Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.

Multi-Agent Reinforcement Learning:
Foundations and Modern Approaches
by Stefano V. Albrecht, Filippos Christianos and
Lukas Schäfer
MIT Press, 2024
