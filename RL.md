# Learning note for RLDP

**Author**: Andrew Xu

## 1. Week one 

### Material

Introduction to Reinforcement Learning
Smoov & Curly's Bogus Journey

### HW1

rolling dice

### Learning

1. Q learning is based on Bellman equation

2. goal propagation is similar to spanning tree in the CN

3. In Q learning from ML4T, state is calcuated by indicators. But in this case, it is coupled with reward

4. V(0) is the expected value of the optimal policy

5. From state, value function, still need MDP to find next best action
In Q learning, tells best action to take without having to look at MDP

6. R can be different presentation/format. If R is R(s,a,s'), meaning it also depends on the next state, Q can be rewrite as 
Q= gama* sum (T(s,a,s')*(V(s,a,s')+R(s,a,s')))

## 2. Week two 

### Material

Lectures:

Reinforcement Learning Basics
TD and Friends
Readings:

Littman (1996) chapters 1-2
Sutton (1988)

### HW2

TD

### Learning

1. Q learning is one type of RL algorithm. There are others algo for different conditions. Like plan, conditional plan and stationary polcy

2. Estimate is V(0) after iteration/Kstep estimate

3. K step Estimate means use rewards before that step and only use state-value at K step

4. for the final Estimate, weight is equal to 1- sum of previous weight (for finite horizon)

5. Chapter 7 G is the equations that we use for this homework

6. In the TD formula's in the update rule, we calculate the error and we do subtract the old state value from the new state value. I was wondering why in the example's we didn't do that?
Because we are not calculating the values iteratively. In the lectures, we calculate the E from samples... so you approximate the value with alpha on every new sample. In the homework, you have the MDP. So you can calculate the values straight.

## 3. Week three 

### Material

Lectures:

Convergence
AAA
Readings:

Littman and Szepesvari (1996)
Littman (1994)

### P1

Sutton 1988

### Learning

1. Advantages for TD:easier to compute, they tend to make more efficient use of their experience

2. TD to predict arbitrary events, not lust goal-related ones.

3. In multi-step prediction problems, correctness
is not revealed until more than one step after the prediction is made,
but partial information relevant to its correctness is revealed at each step.

4. they converge more rapidly and make more accurate predictions along the way. TD
methods have this advantage whenever the data sequences have a certain statistical
structure that is ubiquitous in prediction problems.

5. by adjusting its evaluation of the novel state towards the bad state's evaluation,
rather than towards the actual outcome, the TD method makes better use of
the experience

6. TD methods
try to take advantage of the information provided by the temporal sequence
of states, whereas supervised-learning methods ignore it. It is possible for this
information to be misleading, but more often it should be helpful.

7. only required characteristic is that the system predicted be a
dynamical one, that it have a state which can be observed evolving over time.

8. where to initialize weight and delta weight is important

9. sum of lambda part is a vector, determine how far other states will make a difference. All 1 is equal to supervised-learning, only care about result/end state

10. final state is either 0 or 1. 

11. TD(1) is minimize RMSE in the trainingset based on prediction and outcome (same as supervised-learning), whereas TD(0) is maximum likelyhood based, considering overall steps

## 4. Week four

### Material
Lectures:

Messing with Rewards
Readings:

Ng, Harada, Russell (1999)
Asmuth, Littman, Zinkov (2008)
Upcoming due dates:


### HW3

MDP

### learning:

1. In this formulation, the
environment is assumed to be in one of a finite-set of
states, the decision-making agent has a choice of actions
in each state of the environment, executing an
action causes a stochastic change in the state of the
environment, and the agent receives a stochastic reward
in return for executing the action.

2."greedy" policy iteration (which
greedily accepts all single-state action changes that are
improvements)

3.The probability that the process moves into its new state {\displaystyle s'} s' is influenced by the chosen action. Specifically, it is given by the state transition function {\displaystyle P_{a}(s,s')} P_a(s,s'). Thus, the next state {\displaystyle s'} s' depends on the current state {\displaystyle s} s and the decision maker's action {\displaystyle a} a. But given {\displaystyle s} s and {\displaystyle a} a, it is conditionally independent of all previous states and actions; in other words, the state transitions of an MDP satisfies the Markov property. (https://en.wikipedia.org/wiki/Markov_decision_process)

4. possible ways to increase iteration numbers are: a) use different rewards like 1.01; b) increase (s,a) space 

5. accepting 
any non-zero number of single-state improvements
can only improve the policy, and the second
claims that there a! ways exists at least one single-state
improvement that improves the policy, unless the policy is already optimal. (this is very similar to traveler revision)

6.Greedy policy iteration is PI with select(T) = T,
namely, we perform all the possible single-state action
improvements at each policy improvement step

7. reward function help to give learner direction to final goal (mini reward)

8. reward shaping help to avoid suboptimal positive loop. 

9. use potential to keep track

10. Potential function doesn't change optimal polcy, probably help to converge faster