# Learning note for RLDP

**Author**: Andrew Xu

## Week 1

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

7. Markov Decision Process: state, model T, actions, rewards
Markovian property: only present matters,  stationary

8. rewards: encompasses our domain knowledge. So, the Rewards you get from the state tells you the usefulness of entering into that state. delayed rewards

9.credit assignment problem: for the given states you're in, what was the action you took that helped to deter, or actions you took that helped to determine the ultimate sequence of rewards that you saw. a sequence of events over time: temporal credit assignment problem.

## Week 2 

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

7. behavior structure: plan- fixed sequence of actions (during learning, stochasiticity), conditional plan - includes if statement, stationary policy/universal plan -  if at every state same, maping from state to action, very large
And there's always an optimal
stationary policy for any MDP.

8. model based: model learner <--> T (more supervised). Value function based: value update <--> Q, policy search: policy update <--> pi (more direct learning)

9. properties of learning rate in TD: sum is infinitity, sum of square less than infinitity
TD1 is same as outcome based update (if no repeated stats)
TD0 finds maximum likelyhood estimate if data repated infititly often

10. contraction mapping: Bellman operator, has a solution and unique, value iteration converges
max is non expansion
gernalized MDP: regular MDP, pessimistic MDP risk averse, exploration sensitive, maxmin- zero sum game 

11. evaluation a learner: value of returned policy (outcome), computational complexity (time). Experience complexity (time) how much data it requires

## Week 3

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

12. Cramer's rule: argmax Q is optimal
Gamma: horizon. Gamma small--> don't further look into the future 
Easy to optimize over short horizon
short term thinking

13. policy iteration: convergence is exact and complete in finite time; converges ate least as fast as VI
Domination: strict domination, epsilon-optimal
why PI works: B is value improvement (no local minima), monotonic, transitinity, fixed point, no lo

## Week 4

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

1. In this formulation, the environment is assumed to be in one of a finite-set of states, the decision-making agent has a choice of actions in each state of the environment, executing an action causes a stochastic change in the state of the environment, and the agent receives a stochastic reward in return for executing the action.

2."greedy" policy iteration (which greedily accepts all single-state action changes that are improvements)

3.The probability that the process moves into its new state {\displaystyle s'} s' is influenced by the chosen action. Specifically, it is given by the state transition function {\displaystyle P_{a}(s,s')} P_a(s,s'). Thus, the next state {\displaystyle s'} s' depends on the current state {\displaystyle s} s and the decision maker's action {\displaystyle a} a. But given {\displaystyle s} s and {\displaystyle a} a, it is conditionally independent of all previous states and actions; in other words, the state transitions of an MDP satisfies the Markov property. (https://en.wikipedia.org/wiki/Markov_decision_process)

4. possible ways to increase iteration numbers are: a) use different rewards like 1.01; b) increase (s,a) space 

5. accepting any non-zero number of single-state improvements can only improve the policy, and the second claims that there a! ways exists at least one single-state improvement that improves the policy, unless the policy is already optimal. (this is very similar to traveler revision)

6.Greedy policy iteration is PI with select(T) = T,namely, we perform all the possible single-state action improvements at each policy improvement step

7. reward function help to give learner direction to final goal (mini reward)

8. reward shaping help to avoid suboptimal positive loop. 

9. change reward without changing optimal: multiply by positive constant, shift by constant, non-linear potentioal based

10. Potential function doesn't change optimal polcy, probably help to converge faster; Q learning with potential is equal to Q learning with Q function initialization

11. PI iteration: solve equations at every step. More equations and less overalap with each other, slower the propagation rate. 



## Week 5

### Material
Lectures:

Exploring Exploration
Readings:

Fong (1995)
Li, Littman, Walsh (2008)
Sutton (1988)


### HW4
Taxi-v2


### learning:

1. General Rmax: combining sochastic and deterministic algo

2. When updating Q table, we need pay special atention to final state. V(s+1) = 0

3. Even take random action, Q can converge, but require large episodes to converge to right values

4. Need to tune rar, radr for every problem

5. Exploration: getting better at estimating your values, get more information soon so that you can learn entire q function; exploitation: narrow that information into getting short-term reward 


6. many exploration strategy: epsilon - greedy, boltzman, bayesian, optmistic

7. Q learning convergence: Every state-action pair has to be revisted often, learning rate needs to decay (common not to 0 for nonstationary world)

8. Sarsa is non policy reinforcement learning

9.Hoeffding theorem: 100 (1-theta) % confidence interval for miu is ... set bounds for bandits problem

10. general stochastic MDP: stochastic use Hoeffding bound until estimate are accurate; sequential for unknown state-action pairs assumed optimistic (use Maximum likelyhood estimate)
if all transition are either accurately estimate or unknown, optimial policy is either near optimal or an unknow state is reached quickly. 

11. KWIK: know what it knows. 



## Week 6

### Material
Lectures:

Generalization
Readings:

Gordon (1995)
Baird (1995)


### P2
Lunar Lander


### learning:

1. prediction: policy evaluation problem, doen't have to be optimal, e.g TD
Control: optimization, learn optimal values,  e.g. Q-learning and Sarsa

2. P2 tips: use carpool to debug algorithm, go one layer higher
openai observation space.shape -- input layer; env.action space.n --output layer

3. n contrast to the dropout layer, a dense layer is simply a layer where each unit or neuron is connected to each neuron in the next layer.

4. We can adjust parameters by rendering enviroment. For LL, if it doesn't drop, it means short sighted, need to adjust gamma

5. time and episode is very different

6. gamma is not hyper parameter like any other, property of enviroment

7. hover forever/get stuck at local optimal: how oftern update target network, gamma can also be issue, adam, huberloss. if see spike, then start increasing update frequency. implement soft update, can also be lack of eploration. target network, 25 time stampls. not recommend to restrict to 1000 steps.implement on cartpole in 20 episode 

8. advantage of dqn: it becomes more stable, whereas value function is super wild, have to do on-policy function. with linear function approximation there is guarantee for solution. policy gradient explore entire space, high randomness, high schoacity so it has lot of variants, although there is no bias. value function help to reduce variants, but add a bit bias

9. gamma determin planing horizon: translate into episodes and min. if it's very high, going to infinite, you are very slow to get correct value

10. each action gets a set of n weights to present the Q vlaues for that actions. weights give importance of all the features in contributing to an action value

## Week 7

### Material

Lectures:

Partially Observable MDPs

Readings:

Littman (2009)

### HW5
Bar Brawl

### learning

1. H space is much less required than memorization

2. If we don't know where we are or what to do, we can take actions to where we know: go left for 15 min, we know for sure we will hit a wall, and we can go from there

3.Neural networks have gained a lot of popularity in recent years due to the increase in available compute power and shear size of datasets available. Neural Networks are primarily used for classifying between different things (e.g. images) or regression (e.g. housing prices, or in our case approximating our Q-function) i.e. supervised learning although they can also be used for unsupervised learning (can't speak to that). Neural Networks are generally too high power for small datasets and usually simpler machine learning models are employed in this case for simplicity and to avoid generating too much variance in your model. When you have very high dimensional data like images or audio, neural networks are great as they have the ability to extract specific features from your data and learn based on those features through methods like convolutions, pooling, LSTMs, etc.


The general rule of thumb for tuning the number of hidden layers / number of nodes in each hidden layer is to follow the literature for that given area and use very similar architectures within that domain. A lot of the intuition behind why certain values work better than others is still unclear even to expert researchers. Once you've read enough research papers / Jupyter Notebooks, you will gain some intuition as to what range of values generally will work.  


For resources on learning more about neural networks / deep learning, I highly recommend Andrew Ng's Deep Learning courses on Coursera. Fast.ai also has a great course that uses PyTorch so you can get your hands dirty with a popular ML framework. 3Blue1Brown also has a slew of YouTube videos that thoroughly explain the mechanics of how neural networks work. Google is a lifesaver as well.

If you'd like to learn more I can recommend http://cs231n.stanford.edu/ and https://mitpress.mit.edu/books/deep-learning

4.where instructors collectively construct a single answer
I'm just wondering what to do after this?

Keep reading, start with Deepmind and OpenAI's work, take Berkeley's Deep RL course, read Sutton.

Are there other applications to Deep RL other than training agents to play video games? If we're interested in stuff like stock trading and recommender systems or computer vision, speech recognition then have these algos also been successfully tried there?

Yes, yes, yes (specifically I have seen RL used as an 'attention' system in computer vision suggesting where to look), not sure. See Deepmind's application of RL to data center power management. There have been many applications to systems for continuous control, Google Scholar is a great way to find this. I have heard of RL being used in finance but not surprisingly finance doesn't tend to publish their research. RL has long lived in the space of recommender systems, you can search for papers with many advertising applications. I haven't seen a speech recognition system (I haven't looked though) but I have seen a paper from Facebook applying RL to chat bots.

5.POMDP: a way of talking about non-markov enviroment. Z observables, O observation function, O(s,Z) probability of seeing z in state s --> MDP with infinite states (LP, PI, VI) use PWLC (piecewise linear and convex method)


6. model based RL --> learned model and use it (learn POMDP), model free RL --> don't bother (map observation to actions)

7. RL: expectation maximization, memoryless may be random.  
Bayesian RL: planning a kind of continious POMDP, piecewise polynomial and convex, prob distribution over Q-function

8. belief state in POMDP is distribution over states, state never observed. predictive state representation: predicive state is probability of future prediction

9.PSR theorem: any n-state POMDP can be represented by a PSR with no more than n tests, each of which is no longer than n steps long. 


## Week 8 

### Material

Lectures:

Options
Game Theory
Readings:

Sutton, Precup, Singh (1999)
Jong, Hester, Stone (2008) (including slides from resources link)

### HW6

Game thoery

### learning

1. Function approx helps to fill in the gap that we don't see those states very often

2. In temporal abstraction, instead of atomic action like MDP, we take variable time actions (more like real world)

3. We can turn non markovian process into markovian process by keeping track of history

4. 
The expected value defining R is over all sequences of k transitions starting from the state s. R, F depends on K, it matters how many steps you take

5. Other benefit: state abstraction -This abstraction allows learn more efficiently, ignore large spaces where states are not relevant; avoid exploration

6. Not sequencying things directly, but managing competing goals. Beta means either the probability that I have succeeded in executing that option or another goal becomes important, and I need to interrupt the current goal (greatest mass Q-learning, top Q-learning, negotiated W-learning

7. Mote carlo tree search: select, expand, simulate, backup. One step policy based upon what you have learned. useful for large states, need lots of samples to get good estimate. planning time independent, running time exponential in the horizon

8.In an MDP, each action takes a single unit of time to complete.
In an SMDP, actions can take varying amounts of time to complete, possibly even non-integer or randomly-distributed amounts of time. If you're doing any kind of time-based discounting of rewards (which is usually the case in Reinforcement Learning), you must take into account the amount of time that an action requires to complete to adequately update your value functions.

9.Strategy has to say what you would do in all states where you might end up.

10. In a 2 player, zero sum, deterministic, perfect information game, Minimax ===maximin, and there always exists an optimal pure strategy for each player. non-deterministic-->that other theorem still holds, hidden information --> minimax != maxmin,  mix strategy (probability distribution over strrategies). non-zero --> prinson dilemma

11. Nash equilirium： for each player, max utility. no motivation to change. 


## Week 9 

### Material

Lectures:

Game Theory Reloaded
Game Theory Revolutions

Readings:

Littman (1994)
Littman, Stone (2003)
Munoz de Cote, Littman (2008)

### P3 

Soccer game

### learning

1. IPD: iterated prisoner's dilemma. TFT:tit for tat, cooperate on first round, copy opponent's previous move thereafter

2. best response to a finite state strategy: our choice impacts our pay off and future decisions of the opponent. MDP, VI to solve, the matrix is all we need. depends on gamma (probability of play again)

3. TFT is mutual best response. it is nash. 

4.  in repeated games, the possiblity of retaliation opens the door for cooperation. folk theorem: describe the set of payoffs that can result from nash strategy in repeated games. 

5. Convex hull: feasible region from two player plot. minimax/security level profile: acceptable region (achieved by a player defending itself from malicious adversary)

6. any feasible payoff profile that strictly dominates the minimax profile can be realized as Nash equilirium payoff profile with suficiently large discout factor. 

7. grim trigger: also nash, but not subgame perfect (always best response independent of history). Same as TFT. Implausible threats
pavlov is subgame perfect: cooperate if agree, defect if disagree

8. zero-sum stochastic games, minimax Q- VI works， converges, unique solution, policies can be computed independently, update efficient, Q function sufficient to specify policy. For general sum, nash Q - VI doesn't work, doesn't converge, no unique solution, incompatible, not efficient update, Q function not enough to specify policy. Other ideas: Correlated equilirium, coco values (side payment)

9. Correlated equilirium: another coordinator to share information. can be found in poly time. all mixed nash are correlated, so CE exist. all convex combination of mixed Nash are correlated. 

10. Coco: cooperative-competitive values. share: side payment. cooperative (max) and competitive (minimax zero sum). efficiently computable, utility maximization, decompose game into sum of two games, unique, can be extended to stochastic games, not necessarily best response, doesn't generalize.
 
11. mechanism desgin: peer teaching, King Solomon, driving behavior via rewards


## Week 10

### Material

Lectures:

CCC
Readings:

Greenwald, Hall (2003)

### P3 

Soccer game

### Learning

1. POMDP: gain rewards/ gain information. define MDP rewards, maiximize. DEC-POMDP: decentrilized POMDP

2. IRL: inverse RL. behavior to rewards. MLIRL: guess R, compute pi, measure probility (D/pi), gradient on RL, loop back.

3. policy shaping: multiple sources. 

4. drama managment: player, agent, author. story: trajectory through plot points. states: partial sequences, actions: story actions, model: player model, rewards: author evaluation. TTD MDP- polcy is probability distribution over actions

