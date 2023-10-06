# AI plays custom Tetris
## Requirement:
tensorflow [probably v2.5]

## How does it work

#### Reinforcement Learning

At first, the agent will play random moves, saving the states and the given reward in a limited queue (replay memory). At the end of each episode (game), the agent will train itself (using a neural network) with a random sample of the replay memory. As more and more games are played, the agent becomes smarter, achieving higher and higher scores.

Since in reinforcement learning once an agent discovers a good 'path' it will stick with it, it was also considered an exploration variable (that decreases over time), so that the agent picks sometimes a random action instead of the one it considers the best. This way, it can discover new 'paths' to achieve higher scores.


#### Training

The training is based on the [Q Learning algorithm](https://en.wikipedia.org/wiki/Q-learning). Instead of using just the current state and reward obtained to train the network, it is used Q Learning (that considers the transition from the current state to the future one) to find out what is the best possible score of all the given states **considering the future rewards**, i.e., the algorithm is not greedy. This allows for the agent to take some moves that might not give an immediate reward, so it can get a bigger one later on (e.g. waiting to clear multiple lines instead of a single one).

The neural network will be updated with the given data (considering a play with reward *reward* that moves from *state* to *next_state*, the latter having an expected value of *Q_next_state*, found using the prediction from the neural network):

if not terminal state (last round): *Q_state* = *reward* + *discount* × *Q_next_state*
else: *Q_state* = *reward*


#### Best Action

Most of the deep Q Learning strategies used output a vector of values for a certain state. Each position of the vector maps to some action (ex: left, right, ...), and the position with the higher value is selected.

However, the strategy implemented was slightly different. For some round of Tetris, the states for all the possible moves will be collected. Each state will be inserted in the neural network, to predict the score obtained. The action whose state outputs the biggest value will be played.

## Usage:
1. edit "common.py";
   
   choose mode "human_player", "ai_player_training" and "ai_player_watching"
2. edit "tetromino.py" -> create_pool(cls): -> elif GAME_TYPE == 'extra':
    
    add or delete tetromino.
3. run "tetris_ai.py".

    training may take a significant amount of cpu usage.

## Links:
[Example video](https://youtu.be/FTDZN4pPhwA)

[Article on Medium](https://rex-l.medium.com/reinforcement-learning-on-tetris-707f75716c37)



    
