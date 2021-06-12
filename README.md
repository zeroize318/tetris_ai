# AI plays custom Tetris
## Requirement:
tensorflow [probably v2.5]

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



    
