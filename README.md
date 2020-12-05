# AI-GO-Agent: AI Agent for the game of 'GO' on a 5x5 board

##### Algorithm Used 
Minimax Algorithm with Alpha-Beta Pruning. The game tree depth equals the difficulty level chosen by the user. The game supports upto 5 difficulty levels, 1 being the easiest and 5 being the hardest difficulty level.
**Minimax Algorithm**: https://en.wikipedia.org/wiki/Minimax
**Alpha Beta Pruning**: https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning

##### Gameplay Instructions
- Run the AIvsHuman.py script to start playing.
- Enter 'w' or 'b' to play as 'White' or 'Black' when prompted by the script.
- Black moves first.
- The game will be played for 12 full moves (24 ply) or till the time the human player makes an illegal move.
- Enter your move in the format x,y. These correspond to the coordinates on the board.
- Board position indices are in the range [0,4].
- Komi rule is applicable for White.
- KO moves are not allowed. **Komi Rule for GO:** https://en.wikipedia.org/wiki/Ko_fight 
- If the human player makes an illegal move, the AI agent wins.


