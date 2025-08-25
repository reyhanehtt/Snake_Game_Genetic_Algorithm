# Snake GAME (Neuroevolution Snake)

Train a simple neural policy (3-layer MLP) for the classic Snake game using a Genetic Algorithm (GA).

## Features
- Headless training using Pygame's dummy video driver
- Clean engine (`snakega/game.py`) with a grid-based world (30×30, 10px cells)
- Compact MLP policy (7→9→15→3) with softmax output over **left/straight/right** relative turns
- Simple GA with elitism, uniform crossover, and Gaussian mutation
- Reproducible seeds & CLI scripts


