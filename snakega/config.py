GRID_SIZE = 30          # 30 x 30 cells
CELL = 10               # 10 px per cell
WIDTH = GRID_SIZE * CELL
HEIGHT = GRID_SIZE * CELL

# Policy (MLP) sizes
N_X = 7
N_H1 = 9
N_H2 = 15
N_Y = 3

# GA defaults
POPULATION = 30
PARENTS = 12
GENERATIONS = 10
MUTATION_RATE = 0.05
MUTATION_STD = 0.1

# Evaluation / fitness shaping
STEPS_PER_GAME = 800          # longer horizon helps learning
DEATH_PENALTY = 150
REPEAT_DIR_PENALTY = 1
REPEAT_DIR_BONUS = 2
APPLE_BONUS_SCORE = 200       # engine score increment
APPLE_FITNESS_BONUS = 5000    # fitness bump per apple
STEP_PENALTY = 0.01           # small per-step tax
DIST_SHAPING = 0.5            # reward for reducing distance to apple
STARVE_STEPS = 200            # end episode if no apple for this many steps

# Rendering
FPS_TRAIN = 0                 # headless
FPS_PLAY = 12                 # visible demo

# Eval settings
EVAL_GAMES = 3                # seeds per genome (averaged)
N_JOBS = 0                    # 0 => auto (use all cores); 1 => no parallel

RANDOM_SEED = 42
