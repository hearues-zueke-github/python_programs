# Genetic Algorithm with the Tetris game

First, some file structure and there usage:

### Create pieces
* `create_tetromino_pieces.py`
    * Create all possible polyomino with a certain cell size (4 neighborhood)

### Calculate next piece position and create images/video
* `utils_tetris.h, utils_tetris.cpp`
    * Some basic functions which are used in tetris_deep_search_path_optimization.cpp
* `tetris_deep_search_path_optimization.cpp`
    * Is the main program for finding the best possible piece for the next move
    * The field size can be changed dynamically via program call + other arguments
    * At the beginning of the program a fixed sequence of `n` pieces is set
* `create_images_from_tetris_game_data.py`
    * Is used to create a `gif`/`mp4` file from the `*.ttrfields` files

### Execute the scripts in an order
* `simple_example.sh`
    * Execute the file `tetris_deep_search_path_optimization.o` and `create_images_from_tetris_game_data.py` together
* `do_many_simple_examples.sh`
    * Execute the simple_example.sh file many times in parallel!
