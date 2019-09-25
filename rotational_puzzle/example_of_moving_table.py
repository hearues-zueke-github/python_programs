movingTable = {
    (4, 4): { # field size x=4 and y=4
        15: { # for the last number, with is 16, but index is 15!
            (0, 0): [(1, 1, 3, True), (2, 2, 3, True)], # list of moves, where one move is (x, y, r, c)
            (1, 0): [(1, 1, 2, True), (2, 2, 3, True)], # x, y ... position of the center of the rotation!
            (2, 0): [(1, 1, 1, True), (2, 2, 3, True)], # r ... amount of rotations (1 to 4 with the size 3x3 of the rotational field)
            (3, 0): [(2, 1, 1, True), (2, 2, 2, True)], # c ... clockwise = True or counter/anti clockwise = False

            (0, 1): [(1, 2, 1, True), (2, 2, 4, True)],
            (1, 1): [(2, 2, 4, True)],
            (2, 1): [(2, 2, 3, True)],
            (3, 1): [(2, 2, 2, True)],

            (0, 2): [(1, 2, 3, False), (2, 2, 1, False)],
            (1, 2): [(2, 2, 3, False)],
            (2, 2): [(1, 2, 1, True), (2, 2, 1, False)],
            (3, 2): [(2, 2, 1, True)],

            (0, 3): [(1, 2, 1, False), (2, 2, 2, False)],
            (1, 3): [(2, 2, 2, False)],
            (2, 3): [(2, 2, 1, False)]
        }
    }
}
