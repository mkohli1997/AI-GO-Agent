import numpy as np
from queue import PriorityQueue

# max number of moves in the game
MAX_MOVES = 12

# size of the board
BOARD_SIZE = 5

# komi value
KOMI = 2.5

# array initialized to zeros
INIT_ARRAY = np.zeros((5, 5), dtype=int)

# hard coded the board positions: (0, 0) to (4, 4)
BOARD_POSITIONS = {(i, j) for j in range(BOARD_SIZE) for i in range(BOARD_SIZE)}

# hard coded each board position's orthogonal positions
ORTHOGONAL_POSITIONS = {(0, 0): {(0, 1), (1, 0)},
                        (0, 1): {(0, 0), (1, 1), (0, 2)},
                        (0, 2): {(0, 1), (1, 2), (0, 3)},
                        (0, 3): {(0, 2), (1, 3), (0, 4)},
                        (0, 4): {(0, 3), (1, 4)},
                        (1, 0): {(0, 0), (1, 1), (2, 0)},
                        (1, 1): {(0, 1), (1, 0), (1, 2), (2, 1)},
                        (1, 2): {(0, 2), (1, 3), (2, 2), (1, 1)},
                        (1, 3): {(0, 3), (1, 4), (2, 3), (1, 2)},
                        (1, 4): {(0, 4), (2, 4), (1, 3)},
                        (2, 0): {(1, 0), (2, 1), (3, 0)},
                        (2, 1): {(1, 1), (2, 2), (3, 1), (2, 0)},
                        (2, 2): {(1, 2), (2, 3), (3, 2), (2, 1)},
                        (2, 3): {(1, 3), (2, 4), (3, 3), (2, 2)},
                        (2, 4): {(1, 4), (3, 4), (2, 3)},
                        (3, 0): {(2, 0), (3, 1), (4, 0)},
                        (3, 1): {(2, 1), (3, 2), (4, 1), (3, 0)},
                        (3, 2): {(2, 2), (3, 3), (4, 2), (3, 1)},
                        (3, 3): {(2, 3), (3, 4), (4, 3), (3, 2)},
                        (3, 4): {(2, 4), (4, 4), (3, 3)},
                        (4, 0): {(3, 0), (4, 1)},
                        (4, 1): {(3, 1), (4, 2), (4, 0)},
                        (4, 2): {(3, 2), (4, 3), (4, 1)},
                        (4, 3): {(3, 3), (4, 4), (4, 2)},
                        (4, 4): {(3, 4), (4, 3)}}


class Board(object):
    def __init__(self, current_state=np.array(INIT_ARRAY), previous_state=np.array(INIT_ARRAY), color_to_move=1):
        self.to_move = color_to_move
        self.board = np.array(INIT_ARRAY, dtype='object')
        self.state = np.array(current_state)
        self.previous_state = previous_state
        self.place_obj()
        self.chains = dict()
        self.free_pos = dict()
        self.find_free_ortho_spaces()
        self.update_chains()
        self.update_liberties()
        self.remove_stones_with_zero_liberties()
        self.null_positions, self.black_positions, self.white_positions = self.get_positions()

    def get_positions(self):
        """
        Finds the positions occupied by the black and white stones, and also records the ones which are unoccupied.

        :return: white, black and null positions.
        """
        null_pos, black_pos, white_pos = set(), set(), set()
        for pos in BOARD_POSITIONS:
            if self.state[pos[0]][pos[1]] == 0:
                null_pos.add(pos)
            elif self.state[pos[0]][pos[1]] == 1:
                black_pos.add(pos)
            else:
                white_pos.add(pos)
        return null_pos, black_pos, white_pos

    def remove_stones_with_zero_liberties(self):
        """
        Removes stones with zero liberties from the board.

        :return: None
        """
        for pos in BOARD_POSITIONS:
            if self.board[pos[0]][pos[1]].liberty == 0:
                self.board[pos[0]][pos[1]] = Stone(color=0, pos=(pos[0], pos[1]))
                self.state[pos[0]][pos[1]] = 0

    @staticmethod
    def return_state(board):
        """
        Constructs an integer array showing the current position of the board.

        :param board: board object
        :return: current state in the form of an integer array
        """
        state = np.array(INIT_ARRAY)
        for pos in BOARD_POSITIONS:
            state[pos[0]][pos[1]] = board[pos[0]][pos[1]].color

        return state

    def find_free_ortho_spaces(self):
        """
        For each occupied position, finds the number of empty orthogonal positions for computing the final liberty
        scores.

        :return: None
        """
        for pos, ortho in ORTHOGONAL_POSITIONS.items():
            if self.board[pos[0]][pos[1]].color != 0:
                for p in ortho:
                    if self.board[p[0]][p[1]].color == 0:
                        if (pos[0], pos[1]) in self.free_pos.keys():
                            self.free_pos[(pos[0], pos[1])].add(p)
                        else:
                            self.free_pos[(pos[0], pos[1])] = {p}

    def update_liberties(self):
        """
        Updates liberties of each stone, including changes in the liberty due to chain formations.

        :return: None
        """
        for k, positions in self.chains.items():  # positions of chained stones
            free_pos = set()
            for position in positions:  # a single position out of those positions
                fpos = set()
                if position in self.free_pos.keys():
                    fpos = self.free_pos[position]  # extract orthogonal free positions for that position
                for pos in fpos:
                    if pos not in free_pos:
                        free_pos.add(pos)

            for position in positions:
                self.board[position[0]][position[1]].liberty = len(free_pos)
                self.board[position[0]][position[1]].chain_liberty = len(free_pos)

    def get_neighbors(self, node):
        """
        Finds neighbors of a stone, i.e. a stone with same color in any of the orthogonal positions.

        :param node: current position for which neighbors need to be found.
        :return: neighbors of the current stone
        """
        neighbors = set()
        for neighbor in ORTHOGONAL_POSITIONS[(node.pos[0], node.pos[1])]:
            if self.board[neighbor[0]][neighbor[1]].color == node.color:
                neighbors.add(neighbor)
            else:
                continue
        return neighbors

    def bfs(self, color_list, color):
        """
        Using breadth-first search, finds the chains formed by the given color.

        :param color_list: positions occupied by the given color.
        :param color: black or white (1 or 2)
        :return: None
        """
        chain_count = 0
        for position in color_list:
            found = False
            for k, val in self.chains.items():
                if position in val:
                    found = True
            if found:
                continue
            queue, visited = [], set()
            root = self.board[position[0]][position[1]]

            if len(self.get_neighbors(root)) == 0:
                continue

            queue.append(root)
            visited.add(position)
            while len(queue) != 0:
                node = queue.pop(0)
                self.board[node.pos[0]][node.pos[1]].chain_num = chain_count
                for neighbor in self.get_neighbors(node):
                    if neighbor not in visited:
                        queue.append(self.board[neighbor[0]][neighbor[1]])
                        visited.add(neighbor)
                        self.board[neighbor[0]][neighbor[1]].chain_num = chain_count
            self.chains[(chain_count, color)] = set(visited)
            chain_count += 1

    def update_chains(self):
        """
        Calls the bfs function to store the chains formed by each player.

        :return: None
        """
        _, black_positions, white_positions = self.get_positions()

        self.bfs(black_positions, 1)
        self.bfs(white_positions, 2)

    def place_obj(self):
        """
        Constructs an array of objects, in which each element is an object of the Stone class.

        :return: None
        """
        for pos in BOARD_POSITIONS:
            self.board[pos[0]][pos[1]] = Stone(color=self.state[pos[0]][pos[1]], pos=(pos[0], pos[1]))
            self.board[pos[0]][pos[1]].liberty = self.board[pos[0]][pos[1]].compute_liberty(self.state)

    def legal_moves_generator(self, custom=False):
        """
        Finds the legal moves possible in the current state of the board.

        :param custom: If True, returns just a list of tuples, which are the legal moves. Else returns a priority queue
        of legal moves, containing objects of the Move class, prioritized by the priority value assigned to each move.
        :return: Legal moves
        """
        possible_moves = self.null_positions
        possible_moves.add('PASS')
        temp_state = np.array(self.state)
        illegal_moves = set()
        for pos in possible_moves:
            illegal = True
            if pos != 'PASS':
                ortho = ORTHOGONAL_POSITIONS[(pos[0], pos[1])]
                for p in ortho:
                    if self.state[p[0]][p[1]] == 0:
                        illegal = False
                        break
                    elif self.to_move != self.board[p[0]][p[1]].color:
                        if self.board[p[0]][p[1]].liberty == 1:
                            illegal = False
                            break

                    elif self.state[p[0]][p[1]] == self.to_move:
                        if self.board[p[0]][p[1]].liberty > 1:
                            illegal = False
                            break
                if illegal:
                    illegal_moves.add(pos)
                    temp_state = np.array(self.state)
                    continue

                for p in ortho:
                    if self.to_move != self.board[p[0]][p[1]].color:
                        if self.board[p[0]][p[1]].liberty == 1:
                            temp_state[p[0]][p[1]] = 0

                temp_state[pos[0]][pos[1]] = self.to_move
                if (temp_state == self.previous_state).all():  # KO RULE CHECK
                    illegal_moves.add(pos)
                    temp_state = np.array(self.state)
                    continue
                temp_state = np.array(self.state)

        possible_move_pos = possible_moves - illegal_moves
        if custom:
            return possible_move_pos

        legal_moves_queue = PriorityQueue()

        for possible_move in possible_move_pos:
            move_obj = Move(possible_move, self.to_move, self)
            legal_moves_queue.put((-move_obj.priority, move_obj))
        return legal_moves_queue

    def remove_chain(self, chain, color, current_state):
        """
        Removes a given chain from the board.

        :param chain: chain to be removed
        :param color: color of the chain
        :param current_state: current state on the board
        :return: current state of the board
        """
        for position in self.chains[(chain, color)]:
            current_state[position[0]][position[1]] = 0
        return current_state

    def make_move(self, move_to_play, color_to_move, return_capture=False):
        """
        Plays the given move on the board and returns a new Board object with the resulting state.

        :param move_to_play: the move which has to be played.
        :param color_to_move: the color which is playing the move
        :param return_capture: If true, returns the number of captures made by the move.
        :return: new Board object with the resulting state
        """
        captures = 0
        if move_to_play == 'PASS':
            board_copy = Board(self.state, self.previous_state, self.to_move)
            if self.to_move == 1:
                board_copy.to_move = 2
            else:
                board_copy.to_move = 1
            if return_capture:
                return board_copy, captures
            else:
                return board_copy

        current_state = np.array(self.state)
        ptemp_state = np.array(current_state)

        for p in ORTHOGONAL_POSITIONS[move_to_play]:
            if self.board[p[0]][p[1]].chain_liberty == 1 and self.board[p[0]][p[1]].color != color_to_move:
                captures += len(self.chains[(self.board[p[0]][p[1]].chain_num, self.board[p[0]][p[1]].color)])
                current_state = self.remove_chain(self.board[p[0]][p[1]].chain_num, self.board[p[0]][p[1]].color,
                                                  current_state)

            elif self.board[p[0]][p[1]].liberty == 1 and self.board[p[0]][p[1]].color != color_to_move:
                captures += 1
                current_state[p[0]][p[1]] = 0

        current_state[move_to_play[0]][move_to_play[1]] = color_to_move
        if color_to_move == 1:
            temp_board = Board(current_state, ptemp_state, 2)
        else:
            temp_board = Board(current_state, ptemp_state, 1)
        if return_capture:
            return temp_board, captures
        else:
            return temp_board

    def split(self, moves):
        """
        Given a priority queue of potential moves, returns the resulting board for each, alongwith the Move object
        corresponding to the move that was played.
        :param moves: Priority queue of moves
        :return: resulting Board objects from the given set of moves.
        """
        children = []
        while not moves.empty():
            move_pos = moves.get()[1]
            children.append((self.make_move(move_pos.pos, self.to_move), move_pos))

        return children


class Move(object):
    def __init__(self, pos, for_color, board_obj):
        self.pos = pos
        self.for_color = for_color
        self.result_board, self.captures = board_obj.make_move(self.pos, self.for_color, return_capture=True)
        self.priority = 0
        if self.pos != 'PASS':
            self.is_capturing()
            self.is_building_chain()
            self.is_on_the_edge()
            self.is_maximizing_liberties()
            self.is_defending()

    def __lt__(self, other):
        return True

    def is_capturing(self):
        """
        Increases the priority of the move if it is a capturing move
        :return: None
        """
        self.priority += self.captures*10

    def is_building_chain(self):
        """
        Increases the priority of the move if it is forming a chain
        :return: None
        """
        for k, val in self.result_board.chains.items():
            if self.pos in val:
                self.priority += len(val)
                break

    def is_on_the_edge(self):
        """
        Decreases the priority of the move if it is on the edge of the board.
        :return: None
        """
        if self.for_color == 1:
            new_sum = np.count_nonzero(self.result_board.state[:, 0] == 1) + np.count_nonzero(
                self.result_board.state[:, 4] == 1) + np.count_nonzero(self.result_board.state[0, 1:4] == 1) +\
                      np.count_nonzero(self.result_board.state[4, 1:4] == 1)
        else:
            new_sum = np.count_nonzero(self.result_board.state[:, 0] == 2) + np.count_nonzero(
                self.result_board.state[:, 4] == 2) + np.count_nonzero(
                self.result_board.state[0, 1:4] == 2) + np.count_nonzero(
                self.result_board.state[4, 1:4] == 2)

        self.priority += (-new_sum) * 0.1

    def is_building_eye(self):
        """
        This can be a potential addition to the mechanism of assigning each legal move a priority before it is explored
        by the algorithm. A move can also be prioritized if it is forming an eye.
        :return:
        """
        pass

    def is_defending(self):
        """
        Increases the priority of the move if it defends a potential capture by the opponent.
        :return: None
        """
        if self.for_color == 1:
            new_positions = self.result_board.black_positions
        else:
            new_positions = self.result_board.white_positions

        new_sum = 0
        for pos in new_positions:
            if self.result_board.board[pos[0]][pos[1]].liberty == 1:
                new_sum += 1
        self.priority += (-new_sum)

    def is_maximizing_liberties(self):
        """
        Increases the priority of the move if it maximizes the total liberty for its team.
        :return: None
        """
        if self.for_color == 1:
            new_positions = self.result_board.black_positions
        else:
            new_positions = self.result_board.white_positions

        new_sum = 0
        for pos in new_positions:
            new_sum += self.result_board.board[pos[0]][pos[1]].liberty
        self.priority += (new_sum * 0.1)


class Stone(object):
    def __init__(self, color=None, pos=None):
        self.color = color
        self.pos = pos
        self.liberty = 0
        self.chain_liberty = 0
        self.chain_num = None

    def compute_liberty(self, state):
        """
        Computes the initial liberty of the stone.

        :param state: current state of the board
        :return: liberty of the current stone object
        """
        if self.color == 0:
            return 0
        liberty = 0
        for p in ORTHOGONAL_POSITIONS[self.pos]:
            if state[p[0]][p[1]] == 0:
                liberty += 1
        return liberty
