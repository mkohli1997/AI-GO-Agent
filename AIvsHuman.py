import numpy as np
import time
from queue import PriorityQueue


MAX_MOVES = 12
BOARD_SIZE = 5
KOMI = 2.5
INIT_ARRAY=np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
BOARD_POSITIONS={(0,0),(0,1),(0,2),(0,3),(0,4),
                 (1,0),(1,1),(1,2),(1,3),(1,4),
                 (2,0),(2,1),(2,2),(2,3),(2,4),
                 (3,0),(3,1),(3,2),(3,3),(3,4),
                 (4,0),(4,1),(4,2),(4,3),(4,4)}
ORTHOGONAL_POSITIONS={(0, 0): {(0, 1), (1, 0)},
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
                      (4, 4): {(3, 4), (4, 3)},
                      }


def parse_input(file):
    pstate = np.array(INIT_ARRAY)
    cstate = np.array(INIT_ARRAY)

    content = file.read()
    list_content = content.split('\n')
    col_move = int(list_content[0])
    prev_str = list_content[1:6]
    c_str = list_content[6:]
    for x in range(len(prev_str)):
        for y in range(len(prev_str[x])):
            pstate[x][y] = int(prev_str[x][y])
            cstate[x][y] = int(c_str[x][y])

    return col_move, pstate, cstate


#######################################################################################################################


# EVAL METRIC 1: PARTIAL AREA SCORE
def get_pas_eval(white_p, black_p):
    pas_w = len(white_p)
    pas_b = len(black_p)
    w_score = pas_w + KOMI
    b_score = pas_b
    return w_score - b_score


# EVAL METRIC 2: MAXIMIZE LIBERTIES
def maximize_liberties(position, white_p, black_p):
    white_liberty_sum = 0
    black_liberty_sum = 0
    for pos in white_p:
        white_liberty_sum += position.board[pos[0]][pos[1]].liberty

    for pos in black_p:
        black_liberty_sum += position.board[pos[0]][pos[1]].liberty

    return white_liberty_sum - black_liberty_sum


# EVAL METRIC 3: MAKING 'EYES'
def detect_eyes(position, null_positions):
    w_eye_count = 0
    b_eye_count = 0

    for pos in null_positions:
        ortho = ORTHOGONAL_POSITIONS[(pos[0], pos[1])]
        if [position.board[p[0]][p[1]].color for p in ortho]==[1 for i in range(len(ortho))]:
            b_eye_count += 1

        if [position.board[p[0]][p[1]].color for p in ortho]==[2 for i in range(len(ortho))]:
            w_eye_count += 1

    return w_eye_count - b_eye_count


# EVAL METRIC 4: ANALYZE CHAINS FOR EACH
def analyze_chains(position):
    sum_chains_white = 0
    sum_chains_black = 0
    # print(position.chains)
    for key, val in position.chains.items():
        if key[1] == 2:
            sum_chains_white += len(val)
        elif key[1] == 1:
            sum_chains_black += len(val)

    return sum_chains_white - sum_chains_black


# EVAL METRIC 5: AVOID MOVES ON THE EDGE
def avoid_life_on_the_edge(position):
    whites_on_edge=np.count_nonzero(position.state[:,0]==2)+np.count_nonzero(position.state[:,4]==2)+np.count_nonzero\
        (position.state[0,1:4]==2)+np.count_nonzero(position.state[4,1:4]==2)
    blacks_on_edge=np.count_nonzero(position.state[:,0]==1)+np.count_nonzero(position.state[:,4]==1)+np.count_nonzero\
        (position.state[0,1:4]==1)+np.count_nonzero(position.state[4,1:4]==1)

    return blacks_on_edge - whites_on_edge


# EVAL METRIC 6: COUNT CAPTURABLE STONES
def stones_with_liberty_one(position, white_p, black_p):
    w_capturable, b_capturable = 0, 0
    for pos in white_p:
        if position.board[pos[0]][pos[1]].liberty == 1:
            w_capturable += 1
    for pos in black_p:
        if position.board[pos[0]][pos[1]].liberty == 1:
            b_capturable += 1
    return b_capturable - w_capturable


# EVAL METRIC 7: INFLUENCE
def compute_territory(position, null_p):
    w_territory = 0
    b_territory = 0

    for pos in null_p:
        right_stones = []
        left_stones = []
        up_stones = []
        down_stones = []
        l = 0
        r = 0
        u = 0
        d = 0
        # check right
        for i in range(pos[1], BOARD_SIZE):
            right_stones.append(position.state[pos[0]][i])
        # check left
        for i in range(pos[1], -1, -1):
            left_stones.append(position.state[pos[0]][i])
        # check down
        for i in range(pos[0], BOARD_SIZE):
            down_stones.append(position.state[i][pos[1]])
        # check up
        for i in range(pos[0], -1, -1):
            up_stones.append(position.state[i][pos[1]])

        # check right bound
        if 1 in right_stones and 2 not in right_stones:
            r = 1
        elif 2 in right_stones and 1 not in right_stones:
            r = 2
        elif 1 in right_stones and 2 in right_stones:
            if right_stones.index(1) < right_stones.index(2):
                r = 1
            else:
                r = 2

        # check left bound
        if 1 in left_stones and 2 not in left_stones:
            l = 1
        elif 2 in left_stones and 1 not in left_stones:
            l = 2
        elif 1 in left_stones and 2 in left_stones:
            if left_stones.index(1) < left_stones.index(2):
                l = 1
            else:
                l = 2
        # check upper bound
        if 1 in up_stones and 2 not in up_stones:
            u = 1
        elif 2 in up_stones and 1 not in up_stones:
            u = 2
        elif 1 in up_stones and 2 in up_stones:
            if up_stones.index(1) < up_stones.index(2):
                u = 1
            else:
                u = 2

        # check down bound
        if 1 in down_stones and 2 not in down_stones:
            d = 1
        elif 2 in down_stones and 1 not in down_stones:
            d = 2
        elif 1 in down_stones and 2 in down_stones:
            if down_stones.index(1) < down_stones.index(2):
                d = 1
            else:
                d = 2

        # mark it under a territory
        if 1 in {u, d, l, r} and 2 not in {u, d, l, r}:
            b_territory += 1
        elif 1 not in {u, d, l, r} and 2 in {u, d, l, r}:
            w_territory += 1

    return w_territory - b_territory


def evaluation_fn(position, AI):
    w_positions = position.white_positions
    b_positions = position.black_positions
    null_positions = position.null_positions

    if AI == 'W':
        PAS_COEFF = 30
        MAX_LIBERTY_COEFF = 2
        EYES_COEFF = 2
        CHAINING_COEFF=2
        EDGE_COEFF = 2
        CAP_EVAL_COEFF = 2
        INFLUENCE_COEFF = 0


    else:
        PAS_COEFF = 30
        MAX_LIBERTY_COEFF = 2
        EYES_COEFF = 2
        CHAINING_COEFF = 4
        EDGE_COEFF = 2
        CAP_EVAL_COEFF = 2
        if len(null_positions) > 12:
            INFLUENCE_COEFF = 2
        else:
            INFLUENCE_COEFF = 0

    pas_eval = get_pas_eval(w_positions, b_positions) * PAS_COEFF
    max_liberties_eval = maximize_liberties(position, w_positions, b_positions) * MAX_LIBERTY_COEFF
    eyes_eval = detect_eyes(position, null_positions)*EYES_COEFF
    chains_eval = analyze_chains(position) * CHAINING_COEFF
    edge_eval = avoid_life_on_the_edge(position) * EDGE_COEFF
    cap_eval = stones_with_liberty_one(position, w_positions, b_positions) * CAP_EVAL_COEFF
    if INFLUENCE_COEFF != 0:
        inf_eval = compute_territory(position, null_positions) * INFLUENCE_COEFF
    else:
        inf_eval = 0

    return pas_eval + max_liberties_eval + eyes_eval + chains_eval + cap_eval + edge_eval + inf_eval



def minimax(position, depth, alpha, beta, maxPlayer, AI, max_depth, count, maxmove=None, minmove=None):

    if time.time()-start_time>7:
        raise ValueError('Time Cutoff')

    if maxmove=='PASS' and minmove=='PASS':
        pas = get_pas_eval(position.white_positions, position.black_positions)
        if pas > 0:
            return 9999
        elif pas<0:
            return -9999

    if count >= MAX_MOVES:
        # print('AI2 knows the future!!')
        pas = get_pas_eval(position.white_positions, position.black_positions)
        if pas > 0:
            return 9999
        elif pas < 0:
            return -9999

    if depth == max_depth:  # or position.game_over
        return evaluation_fn(position, AI)

    if maxPlayer:
        maxEval = -float('inf')
        legal_moves = position.legal_moves_generator()
        children = position.split(legal_moves)

        for child in children:
            evaluation = minimax(child[0], depth + 1, alpha, beta, False, AI, max_depth, count + 1, child[1].pos, minmove)-depth
            if evaluation > maxEval:
                maxEval = evaluation
                if AI=='W':
                    move_stats[depth]=(child[1].pos, evaluation)

            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return maxEval

    else:
        minEval = float('inf')
        legal_moves = position.legal_moves_generator()
        children = position.split(legal_moves)

        for child in children:
            evaluation = minimax(child[0], depth + 1, alpha, beta, True, AI, max_depth, count, maxmove, child[1].pos)+depth
            if evaluation < minEval:
                minEval = evaluation
                if AI=='B':
                    move_stats[depth]=(child[1].pos, evaluation)

            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return minEval


class Move(object):
    def __init__(self, pos, for_color, board_obj):
        self.pos = pos
        self.for_color = for_color
        self.result_board, self.captures=board_obj.make_move(self.pos, self.for_color, return_capture=True)
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
        self.priority += self.captures*10


    def is_building_chain(self):
        for k, val in self.result_board.chains.items():
            if self.pos in val:
                self.priority += len(val)
                break

    def is_on_the_edge(self):
        if self.for_color==1:
            new_sum = np.count_nonzero(self.result_board.state[:, 0] == 1) + np.count_nonzero(
                self.result_board.state[:, 4] == 1) + np.count_nonzero(self.result_board.state[0, 1:4] == 1) + np.count_nonzero(
                self.result_board.state[4, 1:4] == 1)
        else:
            new_sum = np.count_nonzero(self.result_board.state[:, 0] == 2) + np.count_nonzero(
                self.result_board.state[:, 4] == 2) + np.count_nonzero(
                self.result_board.state[0, 1:4] == 2) + np.count_nonzero(
                self.result_board.state[4, 1:4] == 2)

        self.priority += (-new_sum) * 0.1


    def is_building_eye(self):
        pass

    def is_defending(self):
        if self.for_color==1:
            new_positions = self.result_board.black_positions
        else:
            new_positions = self.result_board.white_positions

        new_sum = 0
        for pos in new_positions:
            if self.result_board.board[pos[0]][pos[1]].liberty==1:
                new_sum += 1
        self.priority += (-new_sum)

    def is_maximizing_liberties(self):
        if self.for_color == 1:
            new_positions = self.result_board.black_positions
        else:
            new_positions = self.result_board.white_positions

        new_sum=0
        for pos in new_positions:
            new_sum+=self.result_board.board[pos[0]][pos[1]].liberty
        self.priority+=(new_sum*0.1)


class Board(object):
    def __init__(self, currState, prevState, color_to_move):
        self.to_move = color_to_move
        self.board = np.array(INIT_ARRAY, dtype='object')
        self.state = np.array(currState)
        self.prevState = prevState
        self.place_obj()
        self.chains = dict()
        self.free_pos = dict()
        self.find_free_ortho_spaces()
        self.update_chains()
        self.update_liberties()
        self.remove_stones_with_zero_liberties()
        self.null_positions, self.black_positions, self.white_positions=self.get_positions()


    def get_positions(self):
        null_pos, black_pos, white_pos=set(), set(), set()
        for pos in BOARD_POSITIONS:
            if self.state[pos[0]][pos[1]]==0:
                null_pos.add(pos)
            elif self.state[pos[0]][pos[1]]==1:
                black_pos.add(pos)
            else:
                white_pos.add(pos)
        return null_pos, black_pos, white_pos


    def remove_stones_with_zero_liberties(self):
        for pos in BOARD_POSITIONS:
            if self.board[pos[0]][pos[1]].liberty == 0:
                self.board[pos[0]][pos[1]] = Stone(color=0, pos=(pos[0], pos[1]))
                self.state[pos[0]][pos[1]] = 0

    def return_state(self, board):
        state = np.array(INIT_ARRAY)
        for pos in BOARD_POSITIONS:
            state[pos[0]][pos[1]] = board[pos[0]][pos[1]].color

        return state

    def find_free_ortho_spaces(self):
        for pos, ortho in ORTHOGONAL_POSITIONS.items():
            if self.board[pos[0]][pos[1]].color != 0:
                for p in ortho:
                    if self.board[p[0]][p[1]].color == 0:
                        if (pos[0], pos[1]) in self.free_pos.keys():
                            self.free_pos[(pos[0], pos[1])].add(p)
                        else:
                            self.free_pos[(pos[0], pos[1])] = {p}


    def update_liberties(self):
        # x1=time.time()
        for k, positions in self.chains.items():  # positions of chained stones
            free_pos = set()
            for position in positions:  # a single position out of those positions
                fpos = set()
                if position in self.free_pos.keys():
                    fpos = self.free_pos[position]  # extract ortho free positions for that position
                for pos in fpos:
                    if pos not in free_pos:
                        free_pos.add(pos)

            for position in positions:
                self.board[position[0]][position[1]].liberty = len(free_pos)
                self.board[position[0]][position[1]].chain_liberty = len(free_pos)


    def get_neighbors(self, node):
        neighbors=set()
        for neighbor in ORTHOGONAL_POSITIONS[(node.pos[0], node.pos[1])]:
            if self.board[neighbor[0]][neighbor[1]].color==node.color:
                neighbors.add(neighbor)
            else:
                continue
        return neighbors

    def bfs(self, color_list, color):
        chain_count=0
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
                self.board[node.pos[0]][node.pos[1]].chain_num=chain_count
                for neighbor in self.get_neighbors(node):
                    if neighbor not in visited:
                        queue.append(self.board[neighbor[0]][neighbor[1]])
                        visited.add(neighbor)
                        self.board[neighbor[0]][neighbor[1]].chain_num=chain_count
            self.chains[(chain_count, color)] = set(visited)
            chain_count += 1

    def update_chains(self):
        _,black_positions, white_positions = self.get_positions()

        self.bfs(black_positions, 1)
        self.bfs(white_positions, 2)


    def place_obj(self):
        for pos in BOARD_POSITIONS:
            self.board[pos[0]][pos[1]] = Stone(color=self.state[pos[0]][pos[1]], pos=(pos[0], pos[1]))
            self.board[pos[0]][pos[1]].liberty = self.board[pos[0]][pos[1]].compute_liberty(self.state)

    def legal_moves_generator(self, custom=False):
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
                if (temp_state == self.prevState).all():  # KO CHECK
                    illegal_moves.add(pos)
                    temp_state = np.array(self.state)
                    continue
                temp_state = np.array(self.state)

        possible_move_pos = possible_moves - illegal_moves
        if custom:
            return possible_move_pos
        legal_moves = PriorityQueue()

        for move in possible_move_pos:
            move_obj = Move(move, self.to_move, self)
            legal_moves.put((-move_obj.priority, move_obj))
        return legal_moves


    def remove_chain(self, chain, color, currState):
        for position in self.chains[(chain,color)]:
            currState[position[0]][position[1]]=0
        return currState

    def make_move(self, move, color_to_move, return_capture=False):
        captures = 0
        if move == 'PASS':
            board_copy = Board(self.state, self.prevState, self.to_move)
            if self.to_move == 1:
                board_copy.to_move = 2
            else:
                board_copy.to_move = 1
            if return_capture:
                return board_copy, captures
            else:
                return board_copy

        currState = np.array(self.state)
        ptemp_state = np.array(currState)

        for p in ORTHOGONAL_POSITIONS[move]:
            if self.board[p[0]][p[1]].chain_liberty == 1 and self.board[p[0]][p[1]].color != color_to_move:
                captures+=len(self.chains[(self.board[p[0]][p[1]].chain_num,self.board[p[0]][p[1]].color)])
                currState=self.remove_chain(self.board[p[0]][p[1]].chain_num, self.board[p[0]][p[1]].color, currState)

            elif self.board[p[0]][p[1]].liberty == 1 and self.board[p[0]][p[1]].color != color_to_move:
                captures+=1
                currState[p[0]][p[1]] = 0

        currState[move[0]][move[1]] = color_to_move
        if color_to_move == 1:
            temp_board = Board(currState, ptemp_state, 2)
        else:
            temp_board = Board(currState, ptemp_state, 1)
        if return_capture:
            return temp_board, captures
        else:
            return temp_board

    def split(self, moves):
        children = []
        while not moves.empty():
            move = moves.get()[1]
            children.append((self.make_move(move.pos, self.to_move), move))
        return children


class Stone(object):

    def __init__(self, color=None, pos=None):
        self.color = color
        self.pos = pos
        self.liberty = 0
        self.chain_liberty = 0
        self.chain_num=None

    def compute_liberty(self, state):
        if self.color == 0:
            return 0
        count = 0
        for p in ORTHOGONAL_POSITIONS[self.pos]:
            if state[p[0]][p[1]] == 0:
                count += 1
        return count


if __name__ == "__main__":

    move_stats = dict()

    time_taken=[]

    t1 = time.time()
    input_file = open('test_game1.txt', 'r')
    to_move, prev_state, curr_state = parse_input(input_file)
    game = Board(curr_state, prev_state, to_move)
    move = None
    move_str=None
    best_move = None
    start_time=None
    with open('move_count_vsHuman.txt', 'w') as file:
        file.write(str(0))

    player_color = input("Choose human player's color (w/b): ")

    for i in range(MAX_MOVES):
        print('MOVE---> ', i+1, '\n')
        move_stats = dict()
        print('Current board state:-->')
        print(game.state)
############################################################################################################
        if player_color == 'w':
            print('\nAI is thinking...')
            t = time.time()
            with open('move_count_vsHuman.txt', 'r') as file:
                count = int(file.read())

            if len(game.null_positions) > 23:
                MAX_DEPTH = 2
            else:
                MAX_DEPTH = 4

            start_time=time.time()
            try:
                eval_score = minimax(game, 0, -float('inf'), float('inf'), False, AI='B', max_depth=MAX_DEPTH, count=count)
            except:
                print('Time cutoff reached. More than 7 seconds elapsed!')
            best_move = move_stats[0][0]
            game = game.make_move(best_move, game.to_move)
            print('AI plays {}'.format(best_move))
            # print(move_stats[0][1])
            with open('move_count_vsHuman.txt', 'w') as file:
                file.write(str(count + 1))
            t2 = time.time() - t
            print('Took {} seconds.'.format(t2))
            time_taken.append(t2)
            print('\nCurrent board state-->')
            print(game.state)

            if best_move == 'PASS' and move_str == 'PASS':
                break

            print('\n')


            #############################################################################################################
            move_str = input('Play your move: ')
            if move_str == 'PASS':
                move = 'PASS'
                if best_move == 'PASS':
                    break
                game = game.make_move(move, game.to_move)
            else:
                move = (int(move_str.split(',')[0]), int(move_str.split(',')[1]))
                legal_moves = game.legal_moves_generator(custom=True)
                if move not in legal_moves:
                    print('Illegal move!! Player Lost!!')
                    exit(0)
                else:
                    game = game.make_move(move, game.to_move)

                if best_move == 'PASS' and move_str == 'PASS':
                    break

        #####################################################################################################################
        else:
            move_str = input('Play your move: ')
            if move_str == 'PASS':
                move = 'PASS'
                if best_move == 'PASS':
                    break
                game = game.make_move(move, game.to_move)
            else:
                move = (int(move_str.split(',')[0]), int(move_str.split(',')[1]))
                legal_moves = game.legal_moves_generator(custom=True)
                if move not in legal_moves:
                    print('Illegal move!! Player Lost!!')
                    exit(0)
                else:
                    game = game.make_move(move, game.to_move)

                if best_move == 'PASS' and move_str == 'PASS':
                    break

                print('\n')
#######################################################################################################################

                print('\nAI is thinking...')
                t = time.time()
                with open('move_count_vsHuman.txt', 'r') as file:
                    count = int(file.read())

                if len(game.null_positions) > 23:
                    MAX_DEPTH = 2
                else:
                    MAX_DEPTH = 4

                start_time = time.time()
                try:
                    eval_score = minimax(game, 0, -float('inf'), float('inf'), True, AI='W', max_depth=MAX_DEPTH,
                                         count=count)
                except:
                    print('Time cutoff reached. More than 7 seconds elapsed!')
                best_move = move_stats[0][0]
                game = game.make_move(best_move, game.to_move)
                print('AI plays {}'.format(best_move))
                # print(move_stats[0][1])
                with open('move_count_vsHuman.txt', 'w') as file:
                    file.write(str(count + 1))
                t2 = time.time() - t
                print('Took {} seconds.'.format(t2))
                time_taken.append(t2)
                print('\nCurrent board state-->')
                print(game.state)

                if best_move == 'PASS' and move_str == 'PASS':
                    break


    final_pas = get_pas_eval(game.white_positions, game.black_positions)
    print('Final state-->')
    print(game.state)
    if final_pas > 0:
        print('White wins by %f!!'%final_pas)
    elif final_pas < 0:
        print('Black wins by %f!!'%final_pas)
    else:
        print("It's a tie!!")
    print('Avg time per move for the agent: ', sum(time_taken)/len(time_taken))
    print('Max time for a move: ', max(time_taken))
