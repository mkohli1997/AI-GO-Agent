from utils import BOARD_SIZE, KOMI, ORTHOGONAL_POSITIONS
import numpy as np


# EVAL METRIC 1: PARTIAL AREA SCORE
def get_pas_eval(white_p, black_p):
    """
    Computes area score for each player and returns the difference.

    :param white_p: positions occupied by white stones.
    :param black_p: positions occupied by the black stones.
    :return: difference between area scores of the two players.
    """
    pas_w = len(white_p)
    pas_b = len(black_p)
    w_score = pas_w + KOMI
    b_score = pas_b
    return w_score - b_score


# EVAL METRIC 2: MAXIMIZE LIBERTIES
def maximize_liberties(position, white_p, black_p):
    """
    Computes difference between total liberty of all pieces belonging to each player.

    :param position: board object with the current board configurations.
    :param white_p: positions occupied by white stones.
    :param black_p: positions occupied by the black stones.
    :return: difference between total liberty of each player.
    """
    white_liberty_sum = 0
    black_liberty_sum = 0
    for pos in white_p:
        white_liberty_sum += position.board[pos[0]][pos[1]].liberty

    for pos in black_p:
        black_liberty_sum += position.board[pos[0]][pos[1]].liberty

    return white_liberty_sum - black_liberty_sum


# EVAL METRIC 3: MAKING 'EYES'
def detect_eyes(position, null_positions):
    """
    Computes difference between total number of eyes formed by the players.

    :param position: board object with the current board configurations.
    :param null_positions: empty positions on the board.
    :return: difference in the number of eyes formed by white and black.
    """
    w_eye_count = 0
    b_eye_count = 0

    for pos in null_positions:
        ortho = ORTHOGONAL_POSITIONS[(pos[0], pos[1])]
        if [position.board[p[0]][p[1]].color for p in ortho] == [1 for _ in range(len(ortho))]:
            b_eye_count += 1

        if [position.board[p[0]][p[1]].color for p in ortho] == [2 for _ in range(len(ortho))]:
            w_eye_count += 1

    return w_eye_count - b_eye_count


# EVAL METRIC 4: ANALYZE CHAINS FOR EACH PLAYER
def analyze_chains(position):
    """
    Computes difference between sum of lengths of chains formed by each player.

    :param position: board object with the current position.
    :return: difference between sum of lengths of chains.
    """
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
    """
    Computes difference between the total number of stones on the edge of the board belonging to each player.

    :param position: board object with the current position
    :return: difference between the total number of stones on the edge of the board belonging to each player.
    """
    whites_on_edge = np.count_nonzero(position.state[:, 0] == 2) + np.count_nonzero(position.state[:, 4] == 2) + \
                     np.count_nonzero(position.state[0, 1:4] == 2) + np.count_nonzero(position.state[4, 1:4] == 2)
    blacks_on_edge = np.count_nonzero(position.state[:, 0] == 1) + np.count_nonzero(position.state[:, 4] == 1) + \
                     np.count_nonzero(position.state[0, 1:4] == 1) + np.count_nonzero(position.state[4, 1:4] == 1)

    return blacks_on_edge - whites_on_edge


# EVAL METRIC 6: COUNT CAPTURABLE STONES
def stones_with_liberty_one(position, white_p, black_p):
    """
    Computes difference between the total number of stones with liberties exactly equal to 1 (capturable in the next
    move) belonging to each player.

    :param position: board object.
    :param white_p: positions occupied by white stones.
    :param black_p: positions occupied by black stones.
    :return: difference between the total number of stones with liberties exactly equal to 1 belonging to each player.
    """
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
    """
    A rough heuristic to determine the spread of the territory of each player and return the difference of the scores.

    :param position: board object
    :param null_p: unoccupied positions on the board.
    :return: difference in territory scores.
    """
    w_territory = 0
    b_territory = 0

    for pos in null_p:
        right_stones = []
        left_stones = []
        up_stones = []
        down_stones = []
        left = 0
        right = 0
        up = 0
        down = 0
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
            right = 1
        elif 2 in right_stones and 1 not in right_stones:
            right = 2
        elif 1 in right_stones and 2 in right_stones:
            if right_stones.index(1) < right_stones.index(2):
                right = 1
            else:
                right = 2

        # check left bound
        if 1 in left_stones and 2 not in left_stones:
            left = 1
        elif 2 in left_stones and 1 not in left_stones:
            left = 2
        elif 1 in left_stones and 2 in left_stones:
            if left_stones.index(1) < left_stones.index(2):
                left = 1
            else:
                left = 2
        # check upper bound
        if 1 in up_stones and 2 not in up_stones:
            up = 1
        elif 2 in up_stones and 1 not in up_stones:
            up = 2
        elif 1 in up_stones and 2 in up_stones:
            if up_stones.index(1) < up_stones.index(2):
                up = 1
            else:
                up = 2

        # check down bound
        if 1 in down_stones and 2 not in down_stones:
            down = 1
        elif 2 in down_stones and 1 not in down_stones:
            down = 2
        elif 1 in down_stones and 2 in down_stones:
            if down_stones.index(1) < down_stones.index(2):
                down = 1
            else:
                down = 2

        # mark it under a territory
        if 1 in {up, down, left, right} and 2 not in {up, down, left, right}:
            b_territory += 1
        elif 1 not in {up, down, left, right} and 2 in {up, down, left, right}:
            w_territory += 1

    return w_territory - b_territory


def evaluation_fn(position, agent):
    """
    Combines each evaluation metric and returns a weighted sum of the evaluation scores. A positive score means an
    advantage for the maximizer and a negative score means an advantage for the minimizer.

    :param position: board object
    :param agent: color assigned to the AI agent - "b" or "w"
    :return: evaluation score of the given position
    """
    w_positions = position.white_positions
    b_positions = position.black_positions
    null_positions = position.null_positions

    # assignment of coefficients to use when combining the metrics
    if agent == 'W':
        PAS_COEFF = 30
        MAX_LIBERTY_COEFF = 2
        EYES_COEFF = 2
        CHAINING_COEFF = 2
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
    eyes_eval = detect_eyes(position, null_positions) * EYES_COEFF
    chains_eval = analyze_chains(position) * CHAINING_COEFF
    edge_eval = avoid_life_on_the_edge(position) * EDGE_COEFF
    cap_eval = stones_with_liberty_one(position, w_positions, b_positions) * CAP_EVAL_COEFF

    if INFLUENCE_COEFF != 0:
        inf_eval = compute_territory(position, null_positions) * INFLUENCE_COEFF
    else:
        inf_eval = 0

    return pas_eval + max_liberties_eval + eyes_eval + chains_eval + cap_eval + edge_eval + inf_eval
