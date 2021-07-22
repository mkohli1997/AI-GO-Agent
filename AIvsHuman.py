import time
from utils import MAX_MOVES, Board
from metrics import get_pas_eval, evaluation_fn

# HEADER COMMENT


def minimax(position, depth, alpha, beta, max_player, agent, max_depth, move_count, maxmove=None, minmove=None):
    """
    The minimax function that calls itself recursively and traverses the game tree.

    :param position: The game board object
    :param depth: Current depth of the game tree being analysed
    :param alpha: Alpha score
    :param beta: Beta score
    :param max_player: True if the current player being analysed is the maximizer, else False
    :param agent: color of the AI player
    :param max_depth: Max depth of the game tree
    :param move_count: Keeps track of the move count in recursive calls
    :param maxmove: Best move of the maximizing player
    :param minmove: Best move of the minimizing player
    :return:
    """
    if maxmove == 'PASS' and minmove == 'PASS':
        pas = get_pas_eval(position.white_positions, position.black_positions)
        if pas > 0:
            return 9999
        elif pas < 0:
            return -9999

    if count >= MAX_MOVES:
        pas = get_pas_eval(position.white_positions, position.black_positions)
        if pas > 0:
            return 9999
        elif pas < 0:
            return -9999

    if depth == max_depth:
        return evaluation_fn(position, agent)

    if max_player:
        maxEval = -float('inf')
        legalMoves = position.legal_moves_generator()
        children = position.split(legalMoves)

        for child in children:
            evaluation = minimax(child[0], depth + 1, alpha, beta, False, agent, max_depth, move_count + 1,
                                 child[1].pos, minmove) - depth
            if evaluation > maxEval:
                maxEval = evaluation
                if agent == 'W':
                    move_stats[depth] = (child[1].pos, evaluation)

            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return maxEval

    else:
        minEval = float('inf')
        legalMoves = position.legal_moves_generator()
        children = position.split(legalMoves)

        for child in children:
            evaluation = minimax(child[0], depth + 1, alpha, beta, True, agent, max_depth, move_count, maxmove,
                                 child[1].pos) + depth
            if evaluation < minEval:
                minEval = evaluation
                if agent == 'B':
                    move_stats[depth] = (child[1].pos, evaluation)

            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return minEval


if __name__ == "__main__":

    move_stats = dict()

    time_taken = []

    t1 = time.time()
    game = Board()
    move = None
    move_str = None
    best_move = None
    start_time = None
    with open('move_count_vsHuman.txt', 'w') as file:
        file.write(str(0))

    difficulty = None
    try:
        difficulty = int(input("Enter difficulty level (1-5): "))
        if difficulty > 5:
            raise ValueError()
    except ValueError:
        print("Please specify an integer in the range 1-5.")
        exit(0)

    player_color = input("Choose human player's color (w/b): ")
    if player_color not in {"w", "b"}:
        print("Invalid choice! Terminating...")
        exit(0)

    for i in range(MAX_MOVES):
        print('MOVE---> ', i+1, '\n')
        move_stats = dict()
        print('\nCurrent board state:-->')
        print(game.state)

        if player_color == 'w':
            print('\nAI is thinking...')
            t = time.time()
            with open('move_count_vsHuman.txt', 'r') as file:
                count = int(file.read())

            if len(game.null_positions) > 23:
                MAX_DEPTH = 2
            else:
                MAX_DEPTH = difficulty

            start_time = time.time()

            eval_score = minimax(game, 0, -float('inf'), float('inf'), False, agent='B', max_depth=MAX_DEPTH,
                                 move_count=count)

            best_move = move_stats[0][0]
            game = game.make_move(best_move, game.to_move)
            print('agent plays {}'.format(best_move))

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

                print('\nAI is thinking...')
                t = time.time()
                with open('move_count_vsHuman.txt', 'r') as file:
                    count = int(file.read())

                if len(game.null_positions) > 23:
                    MAX_DEPTH = 2
                else:
                    MAX_DEPTH = difficulty

                start_time = time.time()

                eval_score = minimax(game, 0, -float('inf'), float('inf'), True, agent='W', max_depth=MAX_DEPTH,
                                     move_count=count)

                best_move = move_stats[0][0]
                game = game.make_move(best_move, game.to_move)
                print('AI plays {}'.format(best_move))

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
        print('White wins by %f!!' % final_pas)
    elif final_pas < 0:
        print('Black wins by %f!!' % final_pas)
    else:
        print("It's a tie!!")
    print('Avg time per move for the agent: ', sum(time_taken)/len(time_taken))
    print('Max time for a move: ', max(time_taken))
