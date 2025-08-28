import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import random
from time import sleep

pts_jogador = 0
pts_pc = 0

Q_TABLE_FILE = 'q_table.pkl'

ALPHA = 0.1     # Taxa de aprendizado
GAMMA = 0.9     # Fator de desconto para recompensas futuras
EPSILON = 0.1   # Probabilidade de explorar (jogar aleatoriamente)
NUM_EPISODES = 70000  # Número de jogos de treinamento

q_table = {}

def get_state(board):
    return tuple(board)

def check_winner(board, player):
    win_conditions = [
        # Linhas
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        # Colunas
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        # Diagonais
        (0, 4, 8), (2, 4, 6)
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

def get_reward(board, player):
    if check_winner(board, player):
        return 1    # Vitoria
    elif check_winner(board, 'O' if player == 'X' else 'X'):
        return -1   # Derrota
    elif ' ' not in board:
        return -.5   # Empate
    else:
        return 0    # Jogo continua

def evaluate_agent(q_table, num_games=1000):
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(num_games):
        board = [' '] * 9
        done = False
        
        while not done:
            state = get_state(board)
            if state in q_table and q_table[state]:
                valid_moves = {k: v for k, v in q_table[state].items() if board[k] == ' '}
                if valid_moves:
                    action = max(valid_moves, key=valid_moves.get)
                else: 
                    break
            else:
                possible_moves = [i for i, spot in enumerate(board) if spot == ' ']
                if not possible_moves: break
                action = random.choice(possible_moves)

            board[action] = 'X'
            
            if check_winner(board, 'X'):
                wins += 1
                done = True
                continue
            if ' ' not in board:
                draws += 1
                done = True
                continue

            possible_moves = [i for i, spot in enumerate(board) if spot == ' ']
            if not possible_moves:
                draws += 1
                done = True
                continue
                
            opponent_move = random.choice(possible_moves)
            board[opponent_move] = 'O'

            if check_winner(board, 'O'):
                losses += 1
                done = True
                continue
            if ' ' not in board:
                draws += 1
                done = True
                continue

    return {"wins": wins / num_games, "losses": losses / num_games, "draws": draws / num_games}


def train_q_agent_with_evaluation(alpha, gamma, epsilon, num_episodes):
    q_table = {}
    epsilon_decay = epsilon / num_episodes
    
    performance_history = {'episodes': [], 'wins': [], 'losses': [], 'draws': []}
    
    evaluation_interval = num_episodes // 100 # Avalia a cada 1% dos episódios

    for episode in range(1, num_episodes + 1):
        board = [' '] * 9
        state = get_state(board)
        done = False
        
        while not done:
            if state not in q_table:
                q_table[state] = {i: 0.0 for i in range(9) if board[i] == ' '}

            current_epsilon = epsilon - (episode * epsilon_decay)
            if random.uniform(0, 1) < current_epsilon:
                possible_moves = [i for i, spot in enumerate(board) if spot == ' ']
                if not possible_moves: break
                action = random.choice(possible_moves)
            else:
                if not q_table[state]:
                    possible_moves = [i for i, spot in enumerate(board) if spot == ' ']
                    action = random.choice(possible_moves) if possible_moves else -1
                else:
                    action = max(q_table[state], key=q_table[state].get)
            
            if action == -1: break

            board[action] = 'X'
            reward = get_reward(board, 'X')
            old_q_value = q_table[state].get(action, 0.0)
            
            if check_winner(board, 'X') or check_winner(board, 'O') or ' ' not in board:
                done = True
                max_q_next_state = 0.0
            else:
                # Oponente joga aleatoriamente
                open_spots = [i for i, spot in enumerate(board) if spot == ' ']
                opponent_move = random.choice(open_spots)
                board[opponent_move] = 'O'
                if check_winner(board, 'O') or ' ' not in board:
                    done = True
                next_state_after_opponent = get_state(board)
                if next_state_after_opponent not in q_table:
                    q_table[next_state_after_opponent] = {i: 0.0 for i in range(9) if board[i] == ' '}
                max_q_next_state = max(q_table[next_state_after_opponent].values()) if q_table[next_state_after_opponent] else 0.0
            
            reward = get_reward(board, 'X')
            new_q_value = old_q_value + alpha * (reward + gamma * max_q_next_state - old_q_value)
            q_table[state][action] = new_q_value
            state = get_state(board)
            
        if episode % evaluation_interval == 0:
            print(f"Episódio {episode}/{num_episodes} - Avaliando...")
            performance = evaluate_agent(q_table, num_games=1000)
            performance_history['episodes'].append(episode)
            performance_history['wins'].append(performance['wins'])
            performance_history['losses'].append(performance['losses'])
            performance_history['draws'].append(performance['draws'])
            
    return q_table, performance_history

def train_q_agent():
    global q_table
    print("Treinando o agente de Q-Learning... Por favor, aguarde.")
    
    epsilon_decay = EPSILON / NUM_EPISODES

    for episode in range(NUM_EPISODES):
        board = [' '] * 9
        state = get_state(board)
        done = False
        
        while not done:
            if state not in q_table:
                q_table[state] = {i: 0.0 for i in range(9) if board[i] == ' '}

            if random.uniform(0, 1) < EPSILON - (episode * epsilon_decay):
                possible_moves = [i for i, spot in enumerate(board) if spot == ' ']
                if not possible_moves:
                    break
                action = random.choice(possible_moves)
            else:
                if not q_table[state]:
                    possible_moves = [i for i, spot in enumerate(board) if spot == ' ']
                    action = random.choice(possible_moves) if possible_moves else -1
                else:
                    action = max(q_table[state], key=q_table[state].get)
                    
            if action == -1:
                break

            board[action] = 'X'
            reward = get_reward(board, 'X')
            
            next_state = get_state(board)
            
            old_q_value = q_table[state].get(action, 0.0)
            
            if reward != 0:
                max_q_next_state = 0.0
                done = True
            else:
                open_spots = [i for i, spot in enumerate(board) if spot == ' ']
                if not open_spots:
                    done = True
                else:
                    opponent_move = random.choice(open_spots)
                    board[opponent_move] = 'O'
                    if get_reward(board, 'O') != 0:
                        done = True
                    
                    next_state = get_state(board)
                    
                    if next_state not in q_table:
                        q_table[next_state] = {i: 0.0 for i in range(9) if board[i] == ' '}
                    max_q_next_state = max(q_table[next_state].values()) if q_table[next_state] else 0.0

            new_q_value = old_q_value + ALPHA * (reward + GAMMA * max_q_next_state - old_q_value)
            q_table[state][action] = new_q_value
            
            state = next_state
    
    print("Treinamento concluído. O agente agora está pronto para jogar!")
    save_q_table()

def save_q_table():
    with open(Q_TABLE_FILE, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Tabela Q salva em '{Q_TABLE_FILE}'.")

def load_q_table():
    global q_table
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, 'rb') as f:
            q_table = pickle.load(f)
        print("Tabela Q carregada com sucesso do arquivo.")
        return True
    return False

def show_board(board):
    tabuleiro = f"""
      |     |
    {board[0]} |   {board[1]} |   {board[2]}
  ____|_____|____
      |     |
    {board[3]} |   {board[4]} |   {board[5]}
  ____|_____|____
      |     |
    {board[6]} |   {board[7]} |   {board[8]}
      |     |
    """
    print(tabuleiro)

def get_player_move(board):
    while True:
        try:
            jogada_str = input('Digite a posição da sua jogada (1 a 9) e pressione Enter: ')
            jogada_idx = int(jogada_str) - 1
            if jogada_idx in range(9) and board[jogada_idx] == ' ':
                return jogada_idx
            else:
                print('\nJogada inválida! Posição ocupada ou número fora do intervalo.\n')
        except ValueError:
            print('\nValor digitado inválido. Digite um número inteiro de 1 a 9!\n')

def get_pc_move(board, pc_char):
    state = get_state(board)
    print('Deixe-me pensar na minha jogada...')
    sleep(1.5)

    if state in q_table:
        q_values = q_table[state]
        valid_moves = {k: v for k, v in q_values.items() if board[k] == ' '}
        if valid_moves:
            best_move = max(valid_moves, key=valid_moves.get)
            print(f'\nEu jogo na posição {best_move + 1}!')
            return best_move
    
    open_spots = [i for i, spot in enumerate(board) if spot == ' ']
    if open_spots:
        move = random.choice(open_spots)
        print(f'\nEu jogo na posição {move + 1}!')
        return move
    
    return -1

# --- Lógica principal do jogo ---
if __name__ == "__main__":
    def train_and_save_default_agent():
        global q_table
        print("Nenhuma Q-Table encontrada. Treinando agente padrão...")
        trained_q_table, _ = train_q_agent_with_evaluation(
            alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=50000
        )
        q_table = trained_q_table
        
        with open(Q_TABLE_FILE, 'wb') as f:
            pickle.dump(q_table, f)
        print(f"Agente padrão treinado e salvo em '{Q_TABLE_FILE}'.")
        return q_table

    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, 'rb') as f:
            q_table = pickle.load(f)
        print("Tabela Q carregada com sucesso do arquivo.")
    else:
        q_table = train_and_save_default_agent()

    while True:
        j = ''
        primeiro = ''
        board = [' '] * 9
        vencedor = ''

        tabuleiro_inicial = """
    --- COMO JOGAR ---

    Quando for sua vez, digite o número correspondente à posição no tabuleiro para fazer sua jogada nela.

    Por exemplo, digamos que você queira jogar no centro, então você digita 5.

         |     |    
      1  |  2  |  3  
    ____|_____|____
         |     |    
      4  |  5  |  6  
    ____|_____|____
         |     |    
      7  |  8  |  9  
         |     |    
        """
        print(tabuleiro_inicial)

        print('Você quer ser o X (xis) ou a O (bola)?', end=' ')
        while j not in ('O', 'X'):
            j = str(input('Digite X ou O e pressione Enter para escolher: ')).strip().upper()
            if j not in ('O', 'X'):
                print('\nEscolha inválida!\n')

        adv = 'O' if j == 'X' else 'X'
        print(f'\nEntão eu fico com o {adv}.')

        print('\nQuem joga primeiro?', end=' ')
        while primeiro not in ('EU', 'PC'):
            instr = 'Digite EU e pressione Enter para você começar, ou digite PC e pressione Enter para eu começar: '
            primeiro = str(input(instr)).strip().upper()
            if primeiro not in ('EU', 'PC'):
                print('\nEscolha inválida!\n')

        print(f'\nEntão você joga primeiro.\n' if primeiro == 'EU' else '\nEntão eu jogo primeiro.\n')

        show_board(board)

        turnos = 0
        while True:
            if primeiro == 'EU':
                player_move_idx = get_player_move(board)
                board[player_move_idx] = j
                show_board(board)
                if check_winner(board, j):
                    print('VOCÊ GANHOU!\n')
                    pts_jogador += 1
                    vencedor = 'EU'
                    break
                
                turnos += 1
                if turnos == 5:
                    print('NÓS EMPATAMOS!\n')
                    vencedor = 'EMPATE'
                    break

                pc_move_idx = get_pc_move(board, adv)
                board[pc_move_idx] = adv
                show_board(board)
                if check_winner(board, adv):
                    print('EU GANHEI!\n')
                    pts_pc += 1
                    vencedor = 'PC'
                    break
                
            elif primeiro == 'PC':
                pc_move_idx = get_pc_move(board, adv)
                board[pc_move_idx] = adv
                show_board(board)
                if check_winner(board, adv):
                    print('EU GANHEI!\n')
                    pts_pc += 1
                    vencedor = 'PC'
                    break

                turnos += 1
                if turnos == 5 and not check_winner(board, adv):
                    print('NÓS EMPATAMOS!\n')
                    vencedor = 'EMPATE'
                    break

                player_move_idx = get_player_move(board)
                board[player_move_idx] = j
                show_board(board)
                if check_winner(board, j):
                    print('VOCÊ GANHOU!\n')
                    pts_jogador += 1
                    vencedor = 'EU'
                    break
                
        print('-------- PLACAR --------')
        print(f'Você: {pts_jogador} | Computador: {pts_pc}')
        print('------------------------')

        while True:
            reiniciar = input('\nQuer jogar de novo? Digite S para sim ou N para não: ').lower()
            if reiniciar in ('s', 'n'):
                break
            print('\nResposta inválida!')

        if reiniciar == 's':
            print('\n-----------------------------------------------------')
            continue
        else:
            sys.exit(0)
