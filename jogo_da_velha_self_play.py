import sys
import os
import pickle
import random
from time import sleep

pts_jogador = 0
pts_pc = 0

Q_TABLE_FILE = 'q_table.pkl'

ALPHA = 0.3    # Taxa de aprendizado
GAMMA = 0.9     # Fator de desconto para recompensas futuras
EPSILON = 0.9   # Probabilidade de explorar
NUM_EPISODES = 200000  # Número de jogos de treinamento

q_table = {}

def get_state(board, player_mark):
    return (tuple(board), player_mark)

def train_q_agent():
    global q_table
    print("Treinando o agente com self-play... Por favor, aguarde.")
    
    initial_epsilon = 1.0
    final_epsilon = 0.01
    epsilon_decay = (initial_epsilon - final_epsilon) / NUM_EPISODES

    for episode in range(NUM_EPISODES):
        board = [' '] * 9
        done = False
        current_player = 'X'
        last_move_info = {'X': None, 'O': None} 

        epsilon = initial_epsilon - (episode * epsilon_decay)

        while not done:
            state = get_state(board, current_player)
            
            if state not in q_table:
                possible_moves = [i for i, spot in enumerate(board) if spot == ' ']
                q_table[state] = {move: 0.0 for move in possible_moves}

            if random.uniform(0, 1) < epsilon or not q_table[state]:
                action = random.choice([i for i, spot in enumerate(board) if spot == ' '])
            else:
                action = max(q_table[state], key=q_table[state].get)
            
            previous_player = 'O' if current_player == 'X' else 'X'
            if last_move_info[previous_player] is not None:
                prev_state, prev_action = last_move_info[previous_player]
                reward = 0 
                max_q_next_state = max(q_table[state].values()) if q_table[state] else 0.0    
                old_q_value = q_table[prev_state].get(prev_action, 0.0)
                new_q_value = old_q_value + ALPHA * (reward + GAMMA * (-max_q_next_state) - old_q_value)
                q_table[prev_state][prev_action] = new_q_value
                
            board[action] = current_player
            last_move_info[current_player] = (state, action)

            if check_winner(board, current_player):
                reward = 1 
                done = True
            elif ' ' not in board:
                reward = 0.5 
                done = True

            if done:
                winner_state, winner_action = last_move_info[current_player]
                old_q = q_table[winner_state].get(winner_action, 0.0)
                q_table[winner_state][winner_action] = old_q + ALPHA * (reward - old_q)
                loser = 'O' if current_player == 'X' else 'X'
                if last_move_info[loser] is not None:
                    loser_state, loser_action = last_move_info[loser]
                    old_q = q_table[loser_state].get(loser_action, 0.0)
                    q_table[loser_state][loser_action] = old_q + ALPHA * (-reward - old_q)
                break
            current_player = 'O' if current_player == 'X' else 'X'

        if (episode + 1) % (NUM_EPISODES // 10) == 0:
            print(f"Treinamento... {((episode + 1)/NUM_EPISODES)*100:.0f}% concluído")

    print("Treinamento concluído. O agente agora está pronto para jogar!")
    save_q_table()

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
        return 1  # Vitoria
    elif check_winner(board, 'O' if player == 'X' else 'X'):
        return -1 # Derrota
    elif ' ' not in board:
        return 0.5 # Empate
    else:
        return 0 # Jogo continua

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

def get_pc_move(board):
    state = get_state(board, 'O') 
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

if not load_q_table():
    train_q_agent()

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
_____|_____|____
     |     |    
  4  |  5  |  6  
_____|_____|____
     |     |    
  7  |  8  |  9  
     |     |    
    """
    print(tabuleiro_inicial)
    adv = 'O'
    j = 'X'

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
                
            pc_move_idx = get_pc_move(board)
            board[pc_move_idx] = adv
            show_board(board)
            if check_winner(board, adv):
                print('EU GANHEI!\n')
                pts_pc += 1
                vencedor = 'PC'
                break
        
        elif primeiro == 'PC':
            pc_move_idx = get_pc_move(board)
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
