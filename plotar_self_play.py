from matplotlib import pyplot as plt
import numpy as np
import random

def check_winner(board, player):
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Linhas
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Colunas
        (0, 4, 8), (2, 4, 6)             # Diagonais
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

def evaluate_agent_self_play(q_table, num_games=1000):
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(num_games):
        board = [' '] * 9
        done = False
        
        while not done:
            state = (tuple(board), 'X')
            
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


def train_q_agent_with_evaluation_self_play(alpha, gamma, initial_epsilon, num_episodes):
    q_table = {}
    final_epsilon = 0.01
    epsilon_decay = (initial_epsilon - final_epsilon) / num_episodes
    
    performance_history = {'episodes': [], 'wins': [], 'losses': [], 'draws': []}
    evaluation_interval = num_episodes // 100 

    for episode in range(1, num_episodes + 1):
        board = [' '] * 9
        done = False
        current_player = 'X'
        last_move_info = {'X': None, 'O': None}

        epsilon = initial_epsilon - (episode * epsilon_decay)

        while not done:
            state = (tuple(board), current_player)
            
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
                new_q_value = old_q_value + alpha * (reward + gamma * (-max_q_next_state) - old_q_value)
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
                q_table[winner_state][winner_action] = old_q + alpha * (reward - old_q)
                loser = 'O' if current_player == 'X' else 'X'
                if last_move_info[loser] is not None:
                    loser_state, loser_action = last_move_info[loser]
                    old_q = q_table[loser_state].get(loser_action, 0.0)
                    q_table[loser_state][loser_action] = old_q + alpha * (-reward - old_q)
                break
            current_player = 'O' if current_player == 'X' else 'X'
            
        if episode % evaluation_interval == 0:
            print(f"Episódio {episode}/{num_episodes} - Avaliando...")
            performance = evaluate_agent_self_play(q_table, num_games=1000)
            performance_history['episodes'].append(episode)
            performance_history['wins'].append(performance['wins'])
            performance_history['losses'].append(performance['losses'])
            performance_history['draws'].append(performance['draws'])
            
    return q_table, performance_history

def apply_moving_average(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results_comparison(histories, parameter_name, values):
    plt.figure(figsize=(12, 8))
    plt.title(f'Taxa de Vitórias vs. Episódios (variando {parameter_name})', fontsize=16)
    
    for i, history in enumerate(histories):
        if not history['wins']: 
            print(f"Atenção: Nenhum dado de 'wins' para {parameter_name} = {values[i]}. Skipping plot.")
            continue
        win_rate_smooth = apply_moving_average(history['wins'])
        episodes_smooth = history['episodes'][len(history['episodes']) - len(win_rate_smooth):]
        plt.plot(episodes_smooth, win_rate_smooth, label=f'{parameter_name} = {values[i]}')

    plt.xlabel('Episódios de Treinamento', fontsize=12)
    plt.ylabel('Taxa de Vitórias (Média Móvel de 50 episódios)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05) 
    plt.savefig(f'comparacao_taxa_vitoria_{parameter_name}_self_play.png')
    plt.show()

def plot_win_draw_loss_analysis(history, title):
    plt.figure(figsize=(12, 8))
    plt.title(f'Análise de Resultados: {title}', fontsize=16)

    if not history['wins'] or not history['draws'] or not history['losses']:
        print(f"Atenção: Dados insuficientes para análise de vitórias/empates/derrotas para '{title}'. Skipping plot.")
        plt.close() 
        return
    
    win_rate_smooth = apply_moving_average(history['wins'])
    draw_rate_smooth = apply_moving_average(history['draws'])
    loss_rate_smooth = apply_moving_average(history['losses'])
    
    min_len = min(len(win_rate_smooth), len(draw_rate_smooth), len(loss_rate_smooth))
    episodes_smooth = history['episodes'][len(history['episodes']) - min_len:]
    
    plt.plot(episodes_smooth, win_rate_smooth[-min_len:], label='Vitórias', color='green')
    plt.plot(episodes_smooth, draw_rate_smooth[-min_len:], label='Empates', color='orange')
    plt.plot(episodes_smooth, loss_rate_smooth[-min_len:], label='Derrotas', color='red')
    
    plt.xlabel('Episódios de Treinamento', fontsize=12)
    plt.ylabel('Taxa de Resultados (Média Móvel de 50 episódios)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.savefig(f'analise_vitorias_empates_derrotas_{title.replace(" ", "_")}_self_play.png')
    plt.show()

def plot_final_performance_comparison(histories, parameter_name, values):
    final_win_rates = []
    labels = []
    for i, history in enumerate(histories):
        if not history['wins']:
            print(f"Atenção: Nenhum dado de 'wins' para {parameter_name} = {values[i]} para cálculo de desempenho final.")
            continue
        
        num_episodes = len(history['wins'])
        if num_episodes == 0:
            final_win_rates.append(0.0) 
        else:
            start_index = int(num_episodes * 0.8)
            start_index = max(0, min(start_index, num_episodes - 1))
            avg_win_rate = np.mean(history['wins'][start_index:])
            final_win_rates.append(avg_win_rate)
        labels.append(str(values[i]))
        
    if not final_win_rates:
        print(f"Atenção: Nenhuma taxa de vitória final calculada para {parameter_name}. Skipping plot.")
        plt.close()
        return

    plt.figure(figsize=(12, 8))
    plt.title(f'Comparativo de Desempenho Final (variando {parameter_name})', fontsize=16)
    
    bars = plt.bar(labels, final_win_rates, color='skyblue')
    plt.xlabel(f'Valor do Hiperparâmetro {parameter_name}', fontsize=12)
    plt.ylabel('Taxa de Vitória Média Final', fontsize=12)
    plt.ylim(0, 1.1)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2%}', va='bottom', ha='center', fontsize=10)
        
    plt.savefig(f'comparacao_desempenho_final_{parameter_name}_self_play.png')
    plt.show()

if __name__ == "__main__":
    NUM_EPISODES = 50000 
    
    print("--- Iniciando Experimento com Agente Self-Play: Variando ALPHA ---")
    alphas = [0.1, 0.3, 0.5]
    histories_alpha = []
    for alpha in alphas:
        print(f"\nTreinando com alpha = {alpha}...")
        _, history = train_q_agent_with_evaluation_self_play(
            alpha=alpha, gamma=0.9, initial_epsilon=1.0, num_episodes=NUM_EPISODES
        )
        histories_alpha.append(history)
    
    plot_results_comparison(histories_alpha, "Alpha", alphas)
    plot_final_performance_comparison(histories_alpha, "Alpha", alphas)
    
    best_index_alpha = np.argmax([np.mean(h['wins'][-int(len(h['wins'])*0.2):]) for h in histories_alpha if h['wins']])
    plot_win_draw_loss_analysis(histories_alpha[best_index_alpha], f"Melhor Alpha ({alphas[best_index_alpha]})")

    print("\n--- Iniciando Experimento com Agente Self-Play: Variando GAMMA ---")
    gammas = [0.8, 0.9, 0.99]
    histories_gamma = []
    for gamma in gammas:
        print(f"\nTreinando com gamma = {gamma}...")
        _, history = train_q_agent_with_evaluation_self_play(
            alpha=0.3, gamma=gamma, initial_epsilon=1.0, num_episodes=NUM_EPISODES
        )
        histories_gamma.append(history)
    
    plot_results_comparison(histories_gamma, "Gamma", gammas)
    plot_final_performance_comparison(histories_gamma, "Gamma", gammas)

    best_index_gamma = np.argmax([np.mean(h['wins'][-int(len(h['wins'])*0.2):]) for h in histories_gamma if h['wins']])
    plot_win_draw_loss_analysis(histories_gamma[best_index_gamma], f"Melhor Gamma ({gammas[best_index_gamma]})")

    print("\n--- Iniciando Experimento com Agente Self-Play: Variando EPSILON (Inicial) ---")
    epsilons = [0.8, 1.0] # Epsilon inicial para decaimento
    histories_epsilon = []
    for epsilon in epsilons:
        print(f"\nTreinando com epsilon inicial = {epsilon}...")
        _, history = train_q_agent_with_evaluation_self_play(
            alpha=0.3, gamma=0.9, initial_epsilon=epsilon, num_episodes=NUM_EPISODES
        )
        histories_epsilon.append(history)

    plot_results_comparison(histories_epsilon, "Epsilon", epsilons)
    plot_final_performance_comparison(histories_epsilon, "Epsilon", epsilons)

    best_index_epsilon = np.argmax([np.mean(h['wins'][-int(len(h['wins'])*0.2):]) for h in histories_epsilon if h['wins']])
    plot_win_draw_loss_analysis(histories_epsilon[best_index_epsilon], f"Melhor Epsilon ({epsilons[best_index_epsilon]})")