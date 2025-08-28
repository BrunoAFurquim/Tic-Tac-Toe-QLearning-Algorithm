from matplotlib import pyplot as plt
import numpy as np
from jogo_da_velha_random import train_q_agent_with_evaluation

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
    plt.savefig(f'comparacao_taxa_vitoria_{parameter_name}.png')
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
    plt.savefig(f'analise_vitorias_empates_derrotas_{title.replace(" ", "_")}.png')
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
        
    plt.savefig(f'comparacao_desempenho_final_{parameter_name}.png')
    plt.show()

if __name__ == "__main__":
    NUM_EPISODES = 20000 
    print("--- Iniciando Experimento: Variando ALPHA ---")
    alphas = [0.1, 0.4, 0.8]
    histories_alpha = []
    for alpha in alphas:
        print(f"\nTreinando com alpha = {alpha}...")
        _, history = train_q_agent_with_evaluation(
            alpha=alpha, gamma=0.9, epsilon=0.1, num_episodes=NUM_EPISODES
        )
        histories_alpha.append(history)
    
    plot_results_comparison(histories_alpha, "Alpha", alphas)
    plot_final_performance_comparison(histories_alpha, "Alpha", alphas)
    
    best_index_alpha = -1
    max_final_win_rate_alpha = -1
    for i, h in enumerate(histories_alpha):
        if h['wins']:
            current_avg = np.mean(h['wins'][-int(len(h['wins'])*0.2):])
            if current_avg > max_final_win_rate_alpha:
                max_final_win_rate_alpha = current_avg
                best_index_alpha = i
    
    if best_index_alpha != -1:
        plot_win_draw_loss_analysis(histories_alpha[best_index_alpha], f"Melhor Alpha ({alphas[best_index_alpha]})")
    else:
        print("Não foi possível determinar o melhor Alpha para a análise detalhada.")


    print("\n--- Iniciando Experimento: Variando GAMMA ---")
    gammas = [0.7, 0.9, 0.99]
    histories_gamma = []
    for gamma in gammas:
        print(f"\nTreinando com gamma = {gamma}...")
        _, history = train_q_agent_with_evaluation(
            alpha=0.1, gamma=gamma, epsilon=0.1, num_episodes=NUM_EPISODES
        )
        histories_gamma.append(history)
    
    plot_results_comparison(histories_gamma, "Gamma", gammas)
    plot_final_performance_comparison(histories_gamma, "Gamma", gammas)

    best_index_gamma = -1
    max_final_win_rate_gamma = -1
    for i, h in enumerate(histories_gamma):
        if h['wins']:
            current_avg = np.mean(h['wins'][-int(len(h['wins'])*0.2):])
            if current_avg > max_final_win_rate_gamma:
                max_final_win_rate_gamma = current_avg
                best_index_gamma = i
    
    if best_index_gamma != -1:
        plot_win_draw_loss_analysis(histories_gamma[best_index_gamma], f"Melhor Gamma ({gammas[best_index_gamma]})")
    else:
        print("Não foi possível determinar o melhor Gamma para a análise detalhada.")

    print("\n--- Iniciando Experimento: Variando EPSILON ---")
    epsilons = [0.05, 0.1, 0.3]
    histories_epsilon = []
    for epsilon in epsilons:
        print(f"\nTreinando com epsilon = {epsilon}...")
        _, history = train_q_agent_with_evaluation(
            alpha=0.1, gamma=0.9, epsilon=epsilon, num_episodes=NUM_EPISODES
        )
        histories_epsilon.append(history)

    plot_results_comparison(histories_epsilon, "Epsilon", epsilons)
    plot_final_performance_comparison(histories_epsilon, "Epsilon", epsilons)

    best_index_epsilon = -1
    max_final_win_rate_epsilon = -1
    for i, h in enumerate(histories_epsilon):
        if h['wins']: 
            current_avg = np.mean(h['wins'][-int(len(h['wins'])*0.2):])
            if current_avg > max_final_win_rate_epsilon:
                max_final_win_rate_epsilon = current_avg
                best_index_epsilon = i
    
    if best_index_epsilon != -1:
        plot_win_draw_loss_analysis(histories_epsilon[best_index_epsilon], f"Melhor Epsilon ({epsilons[best_index_epsilon]})")
    else:
        print("Não foi possível determinar o melhor Epsilon para a análise detalhada.")