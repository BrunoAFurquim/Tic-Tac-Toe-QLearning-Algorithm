import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pickle
import os
import random
from time import sleep
import threading
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow não está instalado. Background será cor sólida.")

# Variáveis globais
pts_jogador = 0
pts_pc = 0
Q_TABLE_FILE = 'q_table.pkl'

# Parâmetros Q-Learning - Performance Máxima OTIMIZADA
# Alpha: Taxa de aprendizado - controla o quanto o agente aprende com cada experiência
# Valores altos (0.5) = aprendizado rápido, valores baixos (0.1) = aprendizado conservador
ALPHA = 0.5

# Gamma: Fator de desconto - determina a importância de recompensas futuras
# Valores próximos de 1.0 (0.95) = foca no longo prazo, valores baixos = foca no curto prazo
GAMMA = 0.95

# Epsilon: Taxa de exploração - probabilidade de fazer jogadas aleatórias para explorar
# Valores altos (0.4) = mais exploração, valores baixos (0.1) = mais exploração das estratégias aprendidas
EPSILON = 0.4

# Número de episódios de treinamento - mais episódios = agente mais experiente
NUM_EPISODES = 100000

q_table = {}

# Função para deletar q_table.pkl automaticamente - REMOVIDA
# O arquivo q_table.pkl agora é preservado até ser regenerado pelo treinamento

def get_state(board):
    """
    Converte o estado atual do tabuleiro em uma tupla.
    No Q-Learning, o ESTADO representa a situação atual do ambiente.
    Aqui, o estado é a configuração atual do tabuleiro (onde estão X, O e espaços vazios).
    """
    return tuple(board)

def check_winner(board, player):
    """
    Verifica se um jogador ganhou o jogo.
    Esta função é crucial para o Q-Learning pois determina quando o episódio termina
    e qual RECOMPENSA o agente deve receber.
    """
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Linhas
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Colunas
        (0, 4, 8), (2, 4, 6)              # Diagonais
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

def get_reward(board, player, action):
    """
    SISTEMA DE RECOMPENSAS MELHORADO para Q-Learning
    
    Recompensas baseadas em:
    1. Resultado do jogo (vitória/derrota/empate)
    2. Jogadas estratégicas (ganhar/bloquear)
    3. Posicionamento inteligente (centro/cantos)
    
    RECOMPENSAS:
    +10.0: Vitória imediata
    +5.0: Bloqueou vitória do oponente  
    +2.0: Jogada no centro (estratégica)
    +1.0: Jogada nos cantos (boa posição)
    +0.0: Jogada neutra
    -10.0: Derrota
    -0.5: Empate
    """
    
    # Verificar resultado final
    if check_winner(board, player):
        return 10.0  # Vitória - recompensa máxima
    elif check_winner(board, 'O' if player == 'X' else 'X'):
        return -10.0  # Derrota - penalização máxima
    elif ' ' not in board:
        return -0.5   # Empate - penalização leve
    
    # Recompensas por jogadas estratégicas (durante o jogo)
    reward = 0.0
    
    # Simular se a jogada bloqueia vitória do oponente
    opponent = 'O' if player == 'X' else 'X'
    board_copy = board.copy()
    board_copy[action] = ' '  # Remover a jogada atual
    
    if can_win(board_copy, opponent) == action:
        reward += 5.0  # Bloqueou vitória do oponente
    
    # Recompensas posicionais
    if action == 4:  # Centro
        reward += 2.0
    elif action in [0, 2, 6, 8]:  # Cantos
        reward += 1.0
    # Bordas (1,3,5,7) recebem 0 - neutro
    
    return reward

def train_q_agent_with_evaluation(alpha, gamma, epsilon, num_episodes, progress_callback=None):
    """
    FUNÇÃO PRINCIPAL DE TREINAMENTO Q-LEARNING COM SELF-PLAY
    
    Esta função implementa o algoritmo Q-Learning para ensinar um agente a jogar Jogo da Velha
    usando SELF-PLAY (ambos jogadores usam Q-Learning).
    
    PARÂMETROS:
    - alpha: Taxa de aprendizado (0.0 - 1.0) - controla quanto o agente aprende com cada experiência
    - gamma: Fator de desconto (0.0 - 1.0) - importância de recompensas futuras vs imediatas  
    - epsilon: Taxa de exploração (0.0 - 1.0) - probabilidade de fazer jogadas aleatórias
    - num_episodes: Número de jogos de treinamento
    - progress_callback: Função para reportar progresso (opcional)
    
    ALGORITMO Q-LEARNING:
    Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
    
    Onde:
    - Q(s,a) = Valor da ação 'a' no estado 's'
    - α (alpha) = Taxa de aprendizado
    - r = Recompensa recebida
    - γ (gamma) = Fator de desconto
    - max(Q(s',a')) = Melhor valor possível no próximo estado
    """
    
    # Q-TABLE: Dicionário que mapeia estados para valores de ações
    # Estrutura: {(estado, jogador): {ação: valor_q, ...}, ...}
    q_table = {}
    
    # EPSILON DECAY: Reduz exploração ao longo do tempo
    final_epsilon = 0.01
    epsilon_decay = (epsilon - final_epsilon) / num_episodes
    
    # LOOP PRINCIPAL DE TREINAMENTO
    for episode in range(1, num_episodes + 1):
        # INICIALIZAR EPISÓDIO
        board = [' '] * 9  # Tabuleiro vazio
        done = False  # Flag de fim de jogo
        current_player = 'X'  # Jogador atual
        last_move_info = {'X': None, 'O': None}  # Última jogada de cada jogador
        
        # Epsilon decrescente
        current_epsilon = epsilon - (episode * epsilon_decay)
        current_epsilon = max(current_epsilon, final_epsilon)
        
        # LOOP DO JOGO (até terminar) - SELF-PLAY
        while not done:
            # ESTADO ATUAL (incluindo jogador)
            state = (get_state(board), current_player)
            
            # INICIALIZAR ESTADO NA Q-TABLE (se não existir)
            if state not in q_table:
                # Criar entradas para todas as ações válidas (posições vazias)
                q_table[state] = {i: 0.0 for i in range(9) if board[i] == ' '}

            # ESTRATÉGIA EPSILON-GREEDY
            if random.uniform(0, 1) < current_epsilon or not q_table[state]:
                # EXPLORAÇÃO: Escolher ação aleatória
                possible_moves = [i for i, spot in enumerate(board) if spot == ' ']
                if not possible_moves: break
                action = random.choice(possible_moves)
            else:
                # EXPLORAÇÃO: Escolher melhor ação conhecida
                action = max(q_table[state], key=q_table[state].get)
            
            # ATUALIZAR Q-VALUE DO JOGADOR ANTERIOR (se existir)
            previous_player = 'O' if current_player == 'X' else 'X'
            if last_move_info[previous_player] is not None:
                prev_state, prev_action = last_move_info[previous_player]
                reward = 0  # Recompensa neutra durante o jogo
                max_q_next_state = max(q_table[state].values()) if q_table[state] else 0.0    
                old_q_value = q_table[prev_state].get(prev_action, 0.0)
                # Atualização Q-Learning com recompensa negativa (perspectiva do oponente)
                new_q_value = old_q_value + alpha * (reward + gamma * (-max_q_next_state) - old_q_value)
                q_table[prev_state][prev_action] = new_q_value

            # EXECUTAR AÇÃO
            board[action] = current_player
            last_move_info[current_player] = (state, action)
            
            # VERIFICAR SE O JOGO TERMINOU
            if check_winner(board, current_player):
                reward = 1  # Vitória
                done = True
            elif ' ' not in board:
                reward = 0.5  # Empate
                done = True
            
            # ATUALIZAR Q-VALUES FINAIS
            if done:
                # Atualizar Q-value do jogador vencedor/empatador
                winner_state, winner_action = last_move_info[current_player]
                old_q = q_table[winner_state].get(winner_action, 0.0)
                q_table[winner_state][winner_action] = old_q + alpha * (reward - old_q)
                
                # Atualizar Q-value do jogador perdedor (se existir)
                loser = 'O' if current_player == 'X' else 'X'
                if last_move_info[loser] is not None:
                    loser_state, loser_action = last_move_info[loser]
                    old_q = q_table[loser_state].get(loser_action, 0.0)
                    q_table[loser_state][loser_action] = old_q + alpha * (-reward - old_q)
                break
            
            # TROCAR JOGADOR
            current_player = 'O' if current_player == 'X' else 'X'
        
        # CALLBACK DE PROGRESSO
        if progress_callback and episode % (num_episodes // 100) == 0:
            progress_callback(episode, num_episodes)
            
    return q_table

def save_q_table():
    with open(Q_TABLE_FILE, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"✅ Q-Table salva em '{Q_TABLE_FILE}' com sucesso!")

def load_q_table():
    global q_table
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, 'rb') as f:
            q_table = pickle.load(f)
        return True
    return False

def can_win(board, player):
    """Verifica se o jogador pode ganhar na próxima jogada"""
    for i in range(9):
        if board[i] == ' ':
            board[i] = player
            if check_winner(board, player):
                board[i] = ' '  # Desfazer
                return i
            board[i] = ' '  # Desfazer
    return -1

def get_pc_move(board):
    """
    FUNÇÃO DE DECISÃO DO AGENTE Q-LEARNING MELHORADA
    
    Estratégia híbrida que combina:
    1. Heurísticas básicas (ganhar/bloquear)
    2. Conhecimento da Q-Table
    3. Fallback inteligente
    
    ORDEM DE PRIORIDADE:
    1. Se pode ganhar imediatamente → GANHAR
    2. Se oponente pode ganhar → BLOQUEAR
    3. Usar Q-Table para escolha inteligente
    4. Fallback: centro > cantos > bordas
    """
    
    # PRIORIDADE 1: GANHAR IMEDIATAMENTE
    winning_move = can_win(board, 'O')
    if winning_move != -1:
        return winning_move
    
    # PRIORIDADE 2: BLOQUEAR VITÓRIA DO OPONENTE
    blocking_move = can_win(board, 'X')
    if blocking_move != -1:
        return blocking_move
    
    # PRIORIDADE 3: USAR Q-TABLE (conhecimento aprendido)
    # Novo formato de estado: (tabuleiro, jogador)
    state = (get_state(board), 'O')
    if state in q_table:
        q_values = q_table[state]
        valid_moves = {k: v for k, v in q_values.items() if board[k] == ' '}
        
        if valid_moves:
            # Escolher ação com maior valor Q
            best_move = max(valid_moves, key=valid_moves.get)
            return best_move
    
    # PRIORIDADE 4: FALLBACK INTELIGENTE
    open_spots = [i for i, spot in enumerate(board) if spot == ' ']
    if not open_spots:
        return -1
    
    # Preferência: centro > cantos > bordas
    center = 4
    corners = [0, 2, 6, 8]
    edges = [1, 3, 5, 7]
    
    # Tentar centro primeiro
    if center in open_spots:
        return center
    
    # Tentar cantos
    available_corners = [pos for pos in corners if pos in open_spots]
    if available_corners:
        return random.choice(available_corners)
    
    # Por último, bordas
    available_edges = [pos for pos in edges if pos in open_spots]
    if available_edges:
        return random.choice(available_edges)
    
    # Fallback final
    return random.choice(open_spots)

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Jogo da Velha - Q-Learning")
        self.root.geometry("500x600")
        
        # Configurar background
        self.setup_background()
        
        # Variáveis do jogo
        self.board = [' '] * 9
        self.player_symbol = 'X'
        self.pc_symbol = 'O'
        self.player_turn = True
        self.game_over = False
        self.pts_jogador = 0
        self.pts_pc = 0
        
        self.setup_ui()
        self.check_and_load_qtable()
    
    def setup_background(self):
        """
        Configura o background da janela principal.
        Tenta carregar background.png, se não conseguir usa cor sólida.
        """
        if PIL_AVAILABLE and os.path.exists("background.png"):
            try:
                # Carregar e redimensionar a imagem
                image = Image.open("background.png")
                # Redimensionar para o tamanho da janela
                image = image.resize((500, 600), Image.Resampling.LANCZOS)
                self.bg_image = ImageTk.PhotoImage(image)
                
                # Criar label para o background
                self.bg_label = tk.Label(self.root, image=self.bg_image)
                self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
                
                print("✅ Background carregado com sucesso!")
                
            except Exception as e:
                print(f"❌ Erro ao carregar background: {e}")
                self.root.configure(bg='#2c3e50')
        else:
            # Fallback para cor sólida
            self.root.configure(bg='#2c3e50')
            if not PIL_AVAILABLE:
                print("⚠️  Pillow não disponível. Usando cor sólida.")
            else:
                print("⚠️  background.png não encontrado. Usando cor sólida.")
    
    def setup_ui(self):
        # Definir cores para interface com background
        if hasattr(self, 'bg_image'):
            # Com background de imagem - usar cores semi-transparentes
            frame_bg = '#1a1a1a'  # Preto semi-transparente
            text_bg = '#000000'   # Preto para contraste
            text_fg = '#ffffff'   # Branco para boa legibilidade
        else:
            # Sem background - usar cores originais
            frame_bg = '#2c3e50'
            text_bg = '#2c3e50'
            text_fg = '#ecf0f1'
        
        # Título
        title_label = tk.Label(self.root, text="🎮 Jogo da Velha - Q-Learning", 
                              font=('Arial', 20, 'bold'), 
                              bg=text_bg, fg=text_fg,
                              relief='raised', bd=2)
        title_label.pack(pady=10)
        
        # Placar
        self.score_frame = tk.Frame(self.root, bg=frame_bg, relief='raised', bd=2)
        self.score_frame.pack(pady=10)
        
        self.score_label = tk.Label(self.score_frame, 
                                   text=f"🏆 Você: {self.pts_jogador} | 🤖 PC: {self.pts_pc}",
                                   font=('Arial', 14, 'bold'), 
                                   bg=text_bg, fg=text_fg)
        self.score_label.pack(padx=10, pady=5)
        
        # Tabuleiro
        self.board_frame = tk.Frame(self.root, bg=frame_bg, relief='raised', bd=3)
        self.board_frame.pack(pady=20)
        
        self.buttons = []
        for i in range(9):
            row = i // 3
            col = i % 3
            
            btn = tk.Button(self.board_frame, text=' ', 
                           font=('Arial', 20, 'bold'),
                           width=4, height=2,
                           bg='#34495e', fg='#ecf0f1',
                           activebackground='#3498db',
                           relief='raised', bd=2,
                           command=lambda idx=i: self.player_move(idx))
            btn.grid(row=row, column=col, padx=3, pady=3)
            self.buttons.append(btn)
        
        # Status do jogo
        self.status_label = tk.Label(self.root, text="🎯 Sua vez! Clique em uma posição.", 
                                    font=('Arial', 12, 'bold'), 
                                    bg=text_bg, fg=text_fg,
                                    relief='raised', bd=2)
        self.status_label.pack(pady=10)
        
        # Botões de controle
        control_frame = tk.Frame(self.root, bg=frame_bg, relief='raised', bd=2)
        control_frame.pack(pady=10)
        
        self.new_game_btn = tk.Button(control_frame, text="🔄 Novo Jogo", 
                                     font=('Arial', 12, 'bold'),
                                     bg='#27ae60', fg='white',
                                     relief='raised', bd=2,
                                     command=self.new_game)
        self.new_game_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.train_btn = tk.Button(control_frame, text="🧠 Treinar Agente", 
                                  font=('Arial', 12, 'bold'),
                                  bg='#e74c3c', fg='white',
                                  relief='raised', bd=2,
                                  command=self.show_training_dialog)
        self.train_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.quit_btn = tk.Button(control_frame, text="🚪 Sair", 
                                 font=('Arial', 12, 'bold'),
                                 bg='#95a5a6', fg='white',
                                 relief='raised', bd=2,
                                 command=self.quit_game)
        self.quit_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    def check_and_load_qtable(self):
        """Verifica se existe q_table.pkl e carrega ou treina um agente padrão"""        
        if not load_q_table():
            self.status_label.config(text="Treinando agente padrão... Aguarde.")
            self.root.update()
            
            # Treina agente padrão em thread separada
            threading.Thread(target=self.train_default_agent, daemon=True).start()
    
    def train_default_agent(self):
        """Treina um agente padrão em background"""
        global q_table
        
        def progress_callback(episode, total):
            progress = int((episode / total) * 100)
            self.root.after(0, lambda: self.status_label.config(
                text=f"Treinando agente padrão... {progress}%"))
        
        q_table = train_q_agent_with_evaluation(
            alpha=0.4, gamma=0.9, epsilon=0.5, 
            num_episodes=50000, progress_callback=progress_callback
        )
        
        save_q_table()
        
        self.root.after(0, lambda: self.status_label.config(
            text="Agente treinado! Sua vez, clique em uma posição."))
    
    def show_training_dialog(self):
        """Mostra diálogo para selecionar perfil de treinamento"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Treinar Agente Q-Learning")
        dialog.geometry("500x650")
        dialog.configure(bg='#2c3e50')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Centralizar diálogo
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, 
                                   self.root.winfo_rooty() + 50))
        
        tk.Label(dialog, text="Configurar Parâmetros de Treinamento", 
                font=('Arial', 14, 'bold'), bg='#2c3e50', fg='#ecf0f1').pack(pady=10)
        
        # Perfis de dificuldade (para ajustar sliders)
        profiles_frame = tk.Frame(dialog, bg='#2c3e50')
        profiles_frame.pack(pady=10)
        
        tk.Label(profiles_frame, text="Perfis Predefinidos:", 
                font=('Arial', 12, 'bold'), bg='#2c3e50', fg='#ecf0f1').pack()
        
        # Definir perfis (corrigidos para diferentes níveis de dificuldade)
        profiles = {
            "iniciante": {"alpha": 0.3, "gamma": 0.8, "epsilon": 0.8, "episodes": 20000},
            "intermediario": {"alpha": 0.4, "gamma": 0.9, "epsilon": 0.5, "episodes": 50000},
            "avancado": {"alpha": 0.5, "gamma": 0.95, "epsilon": 0.3, "episodes": 100000}
        }
        
        # Parâmetros
        params_frame = tk.Frame(dialog, bg='#2c3e50')
        params_frame.pack(pady=10)
        
        # Alpha
        alpha_frame = tk.Frame(params_frame, bg='#2c3e50')
        alpha_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
        tk.Label(alpha_frame, text="Alpha (Taxa de Aprendizado): ", 
                bg='#2c3e50', fg='#ecf0f1').pack(side=tk.LEFT)
        alpha_var = tk.DoubleVar(value=ALPHA)
        alpha_value_label = tk.Label(alpha_frame, text=f"{ALPHA}", 
                                    bg='#2c3e50', fg='#3498db', font=('Arial', 10, 'bold'))
        alpha_value_label.pack(side=tk.RIGHT)
        alpha_scale = tk.Scale(params_frame, from_=0.01, to=1.0, resolution=0.01, 
                orient=tk.HORIZONTAL, variable=alpha_var, bg='#34495e', fg='#ecf0f1',
                command=lambda v: alpha_value_label.config(text=f"{float(v):.2f}"))
        alpha_scale.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10)
        
        # Gamma
        gamma_frame = tk.Frame(params_frame, bg='#2c3e50')
        gamma_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
        tk.Label(gamma_frame, text="Gamma (Fator de Desconto): ", 
                bg='#2c3e50', fg='#ecf0f1').pack(side=tk.LEFT)
        gamma_var = tk.DoubleVar(value=GAMMA)
        gamma_value_label = tk.Label(gamma_frame, text=f"{GAMMA}", 
                                    bg='#2c3e50', fg='#3498db', font=('Arial', 10, 'bold'))
        gamma_value_label.pack(side=tk.RIGHT)
        gamma_scale = tk.Scale(params_frame, from_=0.1, to=1.0, resolution=0.01, 
                orient=tk.HORIZONTAL, variable=gamma_var, bg='#34495e', fg='#ecf0f1',
                command=lambda v: gamma_value_label.config(text=f"{float(v):.2f}"))
        gamma_scale.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10)
        
        # Epsilon
        epsilon_frame = tk.Frame(params_frame, bg='#2c3e50')
        epsilon_frame.grid(row=4, column=0, columnspan=2, sticky='ew', pady=5)
        tk.Label(epsilon_frame, text="Epsilon (Taxa de Exploração): ", 
                bg='#2c3e50', fg='#ecf0f1').pack(side=tk.LEFT)
        epsilon_var = tk.DoubleVar(value=EPSILON)
        epsilon_value_label = tk.Label(epsilon_frame, text=f"{EPSILON}", 
                                      bg='#2c3e50', fg='#3498db', font=('Arial', 10, 'bold'))
        epsilon_value_label.pack(side=tk.RIGHT)
        epsilon_scale = tk.Scale(params_frame, from_=0.01, to=1.0, resolution=0.01, 
                orient=tk.HORIZONTAL, variable=epsilon_var, bg='#34495e', fg='#ecf0f1',
                command=lambda v: epsilon_value_label.config(text=f"{float(v):.2f}"))
        epsilon_scale.grid(row=5, column=0, columnspan=2, sticky='ew', padx=10)
        
        # Episódios
        episodes_frame = tk.Frame(params_frame, bg='#2c3e50')
        episodes_frame.grid(row=6, column=0, columnspan=2, sticky='ew', pady=5)
        tk.Label(episodes_frame, text="Número de Episódios: ", 
                bg='#2c3e50', fg='#ecf0f1').pack(side=tk.LEFT)
        episodes_var = tk.IntVar(value=NUM_EPISODES)
        episodes_value_label = tk.Label(episodes_frame, text=f"{NUM_EPISODES:,}", 
                                       bg='#2c3e50', fg='#3498db', font=('Arial', 10, 'bold'))
        episodes_value_label.pack(side=tk.RIGHT)
        episodes_scale = tk.Scale(params_frame, from_=10000, to=200000, resolution=10000, 
                orient=tk.HORIZONTAL, variable=episodes_var, bg='#34495e', fg='#ecf0f1',
                command=lambda v: episodes_value_label.config(text=f"{int(float(v)):,}"))
        episodes_scale.grid(row=7, column=0, columnspan=2, sticky='ew', padx=10)
        
        # Barra de progresso
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(dialog, variable=progress_var, maximum=100)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        
        progress_label = tk.Label(dialog, text="", bg='#2c3e50', fg='#ecf0f1')
        progress_label.pack()
        
        def start_training():
            # Desabilitar botão durante treinamento
            train_button.config(state='disabled')
            cancel_button.config(state='disabled')
            
            def progress_callback(episode, total):
                progress = (episode / total) * 100
                progress_var.set(progress)
                progress_label.config(text=f"Treinando... {episode:,}/{total:,} episódios ({progress:.1f}%)")
                dialog.update()
            
            def training_thread():
                global q_table
                try:
                    q_table = train_q_agent_with_evaluation(
                        alpha=alpha_var.get(),
                        gamma=gamma_var.get(),
                        epsilon=epsilon_var.get(),
                        num_episodes=episodes_var.get(),
                        progress_callback=progress_callback
                    )
                    save_q_table()
                    
                    dialog.after(0, lambda: [
                        progress_label.config(text="Treinamento concluído com sucesso!"),
                        train_button.config(state='normal'),
                        cancel_button.config(text="Fechar"),
                        cancel_button.config(state='normal')
                    ])
                    
                except Exception as e:
                    dialog.after(0, lambda: [
                        progress_label.config(text=f"Erro no treinamento: {str(e)}"),
                        train_button.config(state='normal'),
                        cancel_button.config(state='normal')
                    ])
            
            threading.Thread(target=training_thread, daemon=True).start()
        
        # Função para aplicar perfil aos sliders
        def apply_profile(profile_name):
            profile = profiles[profile_name]
            alpha_var.set(profile["alpha"])
            gamma_var.set(profile["gamma"])
            epsilon_var.set(profile["epsilon"])
            episodes_var.set(profile["episodes"])
            
            # Atualizar labels
            alpha_value_label.config(text=f"{profile['alpha']}")
            gamma_value_label.config(text=f"{profile['gamma']}")
            epsilon_value_label.config(text=f"{profile['epsilon']}")
            episodes_value_label.config(text=f"{profile['episodes']:,}")
            
            progress_label.config(text=f"Perfil {profile_name.upper()} aplicado! Ajuste os sliders se necessário.")
        
        # Botões de perfil
        profile_buttons_frame = tk.Frame(profiles_frame, bg='#2c3e50')
        profile_buttons_frame.pack(pady=10)
        
        tk.Button(profile_buttons_frame, text="🟢 Iniciante", 
                 bg='#4CAF50', fg='white', command=lambda: apply_profile("iniciante"),
                 font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(profile_buttons_frame, text="🟠 Intermediário", 
                 bg='#FF9800', fg='white', command=lambda: apply_profile("intermediario"),
                 font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(profile_buttons_frame, text="🔴 Avançado", 
                 bg='#F44336', fg='white', command=lambda: apply_profile("avancado"),
                 font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        # Botões principais
        button_frame = tk.Frame(dialog, bg='#2c3e50')
        button_frame.pack(pady=20)
        
        # Botão de treinamento
        train_button = tk.Button(button_frame, text="🚀 Iniciar Treinamento", 
                               bg='#e74c3c', fg='white', command=start_training,
                               font=('Arial', 12, 'bold'), width=20)
        train_button.pack(side=tk.LEFT, padx=10)
        
        # Botão de cancelar
        cancel_button = tk.Button(button_frame, text="Cancelar", 
                                bg='#95a5a6', fg='white', command=dialog.destroy,
                                font=('Arial', 12), width=12)
        cancel_button.pack(side=tk.LEFT, padx=10)
        
        # Mensagem inicial
        progress_label.config(text="Use os perfis predefinidos ou ajuste os sliders manualmente")
    
    def player_move(self, position):
        if self.game_over or not self.player_turn or self.board[position] != ' ':
            return
        
        # Jogada do jogador
        self.board[position] = self.player_symbol
        self.buttons[position].config(text=self.player_symbol, fg='#3498db')
        
        if check_winner(self.board, self.player_symbol):
            self.end_game("Você ganhou! 🎉")
            self.pts_jogador += 1
            return
        
        if ' ' not in self.board:
            self.end_game("Empate! 🤝")
            return
        
        # Vez do PC
        self.player_turn = False
        self.status_label.config(text="PC está pensando...")
        self.root.update()
        
        # Adicionar pequeno delay para simular "pensamento"
        self.root.after(1000, self.pc_move)
    
    def pc_move(self):
        if self.game_over:
            return
        
        pc_position = get_pc_move(self.board)
        
        if pc_position != -1:
            self.board[pc_position] = self.pc_symbol
            self.buttons[pc_position].config(text=self.pc_symbol, fg='#e74c3c')
            
            if check_winner(self.board, self.pc_symbol):
                self.end_game("PC ganhou! 🤖")
                self.pts_pc += 1
                return
            
            if ' ' not in self.board:
                self.end_game("Empate! 🤝")
                return
        
        # Volta a ser vez do jogador
        self.player_turn = True
        self.status_label.config(text="Sua vez! Clique em uma posição.")
    
    def end_game(self, message):
        self.game_over = True
        self.status_label.config(text=message)
        self.update_score()
        
        # Perguntar se quer jogar novamente
        self.root.after(2000, lambda: messagebox.askyesno(
            "Fim de Jogo", 
            f"{message}\n\nQuer jogar novamente?",
            parent=self.root
        ) and self.new_game())
    
    def new_game(self):
        self.board = [' '] * 9
        self.player_turn = True
        self.game_over = False
        
        for button in self.buttons:
            button.config(text=' ', fg='#ecf0f1')
        
        self.status_label.config(text="Novo jogo! Sua vez, clique em uma posição.")
    
    def update_score(self):
        self.score_label.config(text=f"🏆 Você: {self.pts_jogador} | 🤖 PC: {self.pts_pc}")
    
    def quit_game(self):
        if messagebox.askyesno("Sair", "Deseja realmente sair do jogo?"):
            self.root.quit()

def main():
    root = tk.Tk()
    app = TicTacToeGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
