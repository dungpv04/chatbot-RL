# improved_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ImprovedChatbotEnv(gym.Env):
    def __init__(self, rag_retriever, qa_pairs_file="qa_pairs.json", max_turns=5):
        super().__init__()
        self.rag_retriever = rag_retriever
        self.max_turns = max_turns
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load Q&A pairs for evaluation
        self.qa_pairs = self._load_qa_pairs(qa_pairs_file)
        
        # Action space: [retrieve_docs, generate_answer, ask_clarification]
        self.action_space = spaces.Discrete(3)
        
        # State space: question embedding (384 dim)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(384,), dtype=np.float32
        )
        
        self.reset()
    
    def _load_qa_pairs(self, qa_file):
        """Load Q&A pairs from file or create sample ones"""
        qa_pairs = [
            {
                "question": "Học phần là gì?",
                "answer": "Học phần là khối lượng kiến thức tương đối trọn vẹn, thuận tiện cho sinh viên tích lũy trong quá trình học tập",
                "keywords": ["học phần", "kiến thức", "sinh viên", "tích lũy", "học tập"]
            },
            {
                "question": "Tín chỉ là gì?",
                "answer": "Tín chỉ là đơn vị được sử dụng để tính khối lượng học tập, tích lũy của sinh viên",
                "keywords": ["tín chỉ", "đơn vị", "khối lượng", "học tập", "sinh viên"]
            },
            {
                "question": "Thời gian học tối đa cho sinh viên đại học là bao lâu?",
                "answer": "Thời gian học tập tối đa cho sinh viên đại học hình thức chính quy là 8,0 năm đến 9,0 năm",
                "keywords": ["thời gian", "tối đa", "sinh viên", "đại học", "chính quy", "năm"]
            },
            {
                "question": "Điều kiện dự thi kết thúc học phần là gì?",
                "answer": "Sinh viên được dự thi khi có mặt ở lớp từ 80% trở lên thời gian quy định",
                "keywords": ["điều kiện", "dự thi", "kết thúc", "học phần", "80%", "thời gian"]
            },
            {
                "question": "Khi nào sinh viên bị cảnh báo học tập?",
                "answer": "Sinh viên bị cảnh báo khi điểm trung bình chung học kỳ đạt dưới 1,00 hoặc tín chỉ F vượt quá 24",
                "keywords": ["cảnh báo", "học tập", "điểm trung bình", "học kỳ", "tín chỉ", "24"]
            }
        ]
        return qa_pairs
    
    def reset(self, seed=None):
        """Reset environment with a random question"""
        self.current_qa = np.random.choice(self.qa_pairs)
        self.current_question = self.current_qa["question"]
        self.expected_keywords = self.current_qa["keywords"]
        self.retrieved_docs = []
        self.turn_count = 0
        self.question_answered = False
        
        # Get question embedding as state
        question_embedding = self.embedding_model.encode(self.current_question)
        return question_embedding.astype(np.float32), {}
    
    def step(self, action):
        reward = 0
        terminated = False
        info = {}
        
        if action == 0:  # Retrieve documents
            if not self.question_answered:
                self.retrieved_docs = self.rag_retriever.retrieve(self.current_question, top_k=3)
                # Calculate reward based on keyword overlap
                reward = self._calculate_retrieval_reward()
                info['retrieved_docs'] = len(self.retrieved_docs)
            else:
                reward = -0.5  # Penalty for retrieving after answering
        
        elif action == 1:  # Generate answer
            if self.retrieved_docs and not self.question_answered:
                # Reward for answering with retrieved context
                reward = self._calculate_answer_reward()
                self.question_answered = True
                terminated = True
            elif not self.retrieved_docs:
                reward = -2.0  # Big penalty for answering without context
                terminated = True
            else:
                reward = -0.5  # Already answered
                terminated = True
        
        elif action == 2:  # Ask clarification
            if not self.question_answered:
                reward = 0.2  # Small positive for being cautious
                terminated = True
            else:
                reward = -0.3  # Too late to ask
                terminated = True
        
        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            terminated = True
            if not self.question_answered:
                reward -= 1.0  # Penalty for not answering
        
        # Get current state (question embedding doesn't change)
        question_embedding = self.embedding_model.encode(self.current_question)
        
        info.update({
            'question': self.current_question,
            'turn': self.turn_count,
            'answered': self.question_answered,
            'expected_answer': self.current_qa["answer"]
        })
        
        return question_embedding.astype(np.float32), reward, terminated, False, info
    
    def _calculate_retrieval_reward(self):
        """Calculate reward based on how relevant retrieved documents are"""
        if not self.retrieved_docs:
            return -1.0
        
        # Check if retrieved docs contain expected keywords
        retrieved_text = " ".join(self.retrieved_docs).lower()
        keyword_matches = sum(1 for keyword in self.expected_keywords 
                            if keyword.lower() in retrieved_text)
        
        # Reward based on keyword coverage
        keyword_ratio = keyword_matches / len(self.expected_keywords)
        
        if keyword_ratio >= 0.6:  # Good retrieval
            return 2.0
        elif keyword_ratio >= 0.3:  # Decent retrieval
            return 1.0
        else:  # Poor retrieval
            return 0.2
    
    def _calculate_answer_reward(self):
        """Calculate reward for answering with context"""
        # Base reward for answering with retrieved documents
        base_reward = 1.5
        
        # Bonus if retrieved docs are highly relevant
        retrieved_text = " ".join(self.retrieved_docs).lower()
        keyword_matches = sum(1 for keyword in self.expected_keywords 
                            if keyword.lower() in retrieved_text)
        keyword_ratio = keyword_matches / len(self.expected_keywords)
        
        # Additional reward based on relevance
        relevance_bonus = keyword_ratio * 1.0
        
        return base_reward + relevance_bonus


# improved_agents.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import random

class ImprovedDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ImprovedDQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_network = ImprovedDQN(state_dim, action_dim)
        self.target_network = ImprovedDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.memory = deque(maxlen=10000)
        self.update_target_freq = 100
        self.steps = 0
        
        # Copy weights to target network
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (0.99 * next_q * ~dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.update_target_network()

class ImprovedQLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space = action_space
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
    
    def _state_to_key(self, state):
        # Discretize continuous state for Q-table
        # Use clustering or binning of the embedding
        discretized = np.round(state * 10).astype(int)  # Scale and round
        # Take only a subset to reduce dimensionality
        key_indices = [0, 50, 100, 150, 200, 250, 300, 350]  # Sample indices
        return tuple(discretized[key_indices])
    
    def select_action(self, state):
        state_key = self._state_to_key(state)
        
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] = current_q + self.lr * (
            reward + self.discount * next_max_q - current_q
        )
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# improved_main.py
def train_improved_agent(env, agent, episodes, agent_type):
    total_rewards = []
    success_rate = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_successful = False
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            if agent_type == "Q-Learning":
                agent.update(state, action, reward, next_state)
            else:  # DQN
                agent.store_transition(state, action, reward, next_state, terminated)
                agent.update()
            
            state = next_state
            total_reward += reward
            
            # Check if episode was successful (answered correctly)
            if info.get('answered', False) and reward > 1.0:
                episode_successful = True
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        success_rate.append(1.0 if episode_successful else 0.0)
        
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            avg_success = np.mean(success_rate[-100:]) * 100
            print(f"{agent_type} Episode {episode}: Avg Reward = {avg_reward:.2f}, Success Rate = {avg_success:.1f}%")
            
            # Print sample interaction
            if episode % 200 == 0 and info:
                print(f"  Sample Question: {info.get('question', 'N/A')}")
                print(f"  Agent Answered: {info.get('answered', False)}")
                print(f"  Retrieved Docs: {info.get('retrieved_docs', 0)}")

def main():
    # Load notebook content and create sample data
    documents = [
        "Học phần là khối lượng kiến thức tương đối trọn vẹn, thuận tiện cho sinh viên tích lũy trong quá trình học tập. Phần lớn học phần có khối lượng từ 2 đến 4 tín chỉ.",
        "Tín chỉ là đơn vị được sử dụng để tính khối lượng học tập, tích lũy của sinh viên.",
        "Thời gian học tập tối đa để sinh viên hoàn thành khoá học được Trường quy định như sau: Hình thức chính quy: 8,0 năm ÷ 9,0 năm.",
        "Sinh viên được dự thi kết thúc học phần khi có đủ các điều kiện: Có mặt ở lớp từ 80% trở lên thời gian quy định cho học phần đó.",
        "Sinh viên bị cảnh báo khi điểm trung bình chung học kỳ đạt dưới 1,00 đối với các học kỳ tiếp theo hoặc tổng số tín chỉ bị điểm F vượt quá 24 tín chỉ."
    ]
    
    # Initialize components
    from src.rag.retriever import RAGRetriever
    rag_retriever = RAGRetriever(documents)
    
    # Create improved environment
    env = ImprovedChatbotEnv(rag_retriever)
    
    print("Training Q-Learning Agent...")
    q_agent = ImprovedQLearningAgent(env.action_space)
    train_improved_agent(env, q_agent, episodes=1000, agent_type="Q-Learning")
    
    print("\nTraining DQN Agent...")
    dqn_agent = ImprovedDQNAgent(env.observation_space.shape[0], env.action_space.n)
    train_improved_agent(env, dqn_agent, episodes=1000, agent_type="DQN")

if __name__ == "__main__":
    main()