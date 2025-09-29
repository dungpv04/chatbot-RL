# improved_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# environment.py (updated)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ChatbotStudentEnv(gym.Env):
    """
    Environment that:
    - uses a provided RAGRetriever (SentenceTransformer + FAISS)
    - state: 5 discrete features (0..2)
      [similarity, retrieval_conf, context_match, ambiguity, scope_match]
    - actions: 5
      0: quote (trích dẫn nguyên văn)
      1: summary (tóm tắt)
      2: paraphrase (diễn giải)
      3: clarify (hỏi lại)
      4: escalate (thoái lui an toàn)
    - reward: mapped to values in the report: +3, -5, -8, -12, +2, -1
    """

    def __init__(self, rag_retriever, qa_pairs_file="qa_pairs.json", max_turns=5, top_k=3):
        super().__init__()
        self.rag_retriever = rag_retriever
        self.top_k = top_k
        self.max_turns = max_turns

        # Load Q&A pairs (for expected answers / keywords)
        self.qa_pairs = self._load_qa_pairs(qa_pairs_file)

        # Actions
        self.action_space = spaces.Discrete(5)

        # Observation: 5 discrete features each in {0,1,2}
        self.observation_space = spaces.MultiDiscrete([3] * 5)

        # internal state
        self.current_qa = None
        self.current_question = None
        self.expected_keywords = None
        self.retrieved_docs = []
        self.state = None
        self.turn_count = 0
        self.question_answered = False
        self.dialog_history = []  # store previous question embeddings (for context match)
        self.reset()

    # -------------------------
    # Public Gym API
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # pick a random QA pair (mocking incoming user question)
        self.current_qa = np.random.choice(self.qa_pairs)
        self.current_question = self.current_qa["question"]
        self.expected_keywords = self.current_qa.get("keywords", [])
        self.retrieved_docs = []
        self.turn_count = 0
        self.question_answered = False
        self.state = self._compute_state(self.current_question)
        return self.state, {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        reward = 0
        terminated = False
        info = {}

        # If agent chooses an answering action
        if action in [0, 1, 2]:
            # calculate reward using retrieval + keyword checks + scope heuristics
            reward = self._calculate_answer_reward(action)
            self.question_answered = True
            terminated = True

        elif action == 3:  # ask clarification
            # +2 if ambiguity present (worth asking), else -1
            ambiguity = int(self.state[3])
            reward = 2 if ambiguity >= 1 else -1
            terminated = True

        elif action == 4:  # escalate / safe retreat
            reward = -1
            terminated = True

        # turn update
        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            terminated = True
            if not self.question_answered:
                reward -= 1.0  # penalty for not answering within max_turns

        # recompute state (simulate next retrieval / new context)
        self.state = self._compute_state(self.current_question)

        info.update({
            "question": self.current_question,
            "turn": self.turn_count,
            "answered": self.question_answered,
            "expected_answer": self.current_qa.get("answer", ""),
            "retrieved_docs": len(self.retrieved_docs)
        })

        return self.state, reward, terminated, False, info

    def render(self, mode='human'):
        print(f"Turn {self.turn_count} | State={self.state} | Retrieved={len(self.retrieved_docs)}")

    # -------------------------
    # Helpers: load data
    # -------------------------
    def _load_qa_pairs(self, qa_file):
        # keep the sample pairs from your original file; you can replace by reading qa_file
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

    # -------------------------
    # Helpers: state computation using RAGRetriever
    # -------------------------
    def _compute_state(self, question):
        """
        Use RAG retriever to get top_k docs and compute the 5 discrete features:
        similarity, retrieval_conf, context_match, ambiguity, scope_match
        """
        top_k = self.top_k

        # Retrieve docs (textual)
        docs = self.rag_retriever.retrieve(question, top_k=top_k)
        self.retrieved_docs = docs

        # compute similarity scores between query and ALL stored docs in retriever
        # (so we can compute accurate top-k similarity)
        try:
            query_emb = self.rag_retriever.model.encode([question]).astype('float32')
            all_embs = np.asarray(self.rag_retriever.embeddings).astype('float32')
            sims = cosine_similarity(query_emb, all_embs)[0]  # shape (n_docs,)
        except Exception:
            # fallback: if embeddings are not available for some reason
            sims = np.array([])

        # pick top-k sims (if sims available)
        if sims.size:
            top_idx = np.argsort(-sims)[:top_k]
            top_sims = sims[top_idx]
            max_sim = float(top_sims.max())
            avg_topk = float(top_sims.mean())
        else:
            max_sim = 0.0
            avg_topk = 0.0

        # similarity level mapping (report): <0.4 ->0, 0.4-0.7 ->1, >=0.7 ->2
        if max_sim < 0.4:
            similarity = 0
        elif max_sim < 0.7:
            similarity = 1
        else:
            similarity = 2

        # retrieval confidence from avg_topk: <0.3 ->0, 0.3-0.6 ->1, >=0.6 ->2
        if avg_topk < 0.3:
            retrieval_conf = 0
        elif avg_topk < 0.6:
            retrieval_conf = 1
        else:
            retrieval_conf = 2

        # context_match: check if last user question in history is similar to current (>0.7 ->2, 0.4-0.7 ->1)
        if len(self.dialog_history) == 0:
            context_match = 0
        else:
            try:
                last_emb = self.dialog_history[-1]
                ctx_sim = float(cosine_similarity(query_emb, last_emb.reshape(1, -1))[0][0])
                if ctx_sim >= 0.7:
                    context_match = 2
                elif ctx_sim >= 0.4:
                    context_match = 1
                else:
                    context_match = 0
            except Exception:
                context_match = 0

        # ambiguity heuristic (simple rule-based):
        q_lower = question.lower()
        if ("hoặc" in q_lower) or ("hay" in q_lower and "hay không" not in q_lower):
            ambiguity = 1
        elif any(p in q_lower for p in ["thông tin", "vấn đề", "như thế nào", "làm sao", "cách"]) or len(q_lower.split()) < 4:
            ambiguity = 2
        else:
            ambiguity = 0

        # scope_match heuristic: check if retrieved docs contain scope-related keywords
        top_docs_text = " ".join(docs).lower() if docs else ""
        scope_keywords = ["sinh viên", "khoa", "ngành", "năm", "chính quy", "học kỳ", "tín chỉ", "phòng đào tạo", "giáo vụ"]
        if any(kw in top_docs_text for kw in scope_keywords):
            scope_match = 2
        else:
            # partial match if docs contain some domain-ish tokens (e.g., 'học', 'kỳ', 'điều kiện')
            if any(kw in top_docs_text for kw in ["học", "điều kiện", "thi", "học phí", "đăng ký"]):
                scope_match = 1
            else:
                scope_match = 0

        # push current query embedding into dialog history (for future steps)
        try:
            self.dialog_history.append(query_emb.flatten())
            # keep max history length small
            if len(self.dialog_history) > 8:
                self.dialog_history.pop(0)
        except Exception:
            pass

        state = np.array([similarity, retrieval_conf, context_match, ambiguity, scope_match], dtype=np.int32)
        return state

    # -------------------------
    # Retrieval / Answer reward functions
    # -------------------------
    def _calculate_retrieval_reward(self):
        """
        Returns a continuous retrieval score (float) based on keyword overlap,
        used internally or for diagnostics. Not used directly as the discrete final reward.
        """
        if not self.retrieved_docs:
            return 0.0

        retrieved_text = " ".join(self.retrieved_docs).lower()
        if not self.expected_keywords:
            return 0.0

        keyword_matches = sum(1 for kw in self.expected_keywords if kw.lower() in retrieved_text)
        keyword_ratio = keyword_matches / len(self.expected_keywords)
        # scale 0..2
        return float(keyword_ratio * 2.0)

    def _calculate_answer_reward(self, action):
        """
        Map (state, retrieval quality, keyword overlap) into the discrete rewards from the report:
          +3: correct answer with valid citation and correct scope
          -5: missing citation / low confidence while question ambiguous (should have asked)
          -8: wrong scope (khoa/ngành/năm)
          -12: hallucination or severe incorrect / user dislike
          +2: good clarification
          -1: useless clarification or escalate cost
        Heuristics are used because we don't have explicit user feedback signals.
        """
        # unpack state
        similarity, retrieval_conf, context_match, ambiguity, scope_match = map(int, self.state)

        # keyword coverage
        retrieved_text = " ".join(self.retrieved_docs).lower() if self.retrieved_docs else ""
        keyword_matches = 0
        if self.expected_keywords:
            keyword_matches = sum(1 for kw in self.expected_keywords if kw.lower() in retrieved_text)
            keyword_ratio = keyword_matches / len(self.expected_keywords)
        else:
            keyword_ratio = 0.0

        # If scope mismatch => -8
        if scope_match == 0:
            return -8

        # No retrieved evidence (low retrieval_conf) -> either -5 (if ambiguous) or -12 (hallucination)
        if retrieval_conf == 0:
            if ambiguity >= 1:
                return -5  # should have asked clarifying
            else:
                return -12  # hallucination / severe error

        # If we have some retrieval evidence:
        # Award +3 when keyword coverage is high AND scope matches
        if keyword_ratio >= 0.6 and scope_match == 2:
            return 3

        # If partial evidence but not sufficient -> penalize as missing citation / low reliability
        return -5




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
    env = ChatbotStudentEnv(rag_retriever)
    
    print("Training Q-Learning Agent...")
    q_agent = ImprovedQLearningAgent(env.action_space)
    train_improved_agent(env, q_agent, episodes=1000, agent_type="Q-Learning")
    
    print("\nTraining DQN Agent...")
    dqn_agent = ImprovedDQNAgent(env.observation_space.shape[0], env.action_space.n)
    train_improved_agent(env, dqn_agent, episodes=1000, agent_type="DQN")

if __name__ == "__main__":
    main()