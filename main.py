import numpy as np
import pickle
import os
import torch

def load_documents():
    """Load documents from the notebook content"""
    with open('data/notebook_content.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into meaningful chunks
    documents = []
    sections = content.split('\n\n')
    
    for section in sections:
        if len(section.strip()) > 100:  # Only keep substantial sections
            documents.append(section.strip())
    
    return np.array(documents[:50])  # Limit to first 50 documents for efficiency

# Define SimpleQLearningAgent at the module level so it can be pickled
class SimpleQLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount=0.95, epsilon=1.0):
        self.action_space = action_space
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.q_table = {}  # Regular dict instead of defaultdict
    
    def _state_to_key(self, state):
        # Discretize continuous state for Q-table
        discretized = np.round(state * 10).astype(int)
        key_indices = [0, 50, 100, 150, 200, 250, 300, 350]
        return tuple(discretized[key_indices])
    
    def _get_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        return self.q_table[state_key]
    
    def select_action(self, state):
        state_key = self._state_to_key(state)
        
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = self._get_q_values(state_key)
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        current_q = self._get_q_values(state_key)[action]
        next_q_values = self._get_q_values(next_state_key)
        next_max_q = np.max(next_q_values)
        
        self.q_table[state_key][action] = current_q + self.lr * (
            reward + self.discount * next_max_q - current_q
        )
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def create_simple_q_agent(action_space):
    """Tạo Q-Learning agent đơn giản có thể pickle được"""
    return SimpleQLearningAgent(action_space)

def train_agent(env, agent, episodes, agent_type):
    """Train an agent and track performance"""
    total_rewards = []
    success_rate = []
    
    print(f"\n🔥 Bắt đầu training {agent_type} agent...")
    
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
            
            # Check if episode was successful
            if info.get('answered', False) and reward > 1.0:
                episode_successful = True
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        success_rate.append(1.0 if episode_successful else 0.0)
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            avg_success = np.mean(success_rate[-100:]) * 100
            print(f"📊 Episode {episode}/{episodes}: Avg Reward = {avg_reward:.2f}, Success Rate = {avg_success:.1f}%")
    
    final_avg_reward = np.mean(total_rewards[-100:])
    final_success_rate = np.mean(success_rate[-100:]) * 100
    
    print(f"✅ {agent_type} training completed!")
    print(f"📈 Final Performance: Avg Reward = {final_avg_reward:.2f}, Success Rate = {final_success_rate:.1f}%")
    
    return {
        'total_rewards': total_rewards,
        'success_rate': success_rate,
        'final_avg_reward': final_avg_reward,
        'final_success_rate': final_success_rate
    }

def save_models_safe(q_agent, dqn_agent, rag_retriever):
    """Save models một cách an toàn"""
    os.makedirs('models', exist_ok=True)
    
    try:
        # Save Q-Learning agent
        try:
            with open('models/q_learning_agent.pkl', 'wb') as f:
                pickle.dump(q_agent, f)
            print("✅ Đã lưu Q-Learning agent")
        except Exception as e:
            print(f"⚠️ Không thể lưu Q-Learning agent: {e}")
            
            # Fallback: Save just the Q-table as a dictionary
            q_table_dict = {str(k): v.tolist() for k, v in q_agent.q_table.items()}
            with open('models/q_table.json', 'w') as f:
                import json
                json.dump(q_table_dict, f)
            print("✅ Đã lưu Q-table dưới dạng JSON thay thế")
        
        # Save DQN agent
        try:
            torch.save({
                'model_state_dict': dqn_agent.q_network.state_dict(),
                'target_model_state_dict': dqn_agent.target_network.state_dict() if hasattr(dqn_agent, 'target_network') else None,
                'epsilon': dqn_agent.epsilon,
                'action_dim': dqn_agent.action_dim
            }, 'models/dqn_agent.pth')
            print("✅ Đã lưu DQN agent")
        except Exception as e:
            print(f"⚠️ Không thể lưu DQN agent: {e}")
        
        # Save RAG retriever
        try:
            with open('models/rag_retriever.pkl', 'wb') as f:
                pickle.dump(rag_retriever, f)
            print("✅ Đã lưu RAG retriever")
        except Exception as e:
            print(f"⚠️ Không thể lưu RAG retriever: {e}")
        
        print("💾 Đã lưu thành công các models!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu models: {e}")
        return False
    
    return True

def main():
    print("🚀 Khởi tạo training environment...")
    
    # Load documents
    documents = load_documents()
    print(f"📚 Đã load {len(documents)} documents")
    
    # Initialize RAG retriever
    from src.rag.retriever import RAGRetriever
    rag_retriever = RAGRetriever(documents)
    
    # Create environment
    from src.environment import ImprovedChatbotEnv, ImprovedDQNAgent
    env = ImprovedChatbotEnv(rag_retriever)
    
    # Initialize agents
    q_agent = create_simple_q_agent(env.action_space)
    dqn_agent = ImprovedDQNAgent(env.observation_space.shape[0], env.action_space.n)
    
    # Training parameters
    episodes = 1000
    
    # Train Q-Learning agent
    q_results = train_agent(env, q_agent, episodes, "Q-Learning")
    
    # Train DQN agent  
    dqn_results = train_agent(env, dqn_agent, episodes, "DQN")
    
    # Compare results
    print("\n" + "="*50)
    print("📊 TRAINING RESULTS COMPARISON")
    print("="*50)
    print(f"Q-Learning Agent:")
    print(f"  - Final Avg Reward: {q_results['final_avg_reward']:.2f}")
    print(f"  - Final Success Rate: {q_results['final_success_rate']:.1f}%")
    print(f"\nDQN Agent:")
    print(f"  - Final Avg Reward: {dqn_results['final_avg_reward']:.2f}")
    print(f"  - Final Success Rate: {dqn_results['final_success_rate']:.1f}%")
    
    # Determine winner
    if q_results['final_success_rate'] > dqn_results['final_success_rate']:
        print(f"\n🏆 Q-Learning Agent wins with {q_results['final_success_rate']:.1f}% success rate!")
    elif dqn_results['final_success_rate'] > q_results['final_success_rate']:
        print(f"\n🏆 DQN Agent wins with {dqn_results['final_success_rate']:.1f}% success rate!")
    else:
        print(f"\n🤝 It's a tie at {q_results['final_success_rate']:.1f}% success rate!")
    
    # Save trained models
    if save_models_safe(q_agent, dqn_agent, rag_retriever):
        print("\n✅ Training hoàn tất! Có thể chạy API server bằng:")
        print("   python api_server_fastapi.py")
    else:
        print("\n⚠️  Training hoàn tất nhưng có lỗi khi lưu models")

if __name__ == "__main__":
    #main()
    import uvicorn
    uvicorn.run(
        "api_server_fastapi:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )