import gymnasium as gym
from src.environment import ImprovedChatbotEnv, ImprovedDQNAgent, ImprovedQLearningAgent
from src.rag.retriever import RAGRetriever
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
import numpy as np
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
    documents  = np.array(documents)
    
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