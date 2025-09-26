import gymnasium as gym
from src.environment import ImprovedChatbotEnv
from src.rag.retriever import RAGRetriever
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
import numpy as np
def main():
    # Load notebook content
    with open('data/notebook_content.txt', 'r') as f:
        documents = f.read().split('\n\n')  # Split by paragraphs
    
    # Initialize RAG
    rag_retriever = RAGRetriever(documents)
    
    # Create environment
    env = ImprovedChatbotEnv(rag_retriever)
    
    # Train Q-Learning agent
    q_agent = QLearningAgent(env.action_space)
    train_agent(env, q_agent, episodes=1000, agent_type="Q-Learning")
    
    # Train DQN agent
    dqn_agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    train_agent(env, dqn_agent, episodes=1000, agent_type="DQN")

def train_agent(env, agent, episodes, agent_type):
    total_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            if agent_type == "Q-Learning":
                agent.update(state, action, reward, next_state)
            else:  # DQN
                agent.store_transition(state, action, reward, next_state, terminated)
                agent.update()
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"{agent_type} Episode {episode}: Average Reward = {avg_reward:.2f}")

if __name__ == "__main__":
    main()