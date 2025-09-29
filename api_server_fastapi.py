from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import torch
import numpy as np
import os
from typing import Dict, Optional
from src.environment import ImprovedDQNAgent, ImprovedQLearningAgent
from src.rag.retriever import RAGRetriever

# Initialize FastAPI app
app = FastAPI(
    title="🤖 Student Handbook Chatbot API",
    description="Chatbot trả lời câu hỏi từ sổ tay sinh viên sử dụng DQN và Q-Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for loaded models
q_agent = None
dqn_agent = None
rag_retriever = None
env = None

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    
    class Config:
        example = {
            "question": "Học phần là gì?"
        }

class ChatResponse(BaseModel):
    question: str
    answer: str
    agent: str
    status: str = "success"

class CompareResponse(BaseModel):
    question: str
    responses: Dict[str, str]
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    message: str
    timestamp: str

def load_trained_models():
    """Load pre-trained models"""
    global q_agent, dqn_agent, rag_retriever, env
    
    try:
        # Load RAG retriever first
        with open('models/rag_retriever.pkl', 'rb') as f:
            rag_retriever = pickle.load(f)
        
        # Create environment
        from src.environment import ImprovedChatbotEnv
        env = ImprovedChatbotEnv(rag_retriever)
        
        # Load Q-Learning agent
        with open('models/q_learning_agent.pkl', 'rb') as f:
            q_agent = pickle.load(f)
        
        # Load DQN agent
        dqn_agent = ImprovedDQNAgent(env.observation_space.shape[0], env.action_space.n)
        
        # Add numpy scalar to safe globals for PyTorch 2.6+ compatibility
        try:
            import torch.serialization
            torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar', 
                                                'collections.OrderedDict', 
                                                'torch._utils._rebuild_tensor_v2'])
            # Load with security settings
            checkpoint = torch.load('models/dqn_agent.pth', weights_only=False, map_location='cpu')
        except Exception as e:
            print(f"First loading attempt failed: {e}")
            # Fallback: Load with weights_only=False
            checkpoint = torch.load('models/dqn_agent.pth', weights_only=False)
            
        dqn_agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        if 'target_model_state_dict' in checkpoint and checkpoint['target_model_state_dict'] is not None:
            dqn_agent.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        dqn_agent.epsilon = 0.01  # Set low epsilon for inference
        
        print("✅ Đã load thành công tất cả models!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi load models: {e}")
        print("💡 Hãy chạy 'python train_agents.py' trước để train models")
        return False

def get_agent_response(question: str, agent_type: str = "dqn") -> str:
    """Get response from specified agent"""
    if not rag_retriever:
        raise HTTPException(status_code=503, detail="Models chưa được load")
    
    try:
        # Get question embedding
        question_embedding = rag_retriever.model.encode([question])
        state = question_embedding[0].astype(np.float32)
        
        # Select agent
        if agent_type.lower() == "q-learning":
            agent = q_agent
        else:
            agent = dqn_agent
        
        if not agent:
            raise HTTPException(status_code=503, detail=f"{agent_type} agent chưa được load")
        
        # Get agent's action
        action = agent.select_action(state)
        
        # Map actions to responses
        if action == 0:  # Retrieve documents
            retrieved_docs = rag_retriever.retrieve(question, top_k=3)
            response = f"📚 Tôi đã tìm thấy thông tin liên quan:\n\n"
            for i, doc in enumerate(retrieved_docs, 1):
                response += f"{i}. {doc[:200]}...\n\n"
            return response
            
        elif action == 1:  # Generate answer
            retrieved_docs = rag_retriever.retrieve(question, top_k=2)
            if retrieved_docs:
                response = f"💡 Dựa trên thông tin trong sổ tay sinh viên:\n\n"
                response += f"{retrieved_docs[0][:300]}..."
                return response
            else:
                return "😅 Xin lỗi, tôi không tìm thấy thông tin liên quan trong sổ tay sinh viên."
                
        elif action == 2:  # Ask clarification  
            return "🤔 Bạn có thể làm rõ câu hỏi không? Hoặc cung cấp thêm thông tin để tôi có thể hỗ trợ tốt hơn."
        
        else:
            return "😅 Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn."
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

# API Endpoints
@app.get("/", tags=["Home"])
async def home():
    """Trang chủ với thông tin API"""
    return {
        "message": "🤖 Student Handbook Chatbot API với FastAPI",
        "docs": "/docs",
        "redoc": "/redoc", 
        "endpoints": {
            "POST /ask": "Hỏi câu hỏi với DQN agent",
            "POST /ask-qlearning": "Hỏi câu hỏi với Q-Learning agent",
            "POST /compare": "So sánh response của cả 2 agents",
            "GET /health": "Kiểm tra trạng thái API"
        },
        "example_request": {
            "url": "/ask",
            "method": "POST", 
            "body": {"question": "Học phần là gì?"}
        }
    }

@app.post("/ask", response_model=ChatResponse, tags=["DQN Agent"])
async def ask_dqn(request: QuestionRequest):
    """
    Hỏi câu hỏi với DQN Agent
    
    - **question**: Câu hỏi về sổ tay sinh viên
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
    
    response = get_agent_response(request.question, agent_type="dqn")
    
    return ChatResponse(
        question=request.question,
        answer=response,
        agent="DQN"
    )

@app.post("/ask-qlearning", response_model=ChatResponse, tags=["Q-Learning Agent"])
async def ask_qlearning(request: QuestionRequest):
    """
    Hỏi câu hỏi với Q-Learning Agent
    
    - **question**: Câu hỏi về sổ tay sinh viên
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
    
    response = get_agent_response(request.question, agent_type="q-learning")
    
    return ChatResponse(
        question=request.question,
        answer=response,
        agent="Q-Learning"
    )

@app.post("/compare", response_model=CompareResponse, tags=["Compare Agents"])
async def compare_agents(request: QuestionRequest):
    """
    So sánh response của cả DQN và Q-Learning agents
    
    - **question**: Câu hỏi để so sánh
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
    
    dqn_response = get_agent_response(request.question, agent_type="dqn")
    qlearning_response = get_agent_response(request.question, agent_type="q-learning")
    
    return CompareResponse(
        question=request.question,
        responses={
            "DQN": dqn_response,
            "Q-Learning": qlearning_response
        }
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Kiểm tra trạng thái API và models"""
    models_loaded = all([q_agent, dqn_agent, rag_retriever])
    
    return HealthResponse(
        status="healthy" if models_loaded else "models_not_loaded",
        models_loaded=models_loaded,
        message="API đang chạy tốt" if models_loaded else "Cần train models trước",
        timestamp=str(np.datetime64('now'))
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models khi khởi động API"""
    print("🚀 Starting Student Handbook Chatbot API với FastAPI...")
    
    if load_trained_models():
        print("✅ Models đã được load thành công!")
    else:
        print("⚠️  API sẽ chạy nhưng models chưa được load")
        print("💡 Chạy 'python train_agents.py' để train models")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint không tồn tại",
        "message": "Xem docs tại /docs để biết các endpoints có sẵn"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Lỗi server nội bộ", 
        "message": "Vui lòng thử lại sau"
    }

