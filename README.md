# DeFi AI/ML: Advanced Crypto Sentiment Forecaster 🚀

> Leveraging LangChain, RAG, and Advanced AI/ML for Intelligent DeFi Analytics and Predictions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeshwanth-A/defi_aiml/blob/main/Defi_Aiml.ipynb)

## 🌟 Revolutionary Features

This project represents a cutting-edge fusion of **Decentralized Finance (DeFi)** analytics with **LangChain-powered RAG (Retrieval-Augmented Generation)**, creating an intelligent system that not only predicts cryptocurrency prices but also provides contextual insights through conversational AI.

### 🎯 Core Capabilities

- 🤖 **LangChain Integration**: Full RAG pipeline with FAISS vector store for intelligent data querying
- 📊 **Multi-Model Ensemble**: PyTorch LSTM, TensorFlow LSTM, and Random Forest with automated comparison
- 🧠 **Fine-tuned LLM**: DistilBERT with LoRA for context-aware prediction explanations
- 🛠️ **Agent Framework**: Custom LangChain agents with specialized tools for real-time analysis
- 🎮 **Interactive Dashboard**: Complete ipywidgets-based GUI with live model interaction
- 💬 **Conversation Memory**: Persistent chat system with intelligent context retention
- 📈 **Advanced Analytics**: Comprehensive visualization suite with 4-panel comparative analysis

## 🏆 Technical Achievements

### Performance Metrics
- **Model Accuracy**: MAE < 0.04 on normalized predictions
- **Random Forest**: ~0.02-0.04 MAE
- **PyTorch LSTM**: ~0.015-0.03 MAE  
- **TensorFlow LSTM**: ~0.02-0.035 MAE
- **Vector Store**: 35+ embedded documents with <2s query response
- **Data Processing**: 60+ days of multi-feature cryptocurrency data

### Advanced Architecture Components
- **FAISS Vector Store**: Intelligent similarity search with sentence transformers
- **Sequential Chains**: Automated analysis-to-prediction workflows
- **Agent Tools**: CoinGecko API integration, volatility analysis, model predictions
- **Memory System**: JSON-persistent conversation buffer
- **Auto-fixing Dependencies**: Seamless Google Colab deployment

## 🛠️ Tech Stack

### Core ML/AI Framework
- **PyTorch** - Deep learning with LSTM networks
- **TensorFlow** - Alternative LSTM implementation
- **Scikit-learn** - Random Forest baseline and metrics
- **Transformers** - HuggingFace model ecosystem
- **PEFT & LoRA** - Parameter-efficient fine-tuning

### LangChain Ecosystem
- **LangChain Core** - Chains, agents, and memory management
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Text embeddings
- **HuggingFace Embeddings** - Semantic understanding

### Data & Visualization
- **NumPy & Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Advanced visualization
- **ipywidgets** - Interactive dashboard components
- **CoinGecko API** - Real-time cryptocurrency data

## 🚀 Quick Start

### One-Click Deployment

1. **Launch in Google Colab** (Recommended)
   ```
   Click the "Open In Colab" badge above
   ```

2. **Run the Complete Pipeline**
   ```python
   # Block 1: Auto-install dependencies (with conflict resolution)
   # Block 2: Execute full pipeline with LangChain integration
   ```

3. **Interact with Dashboard**
   - Use the interactive controls for model selection
   - Query the RAG system with natural language
   - Test LangChain agents with custom tools
   - View real-time predictions and analysis

### Local Installation

```bash
# Clone repository
git clone https://github.com/jeshwanth-A/defi_aiml.git
cd defi_aiml

# Install dependencies
pip install torch torchvision torchaudio
pip install tensorflow transformers sentence-transformers
pip install langchain langchain-community faiss-cpu
pip install numpy pandas matplotlib scikit-learn
pip install ipywidgets peft datasets accelerate

# Launch Jupyter
jupyter notebook Defi_Aiml.ipynb
```

## 📊 Dashboard Features

### Interactive Controls
- **Token Selector**: Choose from multiple cryptocurrencies
- **Model Selector**: Switch between PyTorch, TensorFlow, Random Forest
- **RAG Query Interface**: Ask intelligent questions about historical data
- **Agent Executor**: Test custom LangChain tools
- **Progress Tracking**: Real-time operation monitoring

### Visualization Suite
1. **Predictions vs Actual**: Multi-model comparison chart
2. **Error Analysis**: Absolute error tracking across models
3. **Feature Trends**: Sentiment and volatility analysis
4. **Performance Metrics**: Comparative MAE visualization

## 🧠 LangChain Integration Details

### RAG System Architecture
```python
# Vector Store Creation
vectorstore = FAISS.from_documents(
    documents,  # Historical data + news context
    HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
)

# Intelligent Querying
response = rag_query("What factors influenced recent price movements?")
```

### Custom Agent Tools
- **CoinGecko Tool**: Real-time cryptocurrency data fetching
- **Volatility Calculator**: Statistical analysis of price movements
- **Model Prediction Tool**: Direct access to trained models

### Sequential Chains
```python
# Analysis → Prediction Pipeline
sequential_chain = SequentialChain(
    chains=[analysis_chain, prediction_chain],
    input_variables=["data_summary"],
    output_variables=["analysis", "prediction"]
)
```

## 📈 Model Performance

### Ensemble Approach
The system employs three complementary models:

1. **Random Forest**: Baseline ensemble method
   - Fast training and inference
   - Good interpretability
   - Robust to outliers

2. **PyTorch LSTM**: Custom neural network
   - Flexible architecture
   - Advanced optimization
   - GPU acceleration support

3. **TensorFlow LSTM**: Production-ready alternative
   - Easy deployment
   - Comprehensive ecosystem
   - Model serving capabilities

### Feature Engineering
- **Price Features**: Current, lagged, and normalized values
- **Volume Analysis**: Trading volume patterns
- **Sentiment Scores**: Market sentiment indicators
- **Volatility Metrics**: Statistical volatility measures
- **Temporal Features**: Time-based patterns

## 🔧 Advanced Configuration

### Model Persistence
```python
# Automatic model saving/loading
save_folder = '/content/drive/MyDrive/crypto_project'
models = {
    'pytorch': 'model_pt.pth',
    'tensorflow': 'model_tf.h5',
    'llm': 'finetuned_llm/',
    'vectorstore': 'faiss_index/'
}
```

### Memory Management
```python
# Conversation persistence
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

## 📱 Usage Examples

### Basic Prediction
```python
# Load models and predict
model = load_pytorch_model()
prediction = model.predict_next_prices(days=7)
```

### RAG Querying
```python
# Ask intelligent questions
response = rag_query("How does sentiment affect price predictions?")
print(response)
```

### Agent Interaction
```python
# Execute agent tools
crypto_data = coingecko_tool.run('uniswap')
volatility = volatility_tool.run('100,102,98,105,103')
model_pred = prediction_tool.run('pytorch')
```

## 🎯 Future Enhancements

- [ ] **Multi-token Analysis**: Expand to portfolio-level predictions
- [ ] **Real-time Streaming**: Live data integration with WebSocket APIs
- [ ] **Advanced NLP**: GPT integration for enhanced explanations
- [ ] **Risk Management**: Volatility-adjusted position sizing
- [ ] **Backtesting Engine**: Historical strategy validation
- [ ] **API Deployment**: FastAPI service for production use

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: Revolutionary framework for LLM applications
- **HuggingFace**: Transformers and model ecosystem
- **CoinGecko**: Reliable cryptocurrency data API
- **FAISS**: Efficient similarity search capabilities
- **Google Colab**: Accessible GPU computing platform

## 📞 Contact

**Jeshwanth Anumala**
- 📧 Email: jeshwanthanumala@gmail.com
- 🐙 GitHub: [@jeshwanth-A](https://github.com/jeshwanth-A)
- 💼 Portfolio: [jeshwanth55.notion.site](https://jeshwanth55.notion.site/portfolio)
- 📍 Location: Hyderabad, Telangana

---

⭐ **Star this repository if you find it helpful!**

*Building the future of intelligent DeFi analytics, one prediction at a time.*

---

**Latest Update**: October 31, 2025 - Added comprehensive LangChain integration with RAG capabilities