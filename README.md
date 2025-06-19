# Intent Classification System

An intelligent text classification system that categorizes student queries into three types: course equivalency, program pathway, and others. The system uses vector similarity search for caching to reduce computational costs and improve response times.

## Features

- **Caching**: Uses Weaviate vector database to cache responses and return similar results without hitting the LLM
- **Custom LLM**: Fine-tuned Llama 3.1 8B model with specific system prompts for accurate classification
- **Real-time Web Interface**: Clean chat interface with adjustable similarity threshold
- **REST API**: FastAPI backend with automatic documentation
- **Evaluation Tools**: Built-in testing and classification reporting

## Tech Stack

- **Backend**: FastAPI, Python
- **Vector Database**: Weaviate Cloud
- **LLM**: Ollama with Llama 3.1 8B
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Testing**: scikit-learn, pandas

## Categories

1. **Course Equivalency**: Questions about credit transfer between institutions (like Assist.org)
2. **Program Pathway**: Questions about degree requirements and course sequences (like Program Mapper)
3. **Others**: General campus questions (financial aid, parking, services, etc.)

## Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed locally
- Weaviate Cloud account (free tier works)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd intent-classification
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn python-dotenv ollama-python weaviate-client pandas scikit-learn requests
   ```

3. **Set up environment variables**
   Create a `.env` file:
   ```
   WEAVIATE_URL=your_weaviate_cluster_url
   WEAVIATE_API_KEY=your_weaviate_api_key
   ```

4. **Create the Ollama model**
   ```bash
   ollama create intent -f modelfile
   ```

5. **Set up Weaviate collection**
   ```bash
   python test_weaviate.py
   ```

### Running the System

1. **Start the FastAPI server**
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

2. **Open the frontend**
   - Serve `frontend/index.html` on `http://127.0.0.1:5500` (using Live Server or similar)
   - Or open directly in browser for local testing

3. **Test the system**
   ```bash
   python test_intent.py
   ```

## Project Structure

```
├── main.py                     # FastAPI backend
├── frontend/
│   └── index.html             # Web interface
├── test_intent.py             # Testing script
├── test_weaviate.py           # Database setup
├── modelfile                  # Ollama model configuration
├── IntentDetection.csv        # Dataset (89 samples)
├── classification_results2.csv # Test results
└── README.md
```

## API Usage

### Chat Endpoint
```bash
POST http://127.0.0.1:8000/chat
Content-Type: application/json

{
  "prompt": "How do I transfer my calculus credits?",
  "similarity_threshold": 0.5
}
```

### Response Format
```json
{
  "prompt": "How do I transfer my calculus credits?",
  "response": "course equivalency",
  "status": "cached_response",
  "similarity": 0.65,
  "stored_in_weaviate": true
}
```

## How It Works

1. **User sends a query** through the web interface or API
2. **Similarity search** checks Weaviate for similar previous queries
3. **If similar found** (above threshold): return cached response
4. **If not found**: send to Ollama LLM for classification
5. **Store result** in Weaviate for future caching
6. **Return classification** with metadata

## Performance

- **Accuracy**: ~96% on test dataset
- **Caching**: ~30-40% of queries use cached responses after initial data buildup
- **Response Time**: <100ms for cached, ~2-3s for new classifications

## Testing

Run the evaluation script to test all prompts in the dataset:

```bash
python test_intent.py
```

This will:
- Send all 89 test prompts to the API
- Compare predictions with true labels  
- Generate a detailed classification report
- Save results to `classification_results2.csv`

## Configuration

### Similarity Threshold
- **Low (0.2-0.4)**: More caching, risk of false positives
- **Medium (0.5-0.6)**: Balanced approach (recommended)
- **High (0.8-1.0)**: No caching, higher precision but computationally expensive

### Model Temperature
Set in `modelfile` (currently 0.1 for deterministic results)
