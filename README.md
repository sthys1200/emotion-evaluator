# Emotion Evaluator

A sentiment analysis evaluation framework that compares different transformer-based models on movie review data. This project provides tools for benchmarking, testing, and deploying sentiment analysis models with a focus on DistilBERT and multilingual BERT variants.

## Features

- **Multiple Model Support**: Compare DistilBERT and multilingual BERT models
- **Automated Benchmarking**: Comprehensive evaluation with accuracy, precision, recall, and F1-score metrics
- **FastAPI Integration**: RESTful API for real-time sentiment prediction
- **CSV Processing**: Batch processing of review datasets with automatic encoding detection
- **Performance Tracking**: Speed benchmarking and detailed reporting

## Project Structure

```
emotion-evaluator/
├── data/
│   └── IMDB-movie-reviews.csv     # Dataset (semicolon-separated)
├── outputs/                       # Generated results and reports
├── scripts/
│   ├── benchmarking.py           # Model performance comparison
│   ├── api.py         # FastAPI web service
│   └── test_model_csv.py         # Batch CSV prediction
├── src/
│   ├── models/
│   │   └── emotion_evaluators.py # Model implementations
│   └── utils/
│       └── data_loader.py        # Data loading utilities
├── exploration.ipynb             # Data exploration notebook
└── requirements.txt              # Python dependencies
```

## Supported Models

### DistilBERT
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Output**: Binary sentiment (0=Negative, 1=Positive)
- **Strengths**: Fast inference, good accuracy on English text

### MultiBERT
- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Output**: Binary sentiment converted from 5-star rating (1-2=Negative, 3-5=Positive)
- **Strengths**: Multilingual support, handles diverse review formats

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sthys1200/emotion-evaluator.git
cd emotion-evaluator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure your dataset is in the `data/` directory as `IMDB-movie-reviews.csv`

## Usage

### 1. Data Exploration

Start with the Jupyter notebook to explore your dataset:
```bash
jupyter notebook exploration.ipynb
```

### 2. Batch Prediction on CSV

Process reviews from a CSV file:
```bash
# Using DistilBERT (default)
python -m scripts.test_model_csv

# Using MultiBERT
python -m scripts.test_model_csv --model MultiBERT

# Custom paths
python -m scripts.test_model_csv \
    --data_dir data \
    --dataset_file IMDB-movie-reviews.csv \
    --output_dir outputs \
    --output_file predictions.csv \
    --model DistilBERT
```

### 3. Model Benchmarking

Compare model performance:
```bash
python -m scripts.benchmarking
```

This generates a comprehensive report in `outputs/benchmark_report.txt` with:
- Accuracy, Precision, Recall, F1-score
- Processing speed comparison
- Detailed metrics for each model

### 4. API Service

Launch the FastAPI web service:
```bash
uvicorn scripts.test_model_api:app --reload
```

The API will be available at `http://localhost:8000` with:
- **POST** `/predict` - Analyze sentiment of text
- **GET** `/health` - Health check endpoint
- **GET** `/docs` - Interactive API documentation

#### API Usage Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was absolutely fantastic!"}
)
print(response.json())  # {"sentiment": "Positive"}
```

## Data Format

The project expects CSV files with the following structure:
- **Separator**: Semicolon (`;`)
- **Columns**: `review` (text), `sentiment` (positive/negative)
- **Encoding**: Auto-detected (typically Windows-1252 or UTF-8)

Example:
```csv
review;sentiment
"Great movie with excellent acting";positive
"Boring and poorly written";negative
```

## Model Performance

Based on benchmarking results, typical performance metrics:

| Model | Accuracy | Precision | Recall | F1-Score | Speed |
|-------|----------|-----------|--------|----------|-------|
| DistilBERT | ~85-90% | ~0.85 | ~0.85 | ~0.85 | Fast |
| MultiBERT | ~80-85% | ~0.80 | ~0.80 | ~0.80 | Moderate |

*Note: Actual performance depends on your specific dataset*

## Configuration Options

### Command Line Arguments

**test_model_csv.py**:
- `--data_dir`: Directory containing the dataset (default: "data")
- `--dataset_file`: CSV filename (default: "IMDB-movie-reviews.csv")
- `--output_dir`: Output directory (default: "outputs")
- `--output_file`: Output CSV filename (default: "output.csv")
- `--model`: Model to use ("DistilBert" or "MultiBert", default: "DistilBert")

**benchmarking.py**:
- `--data_dir`: Directory containing the dataset (default: "data")
- `--dataset_file`: CSV filename (default: "IMDB-movie-reviews.csv")
- `--output_dir`: Output directory (default: "outputs")
- `--output_file`: Report filename (default: "benchmark_report.txt")

## Development

### Adding New Models

1. Create a new class in `src/models/emotion_evaluators.py`:
```python
class YourModel(EmotionalEvaluator):
    def __init__(self):
        super().__init__("your-model-name")
    
    def predict_single(self, text):
        # Implement your prediction logic
        pass
```

2. Add it to the benchmarking script in `scripts/benchmarking.py`

### Custom Data Loaders

Extend `src/utils/data_loader.py` for different data formats:
```python
class CustomDataLoader(ReviewsDataLoader):
    def load_data(self):
        # Implement custom loading logic
        pass
```

## Troubleshooting

### Common Issues

1. **Encoding Errors**: The data loader automatically detects encoding, but if issues persist, manually specify encoding in the CSV reader.

2. **Memory Issues**: For large datasets, consider processing in batches by modifying the `predict_series` method.

3. **Model Download Issues**: Ensure internet connectivity for automatic model downloads from Hugging Face.

4. **API Port Conflicts**: Change the port using `uvicorn scripts.test_model_api:app --port 8001`

### Performance Tips

- Use DistilBERT for faster inference
- Process data in smaller batches for memory efficiency
- Consider GPU acceleration for large-scale processing

## Dependencies

Key libraries used:
- **transformers**: Hugging Face transformer models
- **fastapi**: Web API framework
- **pandas**: Data manipulation
- **scikit-learn**: Evaluation metrics
- **chardet**: Encoding detection
- **tqdm**: Progress bars

## License

This project is available under the MIT License. See the repository for full license details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the exploration notebook for data format examples
- Examine the benchmark report for performance baselines
Examine the benchmark report for performance baselines
