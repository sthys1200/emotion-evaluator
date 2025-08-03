
from transformers import pipeline
from tqdm import tqdm

class EmotionalEvaluator:
    """Base class"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipeline = pipeline("sentiment-analysis", model = model_name)

    #How singular label is outputted can be different per model (binary, score,...)
    def predict_single(self, text):
        label = self.pipeline(text)[0]['label']
        return label
    

    def predict_series(self, text_array):
        labels = []
        for text in tqdm(text_array, desc= f"Predicting using {self.model_name}", unit="item"):
            label = self.predict_single(text)
            labels.append(label)
        return labels


class DistilBert(EmotionalEvaluator):
    """DistilBERT sentiment analyzer"""   
    def __init__(self):
        super().__init__("distilbert-base-uncased-finetuned-sst-2-english")
    def predict_single(self, text):
        label = self.pipeline(text, max_length = 512)[0]['label']
        if label == 'POSITIVE':
            return 1
        else:
            return 0


class MultiBert(EmotionalEvaluator):
    """Multilingual BERT for product reviews"""   
    def __init__(self):
        super().__init__("nlptown/bert-base-multilingual-uncased-sentiment")       
    def predict_single(self, text):
        label = self.pipeline(text, max_length = 512)[0]['label']
        if int(label[0]) >= 3:
            return 1
        else:
            return 0
        
