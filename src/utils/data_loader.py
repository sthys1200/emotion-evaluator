
import pandas as pd
import chardet

class ReviewsDataLoader:

    """Data loader for review dataset"""

    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path

    def load_data(self):

        print("Analysing encoding of data...")
        
        #to avoid encoding issues, detect encoding
        with open(self.data_dir_path, 'rb') as f:
            result = chardet.detect(f.read(100000))  # Reads first 100KB
            encoding = result['encoding']
        print(f"Detected encoding: {encoding}")
        
        print("Loading data...")

        #read data as a DataFrame
        self.df = pd.read_csv(self.data_dir_path, encoding=encoding, sep=';')

        #Clean up: removing HTML indicators and convert label strings to 0 or 1
        self.df['review'] = self.df['review'].str.replace(r'<.*?>', ' ', regex=True)
        self.df['sentiment'] = self.df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

        print("Data loaded!")
        return self.df
    
        
