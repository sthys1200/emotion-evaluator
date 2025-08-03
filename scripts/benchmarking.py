from src.models.emotion_evaluators import DistilBert, MultiBert
from src.utils.data_loader import ReviewsDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset_file", default = "IMDB-movie-reviews.csv")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--output_file", default = 'benchmark_report.txt')
    args = parser.parse_args()


    data_path = args.data_dir + "/"+ args.dataset_file
    output_path = args.output_dir + "/" + args.output_file

    data_loader = ReviewsDataLoader(data_path)
    df = data_loader.load_data()


    input_texts = df['review'].to_list()
    labels = df['sentiment'].to_list()


    models = {
        'DistilBert': DistilBert(),
        'MultiBert': MultiBert()
    }

    results = {}

    for name, model in models.items():

        start_time = time.time()
        #getting predictions
        preds = model.predict_series(input_texts)
        end_time = time.time()

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)


        results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Speed': end_time - start_time
    }
        
        with open(output_path, "a") as file:  # "a" appends to the file
            file.write(f"Results for {name}:\n")
            for metric, value in results[name].items():
                file.write(f"  {metric}: {value}\n")
            file.write("\n")  # Add a blank line for readability
        print(f"Results for {name} written to {output_path}!")

if __name__ == "__main__":
    main()