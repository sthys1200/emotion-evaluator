from src.models.emotion_evaluators import DistilBert, MultiBert
from src.utils.data_loader import ReviewsDataLoader
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset_file", default = "IMDB-movie-reviews.csv")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--output_file", default = 'output.csv')
    parser.add_argument("--model", default = "DistilBert")
    args = parser.parse_args()

    #Determine model to use
    if args.model == "DistilBert":
        
        model = DistilBert()

    elif args.model == "MultiBert":

        model = MultiBert()
    
    #loading data
    data_path = args.data_dir + "/"+ args.dataset_file
    data_loader = ReviewsDataLoader(data_path)
    df = data_loader.load_data()

    #getting predictions
    input_texts = df['review'].to_list()
    preds = model.predict_series(input_texts)


    output_df = df.copy()
    output_df['predictions'] = preds

    output_path = args.output_dir +"/"+args.output_file
    output_df.to_csv(output_path, index = False)
    print(f"Results of prediction written to {output_path}!")

    
if __name__ == "__main__":
    print("Test")
    main()



