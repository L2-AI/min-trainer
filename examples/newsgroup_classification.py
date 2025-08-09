from mintrainer._sample_dataset_utils import get_newsgroup_df_and_targets
from mintrainer.main import MinTrainer

def main():
    train_df, categories = get_newsgroup_df_and_targets(subset='train')
    test_df, _ = get_newsgroup_df_and_targets(subset='test')

    MinTrainer(
        train_df = train_df,
        test_df = test_df,
        categories = categories,
        #cache_dir = # dir where HuggingFace transformers saves and reads models
        #finetuned_model_output_dir = # dir where finetuned model checkpoints are saved
    )

if __name__ == "__main__":
    main()
