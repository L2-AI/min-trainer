from typing import Tuple
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

def get_newsgroup_df_and_target_names(
    subset:str='train',
    categories:str = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space'
    ]) -> Tuple[pd.DataFrame, list[str]]:
    """
    Loads a subset of the 20 Newsgroups dataset and returns it as a DataFrame 
    along with the target category names.

    Parameters:
        subset (str): Which subset of the dataset to load. Options are 'train', 'test', or 'all'. Default is 'train'.
        categories (list[str]): A list of category names to filter the dataset. Default includes 4 specific categories.

    Returns:
        Tuple[pd.DataFrame, list[str]]: 
            - A DataFrame with columns 'text' (truncated to 100 characters) and 'labels' (integer class labels).
            - A list of target category names corresponding to the labels.
    """
    newsgroups_train = fetch_20newsgroups(subset=subset, categories=categories)

    df = pd.DataFrame({
        'text': newsgroups_train.data,
        'labels': newsgroups_train.target,
        #'target_names': [newsgroups_train.target_names[i] for i in newsgroups_train.target]
    })
    df['text'] = df['text'].apply(lambda x: x[0:100]) # Truncate text to the first 100 characters

    return df, newsgroups_train.target_names