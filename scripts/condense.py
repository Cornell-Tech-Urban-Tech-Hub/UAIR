import pandas as pd 
from glob import glob 
from pandarallel import pandarallel
from tqdm import tqdm

import sys
sys.path.append("/share/ju/matt/sensing-ai-risks/")
from utils.logger import setup_logger
logger = setup_logger()
logger.setLevel("INFO")


logger.info("Loading all article metadata via glob.")
all_articles_glob = glob("/share/ju/matt/sensing-ai-risks/data/global_subset/*/content/20*/*.txt")
logger.success(f"Found {len(all_articles_glob)} articles.")

all_articles = pd.DataFrame(all_articles_glob, columns=["article_path"])
all_articles['country'] = all_articles['article_path'].str.split('/').str[-4]
all_articles['year'] = all_articles['article_path'].str.split('/').str[-2]
logger.info(f"Added country and year columns.")


logger.info(all_articles.head().to_string())
logger.info(f"Total articles: {len(all_articles)}")

all_articles.to_parquet("../data/global_subset/all_articles_meta.parquet")
logger.success("Saved all article metadata to parquet at ../data/global_subset/all_articles_meta.parquet.")

def store_article_text(article_path):
    with open(article_path, "r") as f:
        text = f.read()

        return text

logger.info("Storing article text in parallel.")
tqdm.pandas(leave=True)
pandarallel.initialize(progress_bar=True, nb_workers=8)
all_articles['article_text'] = all_articles['article_path'].parallel_apply(store_article_text)  

all_articles.to_parquet("../data/global_subset/all_articles.parquet")
logger.success("Saved all article text to parquet at ../data/global_subset/all_articles.parquet.")







