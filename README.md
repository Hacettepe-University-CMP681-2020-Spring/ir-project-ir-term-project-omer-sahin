# Query Reformulation by Keyword Selection

Query Reformulation model for selecting keywords that provide more precision on fetching relevant documents that trained in the manner of reinforcement learning.

## Dataset

Jeopardy! TV Show

https://www.kaggle.com/tunguz/200000-jeopardy-questions

TREC - Complex Answer Retrieval (TREC-CAR)

http://trec-car.cs.unh.edu/

## Files

- [index_preprocess.py](index_preprocess.py) : Index query-title-documents
- [article.py](article.py) : Wikipedia article and QTA indexer classes
- [query.py](query.py) : Query class and query manager

- [search_engine](search_engine) :
    - [search.py](search_engine/search.py) : Search engine class
    - [rank_bm25.py](search_engine/rank_bm25.py) : BM25 implementation
   
- [model](model) : 
    - [embedding.py](model/embedding.py) : Word embedding class
    - [evaluate.py](model/evaluate.py) : Precision/Recall/NDCG evaluation
    - [preprocess.py](model/preprocess.py) : Preprocess for neural network
    - [query_reformulation.py](model/query_reformulation.py) : Query reformulation model 
    - [train.py](model/train.py) : Train model
    - [util.py](model/util.py) : Utils such as get batch data, recreate query, reward
    
## Indexed Data

- Indexed articles and queries
- Word embedding matrix and word tokenizer
- Search engine

https://drive.google.com/open?id=1xoquzwTFES00TFWYKkQ6KLhm7wlTGJtu

## Trained Models

- CNN, LSTM, BiLSTM and retrained CNN models

https://drive.google.com/open?id=1CT1HGvBhXMiTLeeZ6J6isxghHMylYBeM


## Usage
1. Index Dataset
    - Change paths in the [index_preprocess.py](index_preprocess.py) file and run with 'initial_run' true
 
2. Train Model
    - Set search engine path in [search.py](search_engine/search.py)
    - Set dataset path in [train.py](model/train.py) 
    - Set output path of the model in [train.py](model/train.py), select model network as CNN, LSTM or BiLSTM and run

3. Evaluate
    - Set search engine path in [search.py](search_engine/search.py)
    - Set path of the trained model in [evaluate.py](model/evaluate.py)
    - Set dataset path in [evaluate.py](model/evaluate.py) and run
    
You can start any step if you have the required files.    