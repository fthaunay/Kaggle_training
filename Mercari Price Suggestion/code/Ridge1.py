import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import regexp_tokenize
from nltk.stem.porter import PorterStemmer
import time
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def clean_missing(df):
    df["category_name"] = df.loc[:,"category_name"].replace(np.NaN, "")
    df["brand_name"] = df.loc[:,"brand_name"].replace(np.NaN, "")
    df["item_description"] = df.loc[:,"item_description"].replace(np.NaN, "")
    return df

def split_cat(text):
    if text.count('/') > 1:
        return text.split("/")
    else:
        return (["No Label", "No Label", "No Label"])

def transform_category_name(df):
    df.loc[:,'general_cat'], df.loc[:,'subcat_1'], df.loc[:,'subcat_2'] = \
    zip(*df['category_name'].apply(lambda x: split_cat(x)))
    return df

def stem_tokenize(text, stop_words=[]):
    stemmer = PorterStemmer()
    tokens = regexp_tokenize(text, pattern=r"[A-Za-z]\w+")
    tokens_wo_sw = [x for x in tokens if x not in stop_words and len(x) > 3]
    tokens_stemmed = [stemmer.stem(x) for x in tokens_wo_sw]
    return tokens_stemmed

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def main():
    start_time = time.time()
    data = pd.read_csv("../input/train.tsv", sep='\t')
    test = pd.read_csv("../input/test.tsv", sep='\t')
#    print('[{}] Finished to load train/test'.format(time.time() - start_time))

    sample_data = data[:100000]
    train = sample_data
#    train = clean_missing(train)
#    test = clean_missing(test)
#    train = transform_category_name(train)
#    test = transform_category_name(test)
    y_train = np.log1p(train['price'].as_matrix())

#    print('[{}] Finished to reshape train'.format(time.time() - start_time))
#    X_train = pd.get_dummies(train[['item_condition_id', 'shipping']]).as_matrix()
#    X_test = pd.get_dummies(test[['item_condition_id', 'shipping']]).as_matrix()
#    print('[{}] Finished get dummies'.format(time.time() - start_time))

#    tfidf_vectorizer = TfidfVectorizer(tokenizer=stem_tokenize, decode_error='ignore', strip_accents='unicode', max_df=0.8,
#                                       min_df=0.001)
#    tfidf_vectorizer.fit(train["category_name"])
#    X_train_cn = tfidf_vectorizer.transform(train["category_name"])
#    X_test_cn = tfidf_vectorizer.transform(test["category_name"])

#    print('[{}] Finished to create X_y_cn'.format(time.time() - start_time))

#    X_train = np.concatenate((X_train, X_train_cn.toarray()), axis=1)
#    X_test = np.concatenate((X_test, X_test_cn.toarray()), axis=1)

#    print('[{}] Finished to concatenate'.format(time.time() - start_time))

#    tfidf_vectorizer = TfidfVectorizer(tokenizer=stem_tokenize, decode_error='ignore', strip_accents='unicode', max_df=0.8,
#                                       min_df=0.001)
#    tfidf_vectorizer.fit(train["brand_name"])
#    X_train_bn = tfidf_vectorizer.transform(train["brand_name"])
#    X_test_bn = tfidf_vectorizer.transform(test["brand_name"])

#    print('[{}] Finished to create X_y_bn'.format(time.time() - start_time))

#    X_train = np.concatenate((X_train, X_train_bn.toarray()), axis=1)
#    X_test = np.concatenate((X_test, X_test_bn.toarray()), axis=1)

#    print('[{}] Finished to concatenate'.format(time.time() - start_time))

#    save_sparse_csr("../input/X_train_sparse.csv", csr_matrix(X_train))
#    save_sparse_csr("../input/X_test_sparse.csv", csr_matrix(X_test))



#    print('[{}] Finished to save X_y'.format(time.time() - start_time))

    X_train = load_sparse_csr("../input/X_train_sparse.csv.npz").toarray()
    X_test = load_sparse_csr("../input/X_test_sparse.csv.npz").toarray()

    regr = Ridge(alpha=1.0)
    regr.fit(X_train, y_train)

    print('[{}] Finished to fit regressor'.format(time.time() - start_time))

    y_pred = regr.predict(X_test)

    print('[{}] Finished to predict'.format(time.time() - start_time))

    submission = pd.DataFrame(test[["test_id"]])
    submission['price'] = np.expm1(y_pred)
    submission.to_csv("submission.csv", index=False)

    print('[{}] Finished to write submission file'.format(time.time() - start_time))


if __name__ == '__main__':
    main()