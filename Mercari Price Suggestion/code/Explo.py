import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import regexp_tokenize
from nltk.stem.porter import PorterStemmer
import time
from scipy.sparse import csr_matrix

def main():
    start_time = time.time()

    train = pd.read_csv("../input/train.tsv", sep='\t')
    train = train
    test = pd.read_csv("../input/test.tsv", sep='\t')
    test = test
    print('[{}] Finished to load train/test'.format(time.time() - start_time))

    def clean_missing(df):
        df["category_name"] = df["category_name"].replace(np.NaN, "")
        df["brand_name"] = df["brand_name"].replace(np.NaN, "")
        df["item_description"] = df["item_description"].replace(np.NaN, "")
        df["item_description"] = df["item_description"].replace(np.NaN, "")
        return df


    train = clean_missing(train)
    test = clean_missing(test)

    print('[{}] Finished to replace NaN'.format(time.time() - start_time))

    def split_cat(text):
        if text.count('/') > 1:
            return text.split("/")
        else:
            return (["No Label", "No Label", "No Label"])

    def transform_category_name(df):
        df['general_cat'], df['subcat_1'], df['subcat_2'] = \
        zip(*df['category_name'].apply(lambda x: split_cat(x)))
        return df

    train = transform_category_name(train)
    test = transform_category_name(test)

    print('[{}] Finished to transform category_name'.format(time.time() - start_time))


    stop_words = []
    def stem_tokenize(text, stop_words=stop_words):
        stemmer = PorterStemmer()
        tokens = regexp_tokenize(text, pattern=r"[A-Za-z]\w+")
        tokens_wo_sw = [x for x in tokens if x not in stop_words and len(x) > 3]
        tokens_stemmed = [stemmer.stem(x) for x in tokens_wo_sw]
        return tokens_stemmed

    # category_name :
#    tfidf_vectorizer = TfidfVectorizer(tokenizer=stem_tokenize, decode_error='ignore', strip_accents='unicode', max_df=0.95, min_df=0.01)
#    tfidf_vectorizer.fit(train["category_name"])
#    X1 = tfidf_vectorizer.transform(train["category_name"]).toarray()
#    X11 = tfidf_vectorizer.transform(test["category_name"]).toarray()
#    np.savetxt("X1.csv", X1, delimiter=",")
#    np.savetxt("X11.csv", X11, delimiter=",")

#    X1 = np.loadtxt("X1.csv", delimiter=",")
#    X11 = np.loadtxt("X11.csv", delimiter=",")
#    print('[{}] Finished to create X1'.format(time.time() - start_time))


    # brand_name :
#    tfidf_vectorizer = TfidfVectorizer(tokenizer=stem_tokenize, decode_error='ignore', strip_accents='unicode', max_df=0.95, min_df=0.01)
#    tfidf_vectorizer.fit_transform(train["brand_name"])
#    X2 = tfidf_vectorizer.transform(train["brand_name"]).toarray()
#    X22 = tfidf_vectorizer.transform(test["brand_name"]).toarray()
#    np.savetxt("X2.csv", X2, delimiter=",")
#    np.savetxt("X22.csv", X22, delimiter=",")

#    X2 = np.loadtxt("X2.csv", delimiter=",")
#    X22 = np.loadtxt("X22.csv", delimiter=",")
#    print('[{}] Finished to create X2'.format(time.time() - start_time))


    # item_description
    stop_words_id = ["No", "description", "yet"]
    tfidf_vectorizer = TfidfVectorizer(tokenizer=stem_tokenize, decode_error='ignore', strip_accents='unicode', stop_words=stop_words_id, max_df=0.80, min_df=0.05)
    tfidf_vectorizer.fit(train["item_description"])
    X3 = tfidf_vectorizer.transform(train["item_description"])
    X33 = tfidf_vectorizer.transform(test["item_description"])

    def save_sparse_csr(filename, array):
        np.savez(filename, data=array.data, indices=array.indices,
                 indptr=array.indptr, shape=array.shape)

    save_sparse_csr("X3_sparse.csv", X3)
    save_sparse_csr("X33_sparse.csv", X33)

#    np.savetxt("X3.csv", X3, delimiter=",")
#    np.savetxt("X33.csv", X33, delimiter=",")
    print('[{}] Finished to create X3'.format(time.time() - start_time))

    X4 = pd.get_dummies(train[['item_condition_id', 'shipping']]).as_matrix()
    X44 =pd.get_dummies(test[['item_condition_id', 'shipping']]).as_matrix()
    np.savetxt("X4.csv", X4, delimiter=",")
    np.savetxt("X44.csv", X44, delimiter=",")
    print('[{}] Finished to create X4'.format(time.time() - start_time))


    X_train = np.concatenate(((X1, X2, X3.toarray(), X4)), axis=1)
    X_test = np.concatenate(((X11, X22, X33.toarray(), X44)), axis=1)

    print('[{}] Finished to create X'.format(time.time() - start_time))


    np.savetxt("x_train.csv", X_train, delimiter=",")
    np.savetxt("x_test.csv", X_test, delimiter=",")

    print('[{}] Finished to save X'.format(time.time() - start_time))

if __name__ == '__main__':
    main()