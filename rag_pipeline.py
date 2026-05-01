
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGPipeline:
    def __init__(self):
        self.docs=[]
        self.vectorizer=TfidfVectorizer()
        self.vec=None

    def add_documents(self,docs):
        self.docs.extend(docs)
        self.vec=self.vectorizer.fit_transform(self.docs)

    def answer(self,q):
        qv=self.vectorizer.transform([q])
        scores=cosine_similarity(qv,self.vec).flatten()
        idx=scores.argmax()
        return {"question":q,"context":self.docs[idx]}
