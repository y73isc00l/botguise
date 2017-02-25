from flask import Flask,jsonify,request
import pickle
from firebase import firebase
import random
import nltk
from nltk import word_tokenize,pos_tag
import hashlib
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
app = Flask(__name__)
f=firebase.FirebaseApplication('https://botguise.firebaseio.com')
def npcollocation(doc):
    blob=TextBlob(doc)
    tokens=word_tokenize(doc)
    np=blob.noun_phrases
    np_tokenize=[]
    for phrase in np:
        np_tokenize.append((phrase,word_tokenize(phrase)))
    for phrase,nptokens in np_tokenize:
        sz=len(nptokens)
        for i in range(len(tokens)-sz+1):
            if nptokens==tokens[i:i+sz]:
                tokens[i:i+sz]=[phrase]
                break
    ##converting all tokens ex
    tagged_tokens=nltk.pos_tag(tokens)
    final_tokens=[]
    noun_types=['NN','NNP','NNS']
    for token,pos in tagged_tokens:
        if pos=='.':
            continue
        if not pos in noun_types:
            final_tokens.append(token.lower())
        else:
            final_tokens.append(token)
    return final_tokens
def fe3000(document):
    feats={}
    
    tokens=npcollocation(document)
    tagged_tokens=pos_tag(tokens)
    for pw in tagged_tokens:
        feats["has_token({},{})".format(pw[0],pw[1])]=True
    return feats
def hashfun(num):
    m=hashlib.md5()
    m.update(str(random.random()))
    m.update(str(num))
    return m.hexdigest()
def extOutPerformAlgo(prob_dist,ref,threshhold):
    toppers=[]
    n=len(ref.keys())
    for key in ref.keys():
        if prob_dist.prob(key)>threshhold:
            toppers.append(prob_dist.prob(key))
    toppers.sort(reverse=True)
#     if len(toppers)==1 and toppers[0]>0.5:
#         return True
#     if len(toppers)>1 and toppers[0]>0.3:
#         return True
    if n<15:
	return False    
    if len(toppers)>2:
        z=0
        for tp in toppers:
            z+=tp
        S=0
        for i in range(len(toppers)-1):
            cmp=1.0*(toppers[i]/toppers[i+1]-1)
            if cmp>1.20:
                c=0
            else:
                c=1
            S+=((-1)**(i+c))*cmp*1.0
        if S>0:
            return True
        else:
            return False
    else:
        return False
    return False
class ProBayesClassifier(NaiveBayesClassifier):
    def __init__(self):
        NaiveBayesClassifier.__init__(self,[],feature_extractor=fe3000)
        self.ref={}
        self.threshhold=0.06
#     def update_store(self,doc_train):
#         key=hashfun(random.random())
#         self.update([(doc_train,key)])
#         self.ref.setdefault(key,[]).append(doc_train)
#     def update_store_key(self,doc_train,key):
#         self.update([(doc_train,key)])
#         self.ref.setdefault(key,[]).append(doc_train)
    def postKey(self,doc_train,key):
        self.update([(doc_train,key)])
    def postNewKey(self,doc_train):
        key=hashfun(random.random())
        self.update([(doc_train,key)])
        self.ref.setdefault(key,[])
        return key
    def outPerformAlgo(self,doc_test):
        prob_dist=self.prob_classify(doc_test)
        return extOutPerformAlgo(prob_dist,self.ref,self.threshhold)
fp=open('classifier.pkl','rb')
clf=pickle.load(fp)
fp.close()

@app.route('/')
def index():
    return "<h1>This flask app is running!</h1>"
@app.route('/jsonmes')
def test():
    return jsonify({'message':word_tokenize("Hi how are you doing")})
@app.route('/PBC',methods=['POST'])
def reply():
  uid=request.json['user']
  string=request.json['message']
  part=request.json['part']
  try:
    #obtaining user pack classifier
    ufp=open(uid+part+'.pkl','rb')
    uclf=pickle.load(ufp)
    ufp.close()
  except:
    #initialising user pack classifier
    uclf=ProBayesClassifier()
    str01='Ask me something'
    str02='I dont know'
    key01=uclf.postNewKey(str01)
    key02=uclf.postNewKey(str02)
    f.post('/users/'+uid+'/brain/'+part+'/'+key01+'/curr',{'sentence':str01})
    f.post('/users/'+uid+'/brain/'+part+'/'+key01+'/next',{'key':key02,'rating':800})
    f.post('/users/'+uid+'/brain/'+part+'/'+key02+'/curr',{'sentence':str02})
    f.post('/users/'+uid+'/brain/'+part+'/'+key02+'/next',{'key':key01,'rating':800})
  #connecting non-question to question
  if string=='98a633a4227ede00887ce0e76fbc98d8':
		questions=['Which house was harry in?','Which is ur favourite part?','Do you like Hermione?']
		return jsonify({'reply':random.choice(questions),'reliability':True})  
	#classifying blob
  key=uclf.classify(string)
  reliability=uclf.outPerformAlgo(string)
  if reliability:
    uclf.postKey(string,key)
    newKey=key
  if not reliability:
    newKey=uclf.postNewKey(string)
  #updating brain in firebase
  f.post('/users/'+uid+'/brain/'+part+'/'+newKey+'/curr',{'sentence':string})
  #updating the classifier
  ufp=open(uid+part+'.pkl','wb')
  pickle.dump(uclf,ufp,-1)
  ufp.close()
  #reply by next linking
  nextdic=f.get('/users/'+uid+'/brain/'+part+'/'+key+'/next',None)
  if not nextdic:
		questions=['Which house was harry in?','Which is ur favourite part?','Do you like Hermione?']
		return jsonify({'reply':random.choice(questions),'reliability':True})  

  nextdic=nextdic.items()
  X=[]
  for item in nextdic:
    X.append(item[1])
  optkey=random.choice(X)
  nextKey=optkey['key']
  replys=f.get('/users/'+uid+'/brain/'+part+'/'+nextKey+'/curr',None)
  if not replys:
    return jsonify({'reply':'Nothing to say','reliability':reliability})
  replys=replys.items()
  foo=random.choice(replys)
  reply=foo[1]['sentence']
  return jsonify({'reply':reply,'reliability':reliability})
#   key=clf.classify(string)
#   rep=clf.ref[key]
#   foo=clf.outPerformAlgo(string)
@app.route('/train',methods=['POST'])
def train():
  uid=request.json['user']
  prevblob=request.json['prevblob']
  trainblob=request.json['trainblob']
  part=request.json['part']
  ufp=open(uid+part+'.pkl','rb')
  uclf=pickle.load(ufp)
  ufp.close()
  key=uclf.classify(trainblob)
  reliability=uclf.outPerformAlgo(trainblob)
  if reliability:
    uclf.postKey(trainblob,key)
    newKey=key
  if not reliability:
    newKey=uclf.postNewKey(trainblob)
  #updating brain in firebase
  f.post('/users/'+uid+'/brain/'+part+'/'+newKey+'/curr',{'sentence':trainblob})
  #updating the classifier
  ufp=open(uid+part+'.pkl','wb')
  pickle.dump(uclf,ufp,-1)
  ufp.close()
  #linking prev group keydash with newKey 
  keydash=uclf.classify(prevblob)
  f.post('/users/'+uid+'/brain/'+part+'/'+keydash+'/next',{'key':newKey,'rating':800})
  return jsonify({'train':True})
if __name__ == '__main__':
    app.run(port=8000)
