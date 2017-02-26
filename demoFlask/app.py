from __future__ import division
from flask import Flask,jsonify,request
import pickle
from firebase import firebase
import random
import nltk
from nltk import word_tokenize,pos_tag
from nltk import *
import hashlib
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer("english",ignore_stopwords=True)
app = Flask(__name__)
f=firebase.FirebaseApplication('https://botguise.firebaseio.com')
no_packs=4
K=32
god='3nobNCGU6dTlKhHaZF9pwkJg6Ik1'
packclf=[]
for i in range(no_packs):
	try:
		fp=open(god+'.pkl','rb')
		tmpclf=pickle.load(f)
		fp.close()
		packclf.append(tmpclf)
	except:
		pass
def ELOrating(win,lose):
	r1=10**(win/400)
	r2=10**(win/400)
	e1=r1/(r1+r2)
	e2=1-e1
	win_up=win+K*(1-e1)
	lose_up=lose+K*(0-e2)
	return (win_up,lose_up)
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
    tree=chunk.ne_chunk(tagged_tokens)
    chunks=tree2conlltags(tree)
    for pw in chunks:
        feats["has_token({},{},{})".format(stemmer.stem(pw[0].lower()),pw[1],pw[2])]=True
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
    if len(toppers)==1 and toppers[0]>0.4:
      return True
    if len(toppers)>1 and toppers[0]>0.3:
      return True
    if n<15:
	return False    
    if len(toppers)>2:
        z=0
        for tp in toppers:
            z+=tp
        S=0
        for i in range(len(toppers)-1):
            cmp=1.0*(toppers[i]/toppers[i+1])
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
def godbrainquest(string,part_no):
	tpclf=packclf[part_no-1]
	key=tpclf.classify(string)
	nextdic=f.get('/users/'+god+'/brain/'+part+'/'+key+'/curr',None)
	nextdic=nextdic.items()
	g=random.choice(nextdic)
	repl=g[1]['sentence']
	return repl,True
def godbrain(string,part_no):
	tpclf=packclf[part_no-1]
	key=tpclf.classify(string)
	nextdic=f.get('/users/'+god+'/brain/'+part+'/'+key+'/curr',None)
	nextdic=nextdic.items()
	g=random.choice(nextdic)
	repl=g[1]['sentence']
	return repl,False
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
  part_no=int(part[-2:])
  try:
  	#obtaining user pack classifier
  	ufp=open(uid+part+'.pkl','rb')
  	uclf=pickle.load(ufp)
  	ufp.close()
  except:  	
  	#master brain pack
  	if uid==god:
  		uclf=ProBayesClassifier()
		str01='hi,train me master'
		str02='I dont know master'
		key01=uclf.postNewKey(str01)
		key02=uclf.postNewKey(str02)
		f.post('/users/'+uid+'/brain/'+part+'/'+key01+'/curr',{'sentence':str01})
		f.post('/users/'+uid+'/brain/'+part+'/'+key01+'/curr',{'key':key02,'rating':800})
                f.post('/users/'+uid+'/brain/'+part+'/'+key02+'/curr',{'sentence':str02})
                f.post('/users/'+uid+'/brain/'+part+'/'+key02+'/curr',{'key':key01,'rating':800})
	else:
		#loading pickle
		fp=open(god+part+'.pkl','rb')
		uclf=pickle.load(fp)
		fp.close()
		#getting json
		data=f.get('/users/'+god+'/brain/'+part,None)
		waste=f.put('/users/'+uid+'/brain',part,data)
	#connecting non-question to question
  if string=='98a633a4227ede00887ce0e76fbc98d8':
  	try:
			question_tags=['what','why','where','when','why','whom','did','who']
			quest=random.choice(question_tags)
			repl,reliabilit=godbrainquest(quest,part_no)
			return jsonify({'reply':repl,'reliability':reliabilit})
  	except:
			questions=['Which house was harry in?','Which is ur favourite part?','Do you like Hermione?','Do you like Malfoy?']
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
		try:
			repl,reliabilit=godbrain(string,part_no)
			return jsonify({'reply':repl,'reliability':reliabilit})
		except:
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
    return jsonify({'reply':'Nothing to say','reliability':False})
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
  currblob=request.json['currblob']
  trainblob=request.json['trainblob']
  part=request.json['part']
  ufp=open(uid+part+'.pkl','rb')
  uclf=pickle.load(ufp)
  ufp.close()
  key=uclf.classify(trainblob)
  reliability=uclf.outPerformAlgo(trainblob)
  update_choice=False
  if reliability:
    uclf.postKey(trainblob,key)
    newKey=key
  if not reliability:
    update_choice=True
    newKey=uclf.postNewKey(trainblob)
  #updating brain in firebase
  f.post('/users/'+uid+'/brain/'+part+'/'+newKey+'/curr',{'sentence':trainblob})
  #updating the classifier
  ufp=open(uid+part+'.pkl','wb')
  pickle.dump(uclf,ufp,-1)
  ufp.close()
  #linking prev group keydash with newKey 
  keydash=uclf.classify(prevblob)
  if update_choice:
  	f.post('/users/'+uid+'/brain/'+part+'/'+keydash+'/next',{'key':newKey,'rating':800})
  #receiving the score of winning and losing group
  win_key=newKey
  lose_key=uclf.classify(currblob)
  win_ws=None
  lose_ws=None
  win_score=0
  lose_score=0
  rootdic=f.get('/users/'+uid+'/brain/'+part+'/'+keydash+'/next',None)
  rootdic=rootdic.items()
  for ws,item in rootdic:
  	if item['key']==win_key:
		win_ws=ws
		win_score=item['rating']
	elif item['key']==lose_key:
		lose_ws=ws
		lose_score=item['rating']
  if win_ws and lose_ws and win_score and lose_score:
	win_score_update,lose_score_update=ELOrating(win_score,lose_score)
	f.patch('/users/'+uid+'/brain/'+part+'/'+keydash+'/next/'+win_ws,{'rating':win_score_update})
	f.patch('/users/'+uid+'/brain/'+part+'/'+keydash+'/next/'+lose_ws,{'rating':lose_score_update})

  return jsonify({'train':True})
@app.route('/link',methods=['POST'])
def linker():
  prevblob=request.json['prevblob']
  currblob=request.json['currblob']
  uid=request.json['user']
  part=request.json['part']
  ufp=open(uid+part+'.pkl','rb')
  uclf=pickle.load(ufp)
  ufp.close()
  prevkey=uclf.classify(prevblob)
  currkey=uclf.classify(currblob)
  rootdic=f.get('/users/'+uid+'/brain/'+part+'/'+prevkey+'/next',None)
  rootdic=rootdic.items()
  for ws,item in rootdic:
    if item['key']==currkey:
      return jsonify({'link':True})
  f.post('/users/'+uid+'/brain/'+part+'/'+prevkey+'/next',{'key':currkey,'rating':800})
if __name__ == '__main__':
    app.run(port=8000)
