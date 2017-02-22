from firebase import firebase
import json
f=firebase.FirebaseApplication('https://botguise.firebaseio.com')
fp=open('pack01.txt','rU')
all=fp.readlines()
fp.close()
def entry(prevEntrys,currEntry):
    f.post('/data/pack01',{"prevEntrys":prevEntrys,"currEntry":currEntry})
for line in all:
    entry([],all.strip())

