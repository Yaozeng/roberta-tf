"""
import os
os.system("python run_classifier.py")
os.system("python run_classifier_align.py")
os.system("python eval_roberta.py")
os.system("python eval_roberta_align.py")
"""
import pandas as pd
import tokenization
import os
df=pd.read_csv("./data/train_merge.tsv",sep='\t',header=None)
df=df.sample(frac=1)
examples = []
print(len(df))
for i in range(0, len(df)):
    guid="%s-%s" % ("train", i)
    text_a = tokenization.convert_to_unicode(df.iloc[i][0])
    print(text_a)
    text_b = tokenization.convert_to_unicode(df.iloc[i][1])
    print(text_b)
    label = tokenization.convert_to_unicode(df.iloc[i][2].astype(str))
    print(label)