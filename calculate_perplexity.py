import kenlm

model=kenlm.Model("something.arpa") 
per=model.perplexity("your text sentance")

print(per)