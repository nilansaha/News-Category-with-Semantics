## News-Category-with-Semantics

Classifying news category using word embeddings and cosine similarity

Supervised Learning although is really efficient has the same pipeline most of the time. For a change I wanted to see how far we could go using just
semantics in terms of embeddings and unsupervised similarity metrics to classify categories of the 20newsgroup dataset.

Disclaimer - Framing this as a supervised problem would yield much better results

### PseudoCode

```
function get_doc_embedding(doc){
  tokenize the doc
  for every word in the doc
    if the word is not in stopwords set
      then find its embedding and add it to the doc embedding
   return the overall doc embedding
   
function main {
  for every doc in corpus
    create a matrix from their doc embeddings
  calculate the cosine similarity between every document and store the matrix of the categories of the nearest docs
}

function evaluate {
  for every doc
    find the mode of the categories of the k most similar docs
    check if that category is the same as the actual category
}
```

### Performance Results

For different `k` values 

```
1 0.7520959354770244
2 0.7520959354770244
3 0.7416958505783721
4 0.7401040008489865
5 0.7277936962750716
6 0.7232834553751459
7 0.7182956595564045
8 0.7151650217552796
9 0.7115037673776928
10 0.708956807810676
```

As we can see both `k = 1` and `k = 2` performs the best so might as well go ahead and choose `k = 1`
The reason both of them have the same performance is because if the category of both the first and second nearest document is the same then that category is chosen however if they are different the category of the first document is chosen so in a sense for different categories it essentially just functions as `k = 1`
