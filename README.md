# test_model
From scratch encoder model without pretraining, to be trained on a small data set.
Create model_a package to house classes.

# PURPOSE
practice making a model from scratch without starting with a pretrained model.

# GLOSSARY
- exemplar: single training item, eg article, comment, document, posting
- freq: frequency

# EVALUATION
- the data set should be divided into training and test segments
- write routine that takes the model and the test segments and uses masking to
  calculate an overall predictive success rate.

# COMPONENTS

## tokenizer:
- BPE tokenizer
- vocabulary size default 8000
- has a method gather_stats that collects
  = global freq
  = exemplar freq (a count of how many exemplars have this token at least once in them)
- method to provide global freq given a token id
- method to provide exemplar freq given a token id

## 