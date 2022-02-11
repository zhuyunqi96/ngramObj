## PyTorch implementation of Differentiable N-gram Objective on Abstractive Summarization

[Differentiable N-gram Objective on Abstractive Summarization](https://arxiv.org/abs/2202.04003)

### Dependencies of our run

Anaconda python=3.8

transformers 4.10.3

datasets 1.12.1

rouge-score 0.0.4

rouge 1.0.1

NLTK



### To run the code
CNN/DM

```bash
python bart_test run_cnndm.json
# update json file to fit your required batch size etc.
```

XSUM

```bash
python bart_test run_xsum.json
```



#### To control running with/without n-gram objective

```python
# line 1537 to line 1545 in bart_custom.py, in default.
USE_NgramsLoss = True # if False, any n-gram objective will be skipped
USE_1GramLoss = False
USE_2GramLoss = True
USE_3GramLoss = False
USE_4GramLoss = False
USE_5GramLoss = False
USE_BoN = False
USE_Ngrams_reward = False
USE_p2loss = False
```
to activate BoN, set USE_BoN as True, others will be skipped.

to activate P-P2, set USE_p2loss as True, make sure USE_BoN is False, and ten others will be skipped.

to activate n-gram rewards objective, set any of USE_XGramLoss as True, make sure USE_BoN and USE_p2loss are both False, and USE_Ngrams_reward as True.

to activate n-gram matches objective, set any of USE_XGramLoss as True, make sure USE_BoN and USE_p2loss are both False, and USE_Ngrams_reward as False



Code of bart model came from https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/modeling_tf_bart.py

Code of BoN came from https://github.com/ictnlp/BoN-NAT/blob/master/model.py

Code of P-P2 came from https://github.com/ictnlp/GS4NMT/blob/master/models/losser.py
