# pFunK
A power framwork for functional PTMs prediction by H2.
We developed a hierarchical learning framework, namely prediction of functionally important lysine modification sites (pFunK).
## Model Architecture
By implementing a new hierarchical learning framework, pFunK consists of a lysine prediction model, a specific acylation sites predction model and a functional sites prediction model of the specific acylation.
+ pFunK-P, a model for extracting lysine contextual sequence features, consisting of Transformer algorithm.
+ pFunK-T, a model for extracting the specific acylation sites features, inculding the contextual sequence features, the sequence features, the structural feature. The model consists of Transformer algorithm and DNN algorithm, which are trained in the framwork under the MAML algorithm.
+ pFunK, a model for extracting the specific functional acylation sites features, which is fine-tuned by MAML from pFunK-T.

If you are interested in pFunK and want to know more details, please check the article:
