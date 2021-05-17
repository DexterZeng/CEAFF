# CEAFF

Source code and datasets for TOIS 2021 paper: ***[Reinforcement Learning–based Collective Entity Alignment with Adaptive Features](https://dl.acm.org/doi/10.1145/3446428)***.

Initial datasets are from [GCN-Align](https://github.com/1049451037/GCN-Align) and [JAPE](https://github.com/nju-websoft/JAPE).

## TODO

- [ ] Will further optimize the codes in early June.

## Dependencies

* Python=3.6
* Tensorflow-gpu=1.13.1
* Scipy
* Numpy
* Scikit-learn
* python-Levenshtein

## Datasets
There are nine datasets in this folder:
- zh_en/ja_en/fr_en from the [DBP15K dataset](https://github.com/nju-websoft/BootEA)
- en_fr/en_de/dbp_wd/dbp_yg from the [SRPRS dataset](https://github.com/nju-websoft/RSN)
- [DBP-FB dataset](https://github.com/DexterZeng/EAE)
- wd-imdb dataset created by us

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_2: ids for entities in target KG (EN);
* ill_ent_ids: entity links encoded by ids;
* ref_ent_ids: entity links for testing/validation;
* sup_ent_ids: entity links for training;
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);
* zh_vectorList.json: the input entity feature matrix initialized by word vectors;

### Semantic Information
Regarding the Semantic Information, we obtain the entity name embeddings for DBP15K from [RDGCN](https://github.com/StephanieWyt/RDGCN), 
the entity name embeddings for SRPRS from [CEA](https://github.com/DexterZeng/CEA). 
During the experiment, we found that for dbp_wd, dbp_yg and dbp_fb dataset, using the pre-trained [fastText word embeddings with 
*subword* information](https://fasttext.cc/docs/en/english-vectors.html) are more effective and use them instead. 

We also provide the datasets with entity name embeddings in [Baidu Netdisk](https://pan.baidu.com/s/1dhHE-zbOYN8rtBnFIo5d4w) (Code: s6n1). You can download it and extract it into `data/` directory.


## Running
* First generate the string similarity by running `python stringsim.py --lan dbp_wd` for entities in the test set and validation set in advance since it costs *way too much* time!
* The dataset could be chosen from `zh_en, ja_en, fr_en, en_fr, en_de, dbp_wd, dbp_yg, dbp_fb`
* Then run

```
python Ada.py --lan dbp_wd --method braycurtis --mode first --vali notgen
```
* The metric (method) could be chosen from `cosine, braycurtis, cityblock, euclidean`
* Record the values of `Total Match` and `Total Match True`, which represent the entity pairs detected by preliminary treatment and the amount of correct matches.
* Then run
```
python RL.py --lan fr_en --method braycurtis --type test
```
* `Averaged correct matches` represent the averaged number of correct matches detected by the RL algorithm for the rest of the entities. 
* Adding `Total Match True` and `Averaged correct matches`, and dividing the value from the number of test entities will give the precision results. 
> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit  when running code repeatedly.

> If you have any questions about reproduction, please feel free to email to zengweixin13@nudt.edu.cn.

## Citation

If you use this model or code, please cite it as follows:

* Weixin Zeng, Xiang Zhao, Jiuyang Tang, Xuemin Lin, and Paul Groth. 2021. Reinforcement Learning–based Collective Entity Alignment with Adaptive Features. ACM Trans. Inf. Syst. 39, 3, Article 26 (May 2021), 31 pages. DOI:https://doi.org/10.1145/3446428

```
@article{10.1145/3446428,
    author = {Zeng, Weixin and Zhao, Xiang and Tang, Jiuyang and Lin, Xuemin and Groth, Paul},
    title = {Reinforcement Learning–Based Collective Entity Alignment with Adaptive Features},
    year = {2021},
    volume = {39},
    number = {3},
    url = {https://doi.org/10.1145/3446428},
    doi = {10.1145/3446428},
    journal = {ACM Trans. Inf. Syst.},
}
```
