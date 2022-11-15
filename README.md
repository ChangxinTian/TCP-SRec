# Temporal Contrastive Pre-Training for Sequential Recommendation (TCP-SRec)

This is our Pytorch implementation for the paper:

> Changxin Tian, Zihan Lin, Shuqing Bian, Jinpeng Wang, Wayne Xin Zhao. "Temporal Contrastive Pre-Training for Sequential Recommendation." CIKM 2022.

## Introduction
Recently, pre-training based approaches are proposed to leverage self-supervised signals for improving the performance of sequential recommendation. However, most of existing pre-training recommender systems simply model the historical behavior of a user as a sequence, while lack of sufficient consideration on temporal interaction patterns that are useful for modeling user behavior. In order to better model temporal characteristics of user behavior sequences, we propose a Temporal Contrastive Pre-training method for Sequential Recommendation (TCPSRec for short). Based on the temporal intervals, we consider dividing the interaction sequence into more coherent subsequences, and design temporal pre-training objectives accordingly. Specifically, TCPSRec models two important temporal properties of user behavior, i.e., invariance and periodicity. For invariance, we consider both global invariance and local invariance to capture the long-term preference and short-term intention, respectively. For periodicity, TCPSRec models coarse-grained periodicity and fine-grained periodicity at the subsequence level, which is more stable than modeling periodicity at the item level. By integrating the above strategies, we develop a unified contrastive learning framework with four specially designed pre-training objectives for fusing temporal information into sequential representations. We conduct extensive experiments on six real-world datasets, and the results demonstrate the effectiveness and generalization of our proposed method.

## Requirements:
* Python=3.7.10
* PyTorch=1.7.0
* cudatoolkit=10.1.243
* pandas=1.3.5
* numpy=1.21.5
* recbole=1.0.0

## Dataset:

For `Yelp`, `Amazon-Beauty/Sports/Toys` and `MovieLens-1M`, you can download from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets). Then, you can refer to `utils_data.ipynb` to process the dataset. The processed dataset should be stored in the folder `./dataset/`.

Due to policy restrictions, the industrial dataset `Meituan` will not be released.


## Training:

```
# run TCP-SRec
python -u run_cikm.py

# run SASRec
python -u run_cikm.py --model=SASRec
```

## Reference:
Any scientific publications that use our codes should cite the following paper as the reference:

 ```
@inproceedings{tian2022tcpsrec,
    author = {Tian, Changxin and Lin, Zihan and Bian, Shuqing and Wang, Jinpeng and Zhao, Wayne Xin},
    title = {Temporal Contrastive Pre-Training for Sequential Recommendation},
    year = {2022},
    booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
    pages = {1925–1934},
    numpages = {10},
    series = {CIKM '22}
}

@inproceedings{zhao2021recbole,
    title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
    author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
    booktitle={{CIKM}},
    year={2021}
}
 ```

If you have any questions for our paper or codes, please send an email to tianchangxin@ruc.edu.cn.
