---
title: "A simple review for User Sequence Modeling"
collection: machine_learning
type: "Undergraduate course"
permalink: /machine_learning/uplift_survey
#venue: "Beijing"
date: 2025-02-16
---



## User Sequence Behavior Modelling Taxmony

|      | 名称          | 时间 | 来源         | 思路                                                         | 序列组织形式                        |
| ---- | ------------- | ---- | ------------ | ------------------------------------------------------------ | ----------------------------------- |
| 1    | YoutubeDNN    | 2016 | Youtube      | embedding+pooling                                            | 单- short 序列                      |
| 2    | DIN           | 2016 | 阿里巴巴     | embedding+ target attention                                  | 单- short 序列                      |
| 3    | DIEN          | 2019 | 阿里巴巴     | embedding+ GRU+ target attention                             | 单-short 序列                       |
| 4    | BST           | 2019 | 阿里巴巴     | embedding+transformer+ position embedding                    | 单-short 序列                       |
| 5    | TiSSA         | 2019 | 阿里巴巴     | embedding+RNN + transformer                                  | 单-short 序列+ session 序列         |
| 6    | DIPN          | 2019 | 阿里巴巴     | multi-task+ self-attention                                   | 多行为序列                          |
| 7    | SIM           | 2020 | 阿里巴巴     | embedding+General Search Unit+target attention               | 单-long序列                         |
| 8    | LALI          | 2020 | 阿里巴巴 ATA | embedding+two-stage attention                                | 单-long 序列                        |
| 9    | DFN           | 2020 | 微信         | 正负类型序列的 target-attention+Wide,FM,Deep                 | 多种类型-short 序列                 |
| 10   | KFATT         | 2020 | 京东         | Kalman Filtering + Target-attention                          | 多种类型的-short 序列               |
| 11   | ETA           | 2021 | 阿里巴巴     | two-stage + Target-attention                                 | 长行为序列+短行为序列               |
| 12   |               | 2022 | 高德         | Target-attention + wide & Deep                               | 多种类型的序列(历史好序列+短期序列) |
| 13   | SDIM          | 2022 | 美团         | Two stage +Target-attention                                  | 长行为序列                          |
| 14   | nsTransformer | 2022 | 滴滴         | Transformers                                                 | 短行为序列                          |
| 15   | QIN           | 2023 | 快手         | two stage + target-attention + self-attention + ID&attr embedding | 长行为序列                          |
| 16   | TransAct      | 2023 | Pinterest    | target-attention + transformer + embedding + concat          | 长行为序列+短行为序列               |
| 17   |               | 2024 | 打车风控     | Conv->LSTM->self-attention                                   |                                     |





1. Deep Neural Networks for YouTube Recommendations
2. Deep Interest Network for Click-Through Rate Prediction
3. Deep Interest Evolution Network for Click-Through Rate Prediction
4. Behavior Sequence Transformer for E-commerce Recommendation in Alibaba
5. A Time Slice Self-Attention Approach for Modeling Sequential User Behaviors
6. Buying or Browsing? : Predicting Real-time Purchasing Intent using Attention-based Deep Network with Multiple Behavior
7. *Search-based User Interest Modeling with Lifelong Sequential* *Behavior Data for Click-Through Rate Prediction*
8. *Deep Feedback Network for Recommendation*
9. https://ata.atatech.org/articles/11000203575?layout=%2Fvelocity%2Flayout%2Fblank.vm
10. Kalman Filtering Attention for User Behavior Modeling in CTR Prediction

1. *End-to-End User Behavior Retrieval in Click-Through Rate* *Prediction Model*
2. https://ata.atatech.org/articles/11000229954?spm=ata.23639746.0.0.2b917411wLmznh&layout=%2Fvelocity%2Flayout%2Fblank.vm
3. *Sampling Is All You Need on Modeling Long-Term User* *Behaviors* *for CTR Prediction*
4. Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting
5. Query-dominant User Interest Network for Large-Scale Search Ranking
6. TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest
7. https://alidocs.dingtalk.com/i/nodes/7QG4Yx2JpYkn7M91s2eKAK6g89dEq3XD?cid=473786800:2815591145&utm_source=im&utm_scene=team_space&iframeQuery=utm_medium%3Dim_card%26utm_source%3Dim&utm_medium=im_card&dontjump=true&corpId=dingf898ede2a6eeab33bc961a6cb783455b





## Review Table 

| Name       | My Thinking                                                  | Ref                                              | Time | id   |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------ | ---- | ---- |
| YoutubeDNN | Maybe it's a first paper with ML for a  recommendation system. Two step pipeline: Matching (call back Candidates) and Ranking (rank the candidates).  Using ***Pooling and embedding*** | Deep Neural Networks for YouTube Recommendations | 2016 | 1    |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |
|            |                                                              |                                                  |      |      |



