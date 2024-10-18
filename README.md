# Lora
这是lora训练模型的参考代码

# 1. Lora + EAE
## 1.1 EAE Base
EAE基于传统深度模型训练信息，3次：
- 1次
start trainning....
0:20:26.492532
0.5300 0.3287 0.4058
0 0.40578887627695803
0:20:12.634644
0.4798 0.4156 **0.4454**
1 0.4454299088445429
0:20:13.617798
0.4598 0.4313 0.4451
2 0.44507710557532615
0:20:13.064541
0.4797 0.4028 0.4379
3 0.437890527368158
0:20:14.251547
0.4769 0.3982 0.4340
4 0.43397644700576293
0:20:12.369265
0.4677 0.3922 0.4266
5 0.4266066516629158
0:20:13.156849
0.4762 0.3821 0.4240
6 0.4239795918367347
0:20:14.201914
0.4306 0.3834 0.4056
7 0.4056420233463035
0:20:13.312576
0.4954 0.3706 0.4240
8 0.42398737506575485
0:20:13.825360
0.4558 0.3655 0.4057
9 0.405715743812197
- 2
start trainning....
0:20:29.832050
0.5354 0.3545 0.4266
0 0.42655601659751036
0:20:13.152587
0.5314 0.3733 **0.4386**
1 0.43856332703213613
0:20:13.760264
0.5148 0.3669 0.4285
2 0.4284563758389262
0:20:14.204501
0.4670 0.3770 0.4172
3 0.4171966420758077
0:20:14.008638
0.4741 0.4083 0.4387
4 0.4387351778656126
0:20:13.888507
0.4778 0.3664 0.4148
5 0.4147801196981524
0:20:14.164572
0.4563 0.3959 0.4239
6 0.4239290989660266
0:20:14.524841
0.4246 0.3793 0.4007
7 0.4006799417192812
0:20:13.334480
0.4839 0.3720 0.4206
8 0.4205874707564336
0:20:14.480783
0.4554 0.3802 0.4144
9 0.41443247306439485

- 3次
start trainning....
0:20:19.720377
0.5502 0.3549 0.4315
0 0.43152599217439913
0:20:12.268257
0.5177 0.3770 0.4363
1 0.4362862463421122
0:20:13.849715
0.5390 0.3779 **0.4443**
2 0.44432432432432434
0:20:12.846842
0.4874 0.4014 0.4402
3 0.44024205748865364
0:20:13.900681
0.4625 0.3968 0.4271
4 0.427121999505073
0:20:12.829938
0.4753 0.3756 0.4196
5 0.4196199280945044
0:20:13.648574
0.4789 0.3701 0.4175
6 0.4175311203319502

## 1.2 EAE model with lora
如何设置 Target modules for applying PEFT / LoRA on different models

https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models

figure中也有：modules_to_save

```
DataParallel(
  (module): Bert(
    (model): BertModel(
      (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(30522, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (token_type_embeddings): Embedding(2, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): BertEncoder(
        (layer): ModuleList(
          (0-11): 12 x BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (pooler): BertPooler(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (activation): Tanh()
      )
    )
    (dropout): Dropout(p=0.5, inplace=False)
    (linear_1): Linear(in_features=768, out_features=1, bias=True)
    (linear_2): Linear(in_features=768, out_features=1, bias=True)
  )
)
```

```
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
)


for _ in range(0,2):
    print('-----------------------------------------------------')
    loss_func = torch.nn.CrossEntropyLoss()
    model = Bert()
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    model = get_peft_model(model, config)
    model.print_trainable_parameters()



===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /root/paddlejob/workspace/env_run/log/test_result/py38/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda110.so
CUDA SETUP: CUDA runtime path found: /usr/local/cuda/lib64/libcudart.so
CUDA SETUP: Highest compute capability among GPUs detected: 8.0
CUDA SETUP: Detected CUDA version 110
CUDA SETUP: Loading binary /root/paddlejob/workspace/env_run/log/test_result/py38/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda110.so...
trainable params: 589,824 || all params: 110,073,602 || trainable%: 0.5358
```


问题：
===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
CUDA_SETUP: 。。。

https://powervision.feishu.cn/wiki/DUMbwl0MmiNeA6ksElqcWIzxnqg
https://stackoverflow.com/questions/75918140/getting-runtimeerror-expected-scalar-type-half-but-found-float-in-aws-p3-instan




##### 参考：

https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/simple_thu_chatglm6b



