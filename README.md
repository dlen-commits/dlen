1. 环境准备

本项目基于xdl, 先拉取xdl docker镜像

```
docker run --net=host -v /path/to/workspace/:/home/xdl -dit  registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-gpu-mxnet1.3  /bin/bash
```
2. 数据准备
* 公开数据集的下载、使用和授权
Ali-CCP：Alibaba Click and Conversion Prediction请参阅：[https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408](https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408)

* 在镜像内参考[https://github.com/alibaba/x-deeplearning/tree/master/xdl-algorithm-solution/ESMM/data](https://github.com/alibaba/x-deeplearning/tree/master/xdl-algorithm-solution/ESMM/data)预处理数据

3. 训练模型
```
python mmoe_dlen.py --run_mode=local --ckpt_dir=./ckpt --config=train_config.json
```

4. 测试模型

```
python mmoe_dlen.py --run_mode=local --ckpt_dir=./ckpt --config=test_config.json
```