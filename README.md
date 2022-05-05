# tensorflow2pytorch
本项目可以将 TensorFlow 可用的预训练模型转换为 PyTorch 可用的预训练模型，例如：[Adversarially trained ImageNet models](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)。

本项目在`TensorFlow==1.14.0`和`PyTorch==1.7.1`下测试通过。

## 🎈Preparation

创建虚拟环境：

```python
conda create --name tensorflow2pytorch python=3.7 anaconda tensorflow-gpu==1.14.0
```

安装 PyTorch，请参考 [PyTorch 官网](https://pytorch.org/)：

```python
conda activate tensorflow2pytorch
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

下载项目：

```python
git clone https://github.com/ikaleo/tensorflow2pytorch.git
```

国内用户可通过 [GitHub Proxy](https://ghproxy.com/) 加速或使用：

```python
git clone https://gitee.com/ikaleo/tensorflow2pytorch.git
```



## 🍰Start conversion

基于 TensorFlow 的预训练模型一般是一些类似于`ens3_adv_inception_v3_rename.ckpt.data-00000-of-00001`、`ens3_adv_inception_v3_rename.ckpt.index`的文件，对于基于 ImageNet 的对抗训练模型，可以通过[Adversarially trained ImageNet models](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)下载。

准备好预训练模型后，依次执行如下命令安装所需要的包：

```python
cd ./tensorflow2pytorch
pip install -r requirements.txt
```

之后便可以开始转换：

```python
python tensorflow2pytorch.py --model inceptionv3 --input models/ckpt/ens4_adv_inception_v3_rename.ckpt.index --output models/pth/ens4_adv_inception_v3_rename.pth
```

由于引入了`pretrainedmodels`包来导入模型，因此可以查看 [Available models](https://github.com/Cadene/pretrained-models.pytorch#available-models) 以获得支持导入的模型列表，即参数`--model`的支持范围。



## 🎉Surprise

为了方便使用，本项目提供了一些已经转换完成的预训练模型，如下所示：

+ `adv-Inception-v3`
+ `adv-InceptionResnet-v2`
+ `ens3-adv-Inception-v3`
+ `ens4-adv-Inception-v3`
+ `ens-adv-InceptionResNet-v2`

[点击此处](https://pan.angustar.com/zh-CN/%F0%9F%8E%89%20Public/Models/Adversarially%20trained%20ImageNet%20models/)可以下载这些预训练模型。



## Acknowledgement

[tensorflow-to-pytorch](https://github.com/haolingguang/tensorflow-to-pytorch)

[Adversarially trained ImageNet models](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)

[Pretrained models for Pytorch](https://github.com/Cadene/pretrained-models.pytorch)



## License

This repo is distributed under the GNU GPL version 3. License is available [here](https://github.com/ikaleo/tensorflow2pytorch/blob/main/LICENSE).

