# 接口说明

以下对原型系统中的模型交互相关部分的接口进行描述。接口函数存在于`main.py`中。

接口大致可以分为4类：

1. 基本信息：获得模型和数据的基本信息用于展示。
2. 模型交互：模型访问、搜索不公平样本、模型提升等与分类模型交互的相关功能。
3. 指标：包括准确性、个体/群体公平性指标，以及判断是否一对样本是公平的。
4. 参数设定：设置审计流程中的一些参数。

接口的使用方法可以参考`test_framework.ipynb`。

## 基本信息
### 获取模型：`get_model()`

获取基础的模型。

**Return type:**\
models.model.MLP。

### 模型基本信息：`get_model_info()`

获取模型的名称、层数、训练方法等基本信息。

**Return type:**\
dict。

### 数据基本信息：`get_data_info()`

获取数据集名字、任务，数据集包含字段，训练、测试集大小等基本信息。

**Return type:**\
dict。

### 获取默认的数据范围：`get_default_data_range()`

**Returns:**\
`range_dict`(dict) - 每个特征的数据范围。数值类型的特征范围为`[lower_bound, upper_bound]`闭区间，其中`lower_bound`和`upper_bound`均为float类型。类别类型的特征范围为`[value1, value2, ...]`等构成的。

## 参数设定

### 指定敏感属性： `set_sensitive_attr(sensitive_attr)`

指定敏感属性为性别或种族。

**Parameters:**\
`sensitive_attr`(str) - 敏感属性的名称（`'sex'`或`'race'`）。

### 设定数据范围：`set_data_range(range_dict)`

设定搜索不公平样本的数据范围。

**Parameters:**\
`range_dict`(dict) - 每个特征的数据范围。数值类型的特征范围为`[lower_bound, upper_bound]`闭区间，其中`lower_bound`和`upper_bound`均为float类型。类别类型的特征范围为`[value1, value2, ...]`等构成的。

### 获取用户设定的数据范围：`get_data_range()`

获取用户设置的搜索不公平样本的数据范围。

**Returns:**\
`range_dict`(dict) - 每个特征的数据范围。数值类型的特征范围为`[lower_bound, upper_bound]`闭区间，其中`lower_bound`和`upper_bound`均为float类型。类别类型的特征范围为`[value1, value2, ...]`等构成的。

### 指定个体公平指标：`set_individual_fairness_metric(dx, eps)`

设定个体公平性指标的输入空间相似度度量dx和epsilon值。

**Parameters:**\
`dx`(str) -  若为`'LR'`则指定dx为LR距离度量，若为`'Eu'`则指定dx为欧式距离度量。
`eps`(float) - 指定epsilon的值。

### 获得默认的epsilon值：`get_default_eps()`

**Returns：**\
`eps`(float) - 默认的epsilon取值。

## 模型交互
### 访问模型：`query_model(data_samlpe)`

查询模型对一个样本的分类结果，返回模型将该样本分类为`1`的概率，若概率大于0.5，则说明该样本被分类为`1`，否则分类为`0`。

**Parameters:**\
`data_sample`(*dict*) - 需要查询模型分类结果的样本特征字典。

**Returns**\
`p`(float) - 模型将该`data_sample`分类为`1`的概率。

### 搜索不公平样本：`seek_unfair_pair(init_sample)`

自动化构造模型会分类不公平的成对样本。采用BUFF*的算法，需要指定一个样本作为搜索算法的初始化。

**Parameters:**\
`init_sample`(*dict*) - 搜索不公平样本时的起始样本。

**Returns:**\
`unfair_pair`(list[dict]) - 成对的不公平样本。列表包含两个样本，每个样本都是字典形式。

### 获取搜索到的不公平样本：`get_unfair_pair()`

获取搜索到的不公平样本。

**Returns:**\
`unfair_pair`(list[dict]) - 成对的不公平样本。列表包含两个样本，每个样本都是字典形式。

### 模型优化：`optimize(model, unfair_pair)`

根据提供的成对不公平样本，通过微调的方式优化模型的公平性表现。

**Parameters:**\
`model`(models.model.MLP) - 待优化的模型。\
`unfair_pair`(list[dict]) - 提供给模型的成对不公平样本。

**Returns:**\
`optim_model`(models.model.MLP) - 优化完成的模型。

## 指标
### 准确性指标：`accuracy(model, whether_training_set=False)`

获得模型在训练集或测试集上的准确性。

**Parameters:**\
`model`(models.model.MLP) - 待检验的模型。\
`whether_training_set`(bool) - 若为`True`则在训练集上检验模型，反之在测试集上检验。默认为`False`。

**Return type:**\
float。

### 群体公平性指标：`group_fairness_metric(model, whether_training_set=False)`

获得模型在训练集或测试集上的群体公平性。数值越小代表公平性表现越好。

**Parameters:**\
`model`(models.model.MLP) - 待检验的模型。\
`whether_training_set`(bool) - 若为`True`则在训练集上检验模型，反之在测试集上检验。默认为`False`。

**Return type:**\
float。

### 个体公平性指标（局部表现）：`local_individual_fairness_metric(model, local_sample)`

获得模型在某个数据样本附近局部的个体公平性。数值越小代表公平性表现越好。

**Parameters:**\
`model`(models.model.MLP) - 待检验的模型。\
`local_sample`(dict) - 指定一个样本用于确定一个数据局部范围。

**Return type:**\
float。

### 个体公平性指标（全局表现）：`global_individual_fairness_metric(model)`

获得模型在全局的个体公平性表现。数值越小代表公平性表现越好。

**Parameters:**\
`model`(models.model.MLP) - 待检验的模型。\
`local_sample`(dict) - 指定一个样本用于确定一个数据局部范围。

**Return type:**\
float。

### 判断给定的一对样本是否公平：`fair(model, sample_pair)`

判断模型对给定的成对样本的分类结果是不是公平的。

**Parameters**\
`model`(models.model.MLP) - 待检验的模型。\
`sample_pair`(list[pair]) - 待检验的成对样本。

**Returns**:\
`whether_fair`(bool): 是否公平。\
`dx`：样本在输入空间中的相似性度量。\
`dy`：样本在输出空间中的相似性度量。