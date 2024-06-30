import pandas as pd
import torch

from models.model import MLP
from models.trainer import STDTrainer
from models.metrics import accuracy, equally_opportunity, individual_unfairness
from data import adult_short
from utils import get_data, UnfairMetric, decide_label_by_majority_voting, add_data_to_dataset
from seeker.gradiant_based import BlackboxSeeker
from distances.sensitive_subspace_distances import LogisticRegSensitiveSubspace
from distances.normalized_mahalanobis_distances import ProtectedSEDistances
from distances.binary_distances import BinaryDistance

class AuditingFramework:
    def __init__(self) -> None:
        # data: data class, dataset, data loader, data generator, etc.
        self.data = adult_short
        # initialize by calling set_sensitive_attr
        self.dataset, self.train_dl, self.test_dl, self.data_gen = None, None, None, None
        self.sensitive_attr = None

        # model
        self.feature_dim = 21 # self.dataset.dim_feature()
        self.output_dim = 2
        self.model = MLP(input_size=self.feature_dim, output_size=self.output_dim, n_layers=4)
        checkpoint = 'trained_models/MLP_all-features_0.pth'
        self.model.load(checkpoint)

        # data range
        self.range_dict = None

        # fairness metric
        # initialize by calling set_individual_fairness_metric
        self.unfair_metric = None

        # seeker to automatically find the unfair sample pairs
        # initialize by calling set_individual_fairness_metric
        self.seeker = None

    def _tensor2dict(self, data_tensor):
        data_df = self.data_gen.feature_dataframe(data=data_tensor)
        data_df['marital-status'] = data_df['marital-status'].replace({
            0: 'Divorced',
            1: 'Married-AF-spouse',
            2: 'Married-civ-spouse',
            3: 'Married-spouse-absent',
            4: 'Never-married',
            5: 'Separated',
            6: 'Widowed'
        })
        data_df['race_White'] = data_df['race_White'].apply(lambda x: 'Others' if x < 0.5 else 'White')
        data_df['sex_Male'] = data_df['sex_Male'].apply(lambda x: 'Female' if x < 0.5 else 'Male')
        data_df['workclass'] = data_df['workclass'].replace({
            0: 'Federal-gov',
            1: 'Local-gov',
            2: 'Private',
            3: 'Self-emp-inc',
            4: 'Self-emp-not-inc',
            5: 'State-gov',
            6: 'Without-pay'
        })
        return data_df.to_dict('records')

    def _dict2tensor(self, data_dict):
        if isinstance(data_dict, dict):
            data_dict = [data_dict]

        data_df = pd.DataFrame(data_dict)[self.data_gen.feature_name]
        data_df['marital-status'] = data_df['marital-status'].replace({
            'Divorced': 0,
            'Married-AF-spouse': 1,
            'Married-civ-spouse': 2,
            'Married-spouse-absent': 3,
            'Never-married': 4,
            'Separated': 5,
            'Widowed': 6
        })
        data_df['race_White'] = data_df['race_White'].replace({'Others': 0, 'White': 1})
        data_df['sex_Male'] = data_df['sex_Male'].replace({'Female': 0, 'Male': 1})
        data_df['workclass'] = data_df['workclass'].replace({
            'Federal-gov': 0,
            'Local-gov': 1,
            'Private': 2,
            'Self-emp-inc': 3,
            'Self-emp-not-inc': 4,
            'State-gov': 5,
            'Without-pay': 6
        })
        # print(data_df)
        data_feature = torch.Tensor(data_df.values)
        # print(data_feature)
        return self.data_gen._feature2data(data_feature)

    def _dataloader2s(self, dataloader):
        s_id = self.dataset.sensitive_idxs[0]
        for x, y in dataloader:
            g = x[:, s_id]
            yield x, y, g

    def _set_seeker_data_range(self):
        assert self.seeker != None

        if len(self.range_dict['sex_Male']) == 1:
            self.range_dict['sex_Male'].append(self.range_dict['sex_Male'][0])
        if len(self.range_dict['race_White']) == 1:
            self.range_dict['race_White'].append(self.range_dict['race_White'][0])

        continuous_columns = ['age', 'capital-gain', 'capital-loss', 'education-num',
                              'hours-per-week', 'race_White', 'sex_Male']
        onehot_columns = ['marital-status', 'workclass']
        continuous_range = {key: self.range_dict[key] for key in continuous_columns}
        onehot_value_dict = {key: self.range_dict[key] for key in onehot_columns}
        lower, upper, mask = self.data_gen.gen_range(continuous_range, onehot_value_dict)
        # print(upper, '\n', lower, '\n', mask)
        self.seeker.set_data_range(lower, upper, mask)

    # ----------------------------- 基本信息 -----------------------------------

    def get_model(self):
        return self.model
    
    def get_model_info(self):
        return {
            'name': 'MLP',
            'n_layers': 4,
            'hidden dimension': 64,
            'optimizer': 'Adam',
            'learning rate': 1e-3,
        }
    
    def get_data_info(self):
        return {
            'name': 'adult',
            'task': 'Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.',
            'columns': ['age', 'capital-gain', 'capital-loss', 'education-num', 'hours-per-week', 'race_White',
                        'sex_Male', 'marital-status', 'workclass'],
            'training set size': 36177,
            'testing set size': 9045
        }

    def get_default_data_range(self):
        return {
            'age': [17, 90], # 年龄 - 1~90岁
            'capital-gain': [0, 99999], # 资产收益 - $0 ~ $999999
            'capital-loss': [0, 4356], # 资产损失 - $0 ~ $4356
            'education-num': [1, 16], # 受教育时长 - 1~16年
            'hours-per-week': [1, 99], # 每周工作时长 - 1~99小时
            'race_White': [0, 1], # 是否为白人 - 是/否
            'sex_Male': [0, 1], # 是否为男性 - 是/否
            # 婚姻状况 - [离异，军人家属婚姻，普通婚姻，已婚但夫妻异地，未婚，分居，丧偶]
            'marital-status': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent',
                                'Never-married', 'Separated', 'Widowed'],
            # 工作类型 - [联邦政府雇员，地方政府雇员，私营企业，自雇（收入较高），自雇（收入一般），州政府雇员，无报酬]
            'workclass': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']
        }

    # ----------------------------- 参数设定 -----------------------------------

    def set_sensitive_attr(self, sensitive_attr):
        assert sensitive_attr in ['sex', 'race']

        self.sensitive_attr = ['sex_Male'] if sensitive_attr == 'sex' else ['race_White']

        self.dataset, self.train_dl, self.test_dl = get_data(self.data, 0, self.sensitive_attr)
        self.dataset.use_sensitive_attr = True
        self.data_gen = self.data.Generator(include_sensitive_feature=True, sensitive_vars=self.sensitive_attr, device='cpu')

    def set_data_range(self, range_dict):
        self.range_dict = range_dict

    def set_individual_fairness_metric(self, dx, eps):
        assert dx in ['LR', 'Eu']
        assert self.dataset != None
        assert self.range_dict != None

        if dx == 'LR':
            distance_x = LogisticRegSensitiveSubspace()
            distance_x.fit(self.dataset.get_all_data(), data_gen=self.data_gen, sensitive_idxs=self.dataset.sensitive_idxs)
        else:
            distance_x = ProtectedSEDistances()
            distance_x.fit(num_dims=self.dataset.dim_feature(), data_gen=self.data_gen, sensitive_idx=self.dataset.sensitive_idxs)
        distance_y = BinaryDistance()
        
        self.unfair_metric = UnfairMetric(dx=distance_x, dy=distance_y, epsilon=eps)

        self.seeker = BlackboxSeeker(model=self.model, unfair_metric=self.unfair_metric, data_gen=self.data_gen)
        self._set_seeker_data_range()

    def get_default_eps(self):
        return 1e8

    # def get_samples_to_estimate_eps(self):
    #     pass

    # def estimate_eps(self, similar_pairs):
    #     pass

    # ----------------------------- 模型交互 -----------------------------------
        
    def query_model(self, data_sample):
        data_tensor = self._dict2tensor(data_sample)
        logits = self.model(data_tensor)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities[0][1].item()

    def seek_unfair_pair(self, init_sample):
        init_sample = self._dict2tensor(init_sample)
        pair, n_query  = self.seeker.seek(lamb=1, origin_lr=0.1, max_query=5e4, init_sample=init_sample)
        # print(n_query)
        return self._tensor2dict(pair)

    def optimize(self, model, unfair_pair):
        unfair_pair = self._dict2tensor(unfair_pair)

        num_retrain_samples = 100

        optimized_model = MLP(input_size=self.feature_dim, output_size=self.output_dim, n_layers=4)
        optimized_model.load_state_dict(model.state_dict())

        addition_data1 = self.data_gen.generate_around_datapoint(unfair_pair[0], n=int((num_retrain_samples-2)/2))
        addition_data2 = self.data_gen.generate_around_datapoint(unfair_pair[1], n=int((num_retrain_samples-2)/2))
        addition_data = torch.concat([unfair_pair, addition_data1, addition_data2], dim=0)
        addition_data_label = (torch.ones(addition_data.shape[0]) * decide_label_by_majority_voting(unfair_pair, model, self.data_gen, 1000)).detach()
        gen_train_dl = add_data_to_dataset(self.train_dl, addition_data, addition_data_label, 'cpu')

        trainer = STDTrainer(optimized_model, gen_train_dl, self.test_dl, device='cpu', epochs=1000, lr=1e-3)
        trainer.train()

        return optimized_model

    # ------------------------------- 指标 -------------------------------------

    def accuracy(self, model, whether_training_set=False):
        dataloader = self.train_dl if whether_training_set else self.test_dl
        return accuracy(model, dataloader)

    def group_fairness_metric(self, model, whether_training_set=False):
        dataloader = self.train_dl if whether_training_set else self.test_dl
        return equally_opportunity(model, self._dataloader2s(dataloader))

    def local_individual_fairmess_metric(self, model, local_sample):
        local_sample = self._dict2tensor(local_sample)
        data_around = self.data_gen.generate_around_datapoint(local_sample, n=1000)
        return individual_unfairness(model, self.unfair_metric, data_around)

    def global_individual_fairness_metric(self, model):
        # 这个指标计算太慢
        n = 10000
        data_generated = self.data_gen.gen_by_range(n=n)
        mean_L_list = []
        for i in range(n):
            datapoint = data_generated[i]
            data_around = self.data_gen.generate_around_datapoint(datapoint, n=1000)
            mean_L = individual_unfairness(model, self.unfair_metric, data_around)
            mean_L_list.append(mean_L)
        return sum(mean_L_list)/len(mean_L_list)

    def fair(self, model, sample_pair):
        pair_tensor = self._dict2tensor(sample_pair)
        y1 = model.get_prediction(pair_tensor[0])
        y2 = model.get_prediction(pair_tensor[1])
        fair_or_not = not self.unfair_metric.is_unfair(pair_tensor[0], pair_tensor[1], y1, y2)
        dx = self.unfair_metric.dx(pair_tensor[0], pair_tensor[1]).item()
        dy = self.unfair_metric.dy(y1, y2).item()
        return fair_or_not, dx, dy