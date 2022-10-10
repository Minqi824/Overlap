import os
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import gc
from keras import backend as K

from data_generator import DataGenerator
from myutils import Utils

# from baseline.SGAE.sgae_train import sgae
# from baseline.A3.run import A_3

class RunPipeline():
    def __init__(self, suffix:str=None, generate_duplicates=False, n_samples_threshold=1000,
                 realistic_synthetic_mode=None, parallel=None, architecture=None):
        '''
        generate_duplicates: whether to generate duplicated samples when sample size is too small
        n_samples_threshold: threshold for generating the above duplicates
        '''

        # my utils function
        self.utils = Utils()

        # global parameters
        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold
        self.realistic_synthetic_mode = realistic_synthetic_mode

        # the suffix of all saved files
        if parallel != 'proposed':
            self.suffix = suffix + '_baseline_' + parallel + '_' + 'duplicates(' + str(generate_duplicates) + ')_' + 'synthetic(' + str(realistic_synthetic_mode) + ')'
        else:
            assert architecture is not None
            self.suffix = suffix + '_' + parallel + '_' + architecture + '_' + 'duplicates(' + str(generate_duplicates) + ')_' + 'synthetic(' + str(realistic_synthetic_mode) + ')'

        # data generator instantiation
        self.data_generator = DataGenerator(generate_duplicates=self.generate_duplicates,
                                            n_samples_threshold=self.n_samples_threshold)

        # ratio of labeled anomalies
        self.rla_list = [0.05, 0.10, 0.20]
        # seed list
        self.seed_list = list(np.arange(5) + 1)

        self.parallel = parallel
        self.architecture = architecture

        # model dict
        self.model_dict = {}

        if self.parallel == 'unsup':
            from baseline.PyOD import PYOD

            self.model_dict['IForest'] = PYOD
            self.model_dict['ECOD'] = PYOD
            self.model_dict['DeepSVDD'] = PYOD # need tensorflow 2.0+

        elif self.parallel == 'semi':
            from baseline.GANomaly.run import GANomaly
            from baseline.DeepSAD.src.run import DeepSAD
            from baseline.REPEN.run import REPEN
            from baseline.DevNet.run import DevNet
            from baseline.PReNet.run import PReNet
            from baseline.FEAWAD.run import FEAWAD

            self.model_dict['GANomaly'] = GANomaly
            self.model_dict['DeepSAD'] = DeepSAD
            self.model_dict['REPEN'] = REPEN
            self.model_dict['DevNet'] = DevNet
            self.model_dict['PReNet'] = PReNet
            self.model_dict['FEAWAD'] = FEAWAD

        elif self.parallel == 'sup':
            from baseline.FS.run import fs
            from baseline.FTTransformer.run import FTTransformer

            self.model_dict['FS'] = fs
            self.model_dict['ResNet'] = FTTransformer
            self.model_dict['FTTransformer'] = FTTransformer

        elif self.parallel == 'DualGAN':
            from old.RCCDualGAN.run import RccDualGAN
            self.model_dict['RCCDualGAN'] = RccDualGAN

        elif self.parallel == 'proposed':
            from baseline.ADSD.run import adsd
            from old.ADSD_DeepSAD.src.run import ADSD_DeepSAD

            if architecture == 'DeepSAD':
                self.model_dict['ADSD'] = ADSD_DeepSAD
            else:
                self.model_dict['ADSD'] = adsd

        else:
            raise NotImplementedError

        # SGAE是无监督模型, A3目前效果不好需要进一步调优
        # self.model_dict['SGAE'] = sgae
        # self.model_dict['A3'] = A_3

    # dataset filter for delelting those datasets that do not satisfy the experimental requirement
    def dataset_filter(self):
        # dataset list in the current folder
        dataset_list_org = [os.path.splitext(_)[0] for _ in os.listdir(os.path.join(os.getcwd(), 'datasets', 'new'))
                            if os.path.splitext(_)[-1] != '.md']

        # 将不符合标准的数据集筛除
        dataset_list = []
        dataset_size = []

        for dataset in dataset_list_org:
            add = True
            for seed in self.seed_list:
                self.data_generator.seed = seed
                self.data_generator.dataset = dataset
                data = self.data_generator.generator(la=1.00)

                if not self.generate_duplicates and len(data['y_train']) + len(data['y_test']) < self.n_samples_threshold:
                    add = False
                else:
                    # rla模式中只要训练集labeled anomalies个数超过0即可
                    if sum(data['y_train']) > 0:
                        pass

                    else:
                        add = False

            if add:
                dataset_list.append(dataset)
                dataset_size.append(len(data['y_train']) + len(data['y_test']))
            else:
                print(f"数据集{dataset}被移除")

        # 按照数据集大小进行排序
        dataset_list = [dataset_list[_] for _ in np.argsort(np.array(dataset_size))]

        return dataset_list

    # model fitting function
    def model_fit(self):
        try:
            # model initialization, if model weights are saved, the save_suffix should be specified
            if self.model_name in ['DevNet', 'FEAWAD', 'REPEN']:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, save_suffix=self.suffix)
            elif self.model_name == 'ADSD' and self.architecture != 'DeepSAD':
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, architecture=self.architecture)
            else:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name)

        except Exception as error:
            print(f'Error in model initialization. Model:{self.model_name}, Error: {error}')
            pass

        try:
            # model fitting, currently most of models are implemented to output the anomaly score
            if self.model_name not in ['ADSD', 'FS', 'SGAE', 'A3'] or self.architecture == 'DeepSAD':
                # fitting
                self.clf = self.clf.fit(X_train=self.data['X_train'], y_train=self.data['y_train'],
                                        ratio=sum(self.data['y_test']) / len(self.data['y_test']))
                # predicting score
                score_test = self.clf.predict_score(self.data['X_test'])
                # performance
                result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)
            else:
                result = self.clf.fit2test(self.data)

            K.clear_session()  # 实际发现代码会越跑越慢,原因是keras中计算图会叠加,需要定期清除
            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            result = {'aucroc': np.nan, 'aucpr': np.nan}
            pass

        return result

    # run the experiment
    def run(self):
        #  filteting dataset that do not meet the experimental requirements
        dataset_list = self.dataset_filter()

        # experimental parameters
        experiment_params = list(product(dataset_list, self.rla_list, self.seed_list))


        print(f'共有{len(dataset_list)}个数据集, {len(self.model_dict.keys())}个模型')

        # 记录结果
        df_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_time = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))

        for i, params in tqdm(enumerate(experiment_params)):
            dataset, la, self.seed = params

            print(f'Current experiment parameters: {params}')

            for model_name in tqdm(self.model_dict.keys()):
                self.model_name = model_name
                self.clf = self.model_dict[self.model_name]

                # generate data
                self.data_generator.seed = self.seed
                self.data_generator.dataset = dataset
                self.data = self.data_generator.generator(la=la, realistic_synthetic_mode=self.realistic_synthetic_mode)

                # fit model
                start_time = time.time() # starting time
                result = self.model_fit()
                end_time = time.time() # ending time

                # store and save the result
                df_AUCROC[model_name].iloc[i] = result['aucroc']
                df_AUCPR[model_name].iloc[i] = result['aucpr']
                df_time[model_name].iloc[i] = round(end_time - start_time, 2)

                df_AUCROC.to_csv(os.path.join(os.getcwd(), 'result', 'AUCROC_' + self.suffix + '.csv'), index=True)
                df_AUCPR.to_csv(os.path.join(os.getcwd(), 'result', 'AUCPR_' + self.suffix + '.csv'), index=True)
                df_time.to_csv(os.path.join(os.getcwd(), 'result', 'Time_' + self.suffix + '.csv'), index=True)

# run the experment
pipeline = RunPipeline(suffix='ADSD', parallel='proposed', architecture='FTTransformer',
                       generate_duplicates=True, n_samples_threshold=1000, realistic_synthetic_mode=None)
pipeline.run()