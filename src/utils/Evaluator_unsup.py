import numpy as np
import torch
from abc import ABC, abstractmethod
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }

def from_predefined_split(data):
    assert all([mask is not None for mask in [data.train_mask, data.test_mask, data.val_mask]])
    num_samples = data.num_nodes
    indices = torch.arange(num_samples)
    return {
        'train': indices[data.train_mask],
        'valid': indices[data.val_mask],
        'test': indices[data.test_mask]
    }


def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps

# class BaseEvaluator(ABC):
#     @abstractmethod
#     def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
#         pass
#
#     def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
#         for key in ['train', 'test', 'valid']:
#             assert key in split
#
#         result = self.evaluate(x, y, split)
#         return result


class BaseSKLearnEvaluator():
    def __init__(self, evaluator, params):
        self.evaluator = evaluator
        self.params = params

    def evaluate(self, x, y, split, cv=5, scoring='accuracy'):

        # x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        # ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)

        train_index, test_index = split['train'], split['test']
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier = GridSearchCV(self.evaluator, self.params, cv=cv, scoring=scoring, verbose=0)
        classifier.fit(x_train, y_train)
        test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
        test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')
        test_acc = accuracy_score(y_test, classifier.predict(x_test))
        
        if scoring == 'accuracy':
            return {
                'micro_f1': test_micro,
                'macro_f1': test_macro,
                'accuracy': test_acc,
            }
        else:
            test_rauc = roc_auc_score(y_test, classifier.predict(x_test))
            return {
                'micro_f1': test_micro,
                'macro_f1': test_macro,
                'accuracy': test_acc,
                'roc_auc': test_rauc,
            }

class SVMEvaluator(BaseSKLearnEvaluator):
    def __init__(self, linear=True, params=None):
        if linear:
            self.evaluator = LinearSVC()
        else:
            self.evaluator = SVC()
        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        super(SVMEvaluator, self).__init__(self.evaluator, params)

class LREvaluator(BaseSKLearnEvaluator):
    def __init__(self, params=None):
        self.evaluator = LogisticRegression()
        if params is None:
            params = {
                'penalty': ['l1', 'l2', 'elasticnet'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            }
        super(LREvaluator, self).__init__(self.evaluator, params)


class OGBEvaluator():

    def __init__(self, evaluator, base_classifier='lr', param_search=True, param_dict=None) -> None:
        self.base_classifier = base_classifier
        self.evaluator = evaluator
        self.param_search = param_search
        self.param_dict = param_dict
        self.eval_metric = evaluator.eval_metric

        if self.eval_metric == 'rmse':
            self.gscv_scoring_name = 'neg_root_mean_squared_error'
        elif self.eval_metric == 'mae':
            self.gscv_scoring_name = 'neg_mean_absolute_error'
        elif self.eval_metric == 'rocauc':
            self.gscv_scoring_name = 'roc_auc'
        elif self.eval_metric == 'accuracy':
            self.gscv_scoring_name = 'accuracy'
        else:
            raise ValueError('Undefined grid search scoring for metric %s ' % self.eval_metric)

        self.classifier = None

    def evaluate(self, train_x, train_y, val_x, val_y, test_x, test_y):
        if self.param_search:
            all_res = []
            best_train, best_val, best_test = 0, 0, 0
            for i in range(6):
                if self.base_classifier == 'lr' or self.base_classifier == 'svm':
                    C = np.random.uniform(0.005, 0.2, 1)[0]
                    classifier = self.get_classifier(C=C)
                else:
                    raise ValueError("Base classifier not implemented!")
                train_raw, val_raw, test_raw = self.binary_classification(
                    classifier, train_x, train_y, val_x, val_y, test_x, test_y)
                train_score = self.scorer(train_y, train_raw)
                val_score = self.scorer(val_y, val_raw)
                test_score = self.scorer(test_y, test_raw)
                if val_score > best_val:
                    best_train, best_val, best_test = train_score, val_score, test_score
        else:
            classifier = self.get_classifier()
        
            train_raw, val_raw, test_raw = self.binary_classification(
                classifier, train_x, train_y, val_x, val_y, test_x, test_y)
            best_train = self.scorer(train_y, train_raw)
            best_val = self.scorer(val_y, val_raw)
            best_test = self.scorer(test_y, test_raw)

        return best_train, best_val, best_test

    def binary_classification(self, classifier, train_x, train_y, val_x, val_y, test_x, test_y):
        classifier.fit(train_x, train_y)
        if self.eval_metric == 'accuracy':
            train_raw = classifier.predict(train_x)
            val_raw = classifier.predict(val_x)
            test_raw = classifier.predict(test_x)
        else:
            train_raw = classifier.predict_proba(train_x)[:, 1]
            val_raw = classifier.predict_proba(val_x)[:, 1]
            test_raw = classifier.predict_proba(test_x)[:, 1]

        return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)
    
    def scorer(self, y_true, y_raw):
        input_dict = {"y_true": y_true, "y_pred": y_raw}
        score = self.evaluator.eval(input_dict)[self.eval_metric]
        return score

    def get_classifier(self, C=0.1):
        if self.base_classifier == 'lr':
            base_classifier = LogisticRegression(dual=False, fit_intercept=True, max_iter=5000, C=C)
            classifier = make_pipeline(
                StandardScaler(),
                base_classifier,)
        elif self.base_classifier == 'svm':
            base_classifier = SVC(kernel='linear',probability=True)
            classifier = make_pipeline(
                StandardScaler(),
                base_classifier,)
        else:
            raise ValueError('Base classifier not implemented!')
        return classifier