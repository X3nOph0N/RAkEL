from os import system
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC as SVMClassifier
from sklearn.naive_bayes import GaussianNB as NBClassifier
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from numpy import ndarray, array, zeros,sign, cast, int32, bool_
from numpy import sum as array_sum
from time import ctime, time
from random import sample
from tqdm import tqdm
from itertools import combinations
# from math import factorial


class LP():
    """
    This class imples the LP algorithm
    @params:
    classifier: the classifier used in LP
    """

    def __init__(self, classifier: str, **kwargs) -> None:
        self._classifier = globals()[classifier+'Classifier'](**kwargs)
        self._label_count = None
        self._powerset = None

    def fit(self, x: ndarray, y: ndarray) -> None:
        """
        This function will first build the powerset for current label and start corresponding classifier
        fiting routing.
        @params:
        train_x:train labels, in t*n array form
        train_y:train targets, in t*i array form
        @outputs:
        None
        """
        self.build_dict(y)
        self._classifier.fit(x, self.transform(y))
        return

    def build_dict(self, y: ndarray) -> None:
        """
        This function will build the tranform diction from the y label data.
        @params:
        y:in array form.
        @outputs:
        None
        """
        
        self._powerset = []
        for _ in y:
            find = False
            for __ in self._powerset:
                if (_ == __).all():
                    find = True
                    break
            if not find:
                self._powerset.append(_)
                
        return

    def predict(self, x: ndarray) -> ndarray:
        """
        This function will give prediction to given x.
        WARNING: this function's behavior might change if the LP classifier's implememt is changed.
        @params:
        x: data label in array form.
        @outputs:
        y: predicted label, 1 for positive,-1 for negative and 0 for masked.
        """
        return self.inverse_transform(self._classifier.predict(x.reshape(1,-1)))

    def transform(self, y: ndarray) -> ndarray:
        """
        This function will transform the origin y to a unique label so the origin problem is 
        changed into a multiclass classification problem.
        WARNING: error if called before fit() is called.
        @params:
        y: target labels, in t*i form or 1*i form.
        @outputs:
        a integer stands for the corresponding class that those label belong to.
        """
        # if len(y.shape) > 1:
        #     return array([self._transform_dict[y_i] for y_i in y])
        # else:
        #     return self._transform_dict[y]
        result = []
        for _ in y:
            for idx in range(len(self._powerset)):
                if (_ == self._powerset[idx]).all():
                    result.append(idx)
                    break
        return array(result)

    def inverse_transform(self, y: ndarray) -> ndarray:
        """
        This function will transform the predicted class to origin labels.
        WARNING: error if called before fit() is called.
        @params:
        y: a INTEGER stands for the class of a set of labels.
        @outputs:
        a array that stands the origin label situation,1 for positive, 
        -1 for negative and 0 for masked
        """
        return array([self._powerset[int(_)] for _ in y])


class RAKEL():
    """
    This is the implement for RAKEL algorithm
    @params:
    classifier:should be a model with fit() and prediction() function.
    k: the size of subset
    m: the number of subset
    """

    def __init__(self, classifier: str, *args, **kwargs) -> None:
        self._LP_type = classifier
        self._k = None
        self._m = None
        return

    def fit(self, train_x, train_y, k, m, *args, **kwargs):
        """
        This function starts the fitting routing of the algorithm.
        @params:
        train_x:train labels, in t*n array form
        train_y:train targets, in t*i array form
        k: the size of subset
        m: the number of total subsets
        @output:
        None
        """
        # assert(k<train_y.shape[1])
        # assert(m < factorial(train_y.shape[1])/factorial(m)/factorial(train_y.shape[1]-k))
        self._LP_classifiers = [LP(self._LP_type, **kwargs) for _ in range(m)]
        self._k = k
        self._m = m
        k_set = [_ for _ in combinations(range(train_y.shape[1]),self._k)]
        k_set = sample(k_set,self._m)
        # In our realization, we use 1 for positive, -1 for negative and 0 for not in current subset.
        self._masks = []
        for k_l in k_set:
            mask = zeros(train_y.shape[1])
            mask[[idx for idx in k_l]] = [1 for idx in k_l]
            self._masks.append(mask)
        for i in range(m):
            self._LP_classifiers[i].fit(train_x, train_y*self._masks[i])
        return

    def predict(self, x, *args, **kwargs)->ndarray:
        """
        This function will give the prediction to the given labels.
        @params:
        x: a single data
        @outputs:
        result: predicted labels, 1 for positive,-1 for negative and 0 for masked.
        """
        result = array_sum(array([classifier.predict(x)
                     for classifier in self._LP_classifiers]).squeeze(), axis=0)
        result = sign(result*2+1)
        return result

    def eval(self, val_X, val_Y, print_result: bool = True, save_result: bool = False, *args, **kwargs):
        """
        This function will give the prediction to the given labels.
        @params:
        val_x: validation labels
        val_y: validation targets
        print_result: print the result if true
        save_result: save the result to the given file if true
        @output:
        None
        """
        start_time = time()
        metrics = {}
        pred_Y = []
        for x in tqdm(val_X, desc='predicting...'):
            pred_Y.append(self.predict(x))
        pred_Y = array(pred_Y)
        print('evaluating...')
        metrics['macro F1 score'] = f1_score(val_Y, pred_Y, average='macro')
        metrics['micro F1 score'] = f1_score(val_Y, pred_Y, average='micro')
        metrics['hamming loss score'] = hamming_loss(val_Y, pred_Y)
        metrics['acc score'] = accuracy_score(val_Y, pred_Y)
        print('evaluation done.')
        end_time = time()
        if save_result:
            if 'savePath' in kwargs:
                result_file = open(kwargs['savePath'], 'wt')
            else:
                result_file = open(ctime()+' result.txt', 'wt')
            result_file.write('experiment time:'+ctime()+'sec'+'\n')
            result_file.write("run time:"+str(end_time-start_time)+'\n')
            for metric in metrics:
                result_file.write(metric +
                                  ':'+str(metrics[metric])+'\n')
            result_file.close()
        if print_result:
            system('cls')
            print('experiment time:'+ctime())
            print("run time:"+str(end_time-start_time)+'sec')
            for metric in metrics:
                print('metric '+str(metric)+':'+str(metrics[metric]))
        return

    @property
    def parameter(self):
        return {
            'type': self._LP_type,
            'k': self._k,
            'm': self._m
        }
