from os import system
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC as SVMClassifier
from sklearn.naive_bayes import GaussianNB as NBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from numpy import ndarray, array,zeros,sum,sign,cast,int32,bool_
from time import ctime, time
from random import sample
from math import factorial
from tqdm import tqdm

class LP():
    """
    This class imples the LP algorithm
    @params:
    classifier: the classifier used in LP
    """
    def __init__(self, classifier: str, **kwargs)->None:
        self._classifier = getattr(globals(), classifier+'Classifier')(kwargs)
        self._transform_dict = {}
        self._reverse_transform_dict = {}
        self._label_count = None

    def fit(self, x: ndarray, y: ndarray)->None:
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

    def build_dict(self,y:ndarray)->None:
        """
        This function will build the tranform diction from the y label data.
        @params:
        y:in array form.
        @outputs:
        None
        """
        powerset = set([_ for _ in y])
        self._label_count = 0
        for i in powerset:
            self._transform_dict[i] = self._label_count
            self._reverse_transform_dict[self._label_count] = i
            self._label_count+=1
        return

    def predict(self, x: ndarray)->ndarray:
        """
        This function will give prediction to given x.
        WARNING: this function's behavior might change if the LP classifier's implememt is changed.
        @params:
        x: data label in array form.
        @outputs:
        y: predicted label, 1 for positive,-1 for negative and 0 for masked.
        """
        return self.inverser_transform(self._classifier.predict(x))

    def transform(self, y: ndarray)->int:
        """
        This function will transform the origin y to a unique label so the origin problem is 
        changed into a multiclass classification problem.
        WARNING: error if called before fit() is called.
        @params:
        y: target labels, in t*i form or 1*i form.
        @outputs:
        a integer stands for the corresponding class that those label belong to.
        """
        if len(y.shape)>1:
            return array([self._transform_dict[y_i] for y_i in y])
        else :
            return self._transform_dict[y]
    
    def inverse_transform(self, y)->ndarray:
        """
        This function will transform the predicted class to origin labels.
        WARNING: error if called before fit() is called.
        @params:
        y: a INTEGER stands for the class of a set of labels.
        @outputs:
        a array that stands the origin label situation,1 for positive, 
        -1 for negative and 0 for masked
        """
        if len(y.shape)>1:
            return array([self._transform_dict[y_i] for y_i in y])
        else :
            return self._transform_dict[y]


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
        @output:
        None
        """
        # assert(k<train_y.shape[1])
        # assert(m < factorial(train_y.shape[1])/factorial(m)/factorial(train_y.shape[1]-k))
        self._LP_classifiers = [LP(self._LP_type, kwargs) for _ in range(m)]
        self._k = k
        self._m = m
        label_set = [i for i in range(train_y.shape[1])]
        k_set = set()
        while len(k_set) < self._k:
            t = sample(label_set, self._m)
            k_set = k_set.union(set(t))
        # In our realization, we use 1 for positive, -1 for negative and 0 for not in current subset.
        self._masks = []
        for k_l in k_set:
            mask = zeros(train_y.shape[1])
            mask[[idx for idx in k_l]] = [1 for idx in k_l]
            self._masks.append(mask)
        for i in range(m):
            self._LP_classifiers[i].fit(train_x,train_y*self._masks[i])
        return

    def predict(self,x, *args, **kwargs):
        """
        This function will give the prediction to the given labels.
        @params:
        x: a single data
        @outputs:
        result: predicted labels, 1 for positive,-1 for negative and 0 for masked.
        """
        result = sum([classifier.predict(x) for classifier in self._LP_classifiers],axis=1)
        result = sign(result*2+1)
        return result
    
    def eval(self,val_X, val_Y, print_result: bool = True, save_result: bool = False, *args, **kwargs):
        """
        This function will give the prediction to the given labels.
        @params:
        val_x: validation labels
        val_y: validation targets
        metric: a dict of metric function, if given use all the function inside it to evaluate, otherwise use default metrics
        TODO
        @output:
        None
        """
        start_time = time()
        metrics = {}
        pred_Y = []
        for x in tqdm(val_X,desc='predicting...'):
            pred_Y.append(self.predict(x))
        print('evaluating...')
        metrics['F1 score'] = f1_score(val_Y,pred_Y)
        metrics['Recall score'] = recall_score(val_Y,pred_Y)
        print('evaluation done.')
        end_time = time()
        if save_result:
            if 'savePath' in kwargs:
                result_file = open(kwargs['savePath'], 'wt')
            else :
                result_file = open(ctime()+'result.txt','wt')
            result_file.write('experiment time:'+ctime()+'\n')
            result_file.write("run time:"+str(end_time-start_time)+'\n')
            for metric in metrics:
                result_file.write('metric '+str(metric)+':'+str(metrics[metric])+'\n')
            result_file.close()
        if print_result:
            system('cls')
            print('experiment time:'+ctime()+'\n')
            print("run time:"+str(end_time-start_time)+'\n')
            for metric in metrics:
                print('metric '+str(metric)+':'+str(metrics[metric])+'\n')
        return

    @property
    def parameter(self):
        return {
            'type': self._LP_type,
            'k': self._k,
            'm': self._m
        }
