import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import time
import seaborn as sns
import math

def timer(func):
    def wrapper(*args, **kwargs):
        print('Start time: ', time.ctime())
        start_time = time.time()  # start time

        result = func(*args, **kwargs)  # run

        end_time = time.time()  # end time
        print('End time: ', time.ctime())
        print(f"{func.__name__} executed in {(end_time - start_time):.4f} seconds")
        return result
    return wrapper
def pre_processing(X, mode=None):
    if mode == 'min-max':
        print('Pre-process: min-max normalization')
        min_each_feature = np.min(X, axis=0)
        max_each_feature = np.max(X, axis=0)
        scale = max_each_feature - min_each_feature
        scale[scale == 0] = 1   # To avoid divided by 0
        scaled_train = (X - min_each_feature) / scale
        return scaled_train

    if mode == 'standardization':
        print('Pre-process: standardization')
        std_each_feature = np.std(X, axis=0)
        mean_each_feature = np.mean(X, axis=0)
        std_each_feature[std_each_feature == 0] = 1     # To avoid divided by 0
        norm_train = (X - mean_each_feature) / std_each_feature
        norm_test = (X - mean_each_feature) / std_each_feature
        return norm_train

    print('No pre-process')

    return X
def accuracy(y_hat,y):
    '''
    y_hat : predicted value
    :param y_hat: [batch_size,num_of_class]
    :param y: [batch_size,1]
    :return:
    '''
    preds=y_hat.argmax(axis=1,keepdims=True)
    return np.mean(preds == y)*100
def Xavier_init(n_in,n_out):
    W=np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )
    return W
def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================
    """

    if nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

def _calculate_fan_in_and_fan_out(array):
    dimensions = len(array.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = array.shape[1]
    num_output_fmaps = array.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in array.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(array, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(array)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_normal_(array: np.array, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'relu'):
    fan = _calculate_correct_fan(array, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, array.shape)
class Parameter(object):
    """Parameter class for saving data and gradients"""
    def __init__(self, data, requires_grad, skip_decay=False):
        self.data = data
        self.grad = None
        self.skip_decay = skip_decay
        self.requires_grad = requires_grad
class Layer(object):
    def __init__(self, name, requires_grad=False):
        self.name = name
        self.requires_grad = requires_grad

    def _forward(self, *args):
        pass

    def _backward(self, *args):
        pass
class ReLU(Layer):
    def __init__(self, name, requires_grad=False):
        super().__init__(name, requires_grad)

    def _forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def _backward(self, delta):
        delta[self.x <= 0] = 0
        return delta


class FCLayer(Layer):
    def __init__(self, name: str, n_in: int, n_out: int) -> None:
        '''
        Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,)
        :param n_in: dimensionality of input
        :param n_out: number of hidden units
        '''
        super().__init__(name, requires_grad=True)
        self.n_in = n_in
        self.n_out = n_out
        W = kaiming_normal_(np.array([0] * n_in * n_out).reshape(n_in, n_out), a=math.sqrt(5))
        # W = Xavier_init(n_in, n_out)
        self.W = Parameter(W, self.requires_grad)
        self.b = Parameter(np.zeros(n_out), self.requires_grad)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """
            x: [batch size, n_in]
            W: [n_in, n_out]
            b: [n_out]
        """
        self.x = x
        temp = x @ self.W.data + self.b.data
        # [batch_size,n_in] @ [n_in,n_out] + [n_output] => [batch_size,n_out]
        return x @ self.W.data + self.b.data

    def _backward(self, delta: np.ndarray) -> np.ndarray:
        '''
        delta: the gradient of the loss function respect to this layer's output 这层损失函数对于这层输出的梯度
        :param delta: [batch size, n_out]:
        :return:
        '''
        batch_size = delta.shape[0]
        self.W.grad = self.x.T @ delta / batch_size  # [batch_size,n_in]^T @ [batch size, n_out] => [n_in,n_out]
        self.b.grad = delta.sum(axis=0) / batch_size  # divide by batch size to get average of gradient
        return delta @ self.W.data.T  # return the gradient of input(x) back to last layer


class SoftmaxWithCrossEntropyLoss(Layer):

    def __init__(self, name, requires_grad=False):
        super().__init__(name, requires_grad)
        self.probs = None  # Store softmax probabilities
        self.labels = None  # Store ground truth labels

    def _forward(self, input, labels):
        """
        Compute the softmax probabilities and cross-entropy loss.

        Args:
            input: Logits (raw scores) from the model.
            labels: Ground truth labels (one-hot encoded).

        Returns:
            Cross-entropy loss.
        """
        # Softmax computation (with numerical stability)
        x_max = input.max(axis=-1, keepdims=True)
        x_exp = np.exp(input - x_max)
        self.probs = x_exp / x_exp.sum(axis=-1, keepdims=True)

        # Store ground truth labels
        self.labels = labels

        # Cross-entropy loss computation
        epsilon = 1e-12  # To avoid log(0)
        loss = -np.sum(labels * np.log(self.probs + epsilon)) / labels.shape[0]
        return loss

    def _backward(self):
        """
        Compute the gradient of the loss with respect to the input logits.

        Returns:
            Gradient of the loss with respect to the input logits.
        """
        # Gradient of cross-entropy loss with softmax
        return (self.probs - self.labels) / self.labels.shape[0]
class Softmax(Layer):
    def __init__(self,name,requires_grad=False):
        super().__init__(name,requires_grad)
    def _forward(self, x: np.ndarray) -> np.ndarray:
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x_exp/x_exp.sum(axis=1, keepdims=True)
    def _backward(self, delta: np.ndarray) -> np.ndarray:
        return delta


class CrossEntropy(object):
    def __init__(self):
        self.softmax = Softmax('softmax')

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''

        :param x:
        :param y: [batch_size, 1]
        :return:
        '''
        self.batch_size = x.shape[0]
        self.class_num = x.shape[1]

        y_hat = self.softmax._forward(x) #[batch_size,num_class]

        y=self.one_hot_encoding(y)
        self.grad = y_hat - y

        loss = -1 * (y * np.log(y_hat + 1e-8)).sum() / self.batch_size  # to avoid divided by 0
        return loss

    def one_hot_encoding(self, x):
        one_hot_encoded = np.zeros((self.batch_size, self.class_num))
        one_hot_encoded[np.arange(x.shape[0]), x.flatten()] = 1
        return one_hot_encoded

class MLP(object):
    def __init__(self):
        self.layers = []
        self.num_layers = 0
        self.params=[]
    def add_layer(self,layer):
        self.layers.append(layer)
        self.num_layers+=1
        if layer.requires_grad:
            if hasattr(layer,'W'):
                self.params.append(layer.W)
            if hasattr(layer,'b'):
                self.params.append(layer.b)
            # if hasattr(layer,'gamma'):
            #     self.params.append(layer.gamma)
            # if hasattr(layer,'beta'):
            #     self.params.append(layer.beta)
    def _forward(self,x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer._forward(x)
        return x
    def _backward(self,x: np.ndarray) -> np.ndarray:
        #backward from the last layer to the first layer
        for layer in self.layers[::-1]:
            x = layer._backward(x)
        return x
    def _fit(self,mode='train'):
        if mode=='train':
            for layer in self.layers:
                layer.train=True
        elif mode=='eval':
            for layer in self.layers:
                layer.train=False


'''
    {'type': 'linear','params':{'name':'fc1','n_in':128,'n_out':256}},
    {'type': 'dropout', 'params': {'name': 'dropout1', 'drop_rate': 0.3}},
    {'type':'relu', 'params': {'name': 'relu1'}},
    {'type':'linear', 'params': {'name': 'fc2', 'n_in':256,'n_out':128}},
    {'type': 'dropout', 'params': {'name': 'dropout2', 'drop_rate': 0.3}},
    {'type': 'relu', 'params': {'name': 'relu2'}},
    {'type': 'linear', 'params': {'name': 'fc3', 'n_in': 128, 'n_out': 10}},
'''


class MLP_V2():
    def __init__(self):
        self.layers = [
            FCLayer('fc1', n_in=128, n_out=512),
            Dropout('dropout1', 0.6),
            ReLU('relu1'),
            FCLayer('fc2', n_in=512, n_out=256),
            Dropout('dropout2', 0.4),
            ReLU('relu2'),
            FCLayer('fc3', n_in=256, n_out=128),
            ReLU('relu3'),
            FCLayer('fc4', n_in=128, n_out=10)
        ]
        self.params = []
        for layer in self.layers:
            if layer.requires_grad:
                if hasattr(layer, 'W'):
                    self.params.append(layer.W)
                if hasattr(layer, 'b'):
                    self.params.append(layer.b)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x= layer._forward(x)
        return x


    def _backward(self, delta: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            delta= layer._backward(delta)
        return delta

    def _fit(self,mode='train'):
        if mode=='train':
            for layer in self.layers:
                layer.train=True
        elif mode=='eval':
            for layer in self.layers:
                layer.train=False

class SGDMomentum(object):
    def __init__(self,parameters,lr=0.01,momentum=0.9,weight_decay=0.0001):
        self.parameters = parameters
        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.v = [np.zeros(p.data.shape) for p in self.parameters]
    def step(self):
        for i,(v,p) in enumerate(zip(self.v,self.parameters)):
            if not p.skip_decay:
                p.data -= self.weight_decay * p.data
            v = self.momentum * v + self.lr * p.grad
            self.v[i] = v
            p.data -= self.v[i]



class AverageMeterics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class Dropout(Layer):
    def __init__(self, name, drop_rate, requires_grad=False):
        super().__init__(name, requires_grad)
        self.drop_rate = drop_rate
        self.fix_value = 1 / (1 - self.drop_rate)   # to keep average fixed

    def _forward(self, x):
        if self.train:
            self.mask = np.random.uniform(0, 1, x.shape) > self.drop_rate
            return x * self.mask * self.fix_value
        else:
            return x

    def _backward(self, grad_output):
        if self.train:
            return grad_output * self.mask
        else:
            return grad_output

class Adam:
    pass


class CosineLR:
    pass


class Trainer(object):
    def __init__(self, config, model=None, train_loader=None, valid_loader=None):
        self.config = config
        self.epochs = self.config['epoch']
        self.lr = self.config['lr']
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.print_freq = self.config['print_freq']
        # self.scheduler= self.config['scheduler']
        self.train_precision = []
        self.valid_precision = []
        self.train_loss = []
        self.valid_loss = []
        self.criterion = CrossEntropy()
        if self.config['optimizer'] == 'sgd':
            self.optimizer = SGDMomentum(self.model.params, self.lr, self.config['momentum'],
                                         self.config['weight_decay'])
        # elif self.config['optimizer'] == 'adam':
        #     self.optimizer = Adam(self.model.params, self.lr, self.config['weight_decay'])
        # if self.scheduler == 'cos':
        #     self.train_scheduler = CosineLR(self.optimizer, T_max=self.epochs)

    @timer
    def train(self):
        best_accuracy = 0
        for epoch in range(self.epochs):
            print('current lr {:.5e}'.format(self.optimizer.lr))
            self.train_per_epoch(epoch)

            acc1 = self.validate(epoch)

            # remember best prec@1
            best_acc1 = max(acc1, best_accuracy)
            output_best = 'Best Accuracy: %.3f\n' % (best_acc1)
            print(output_best)

    def train_per_epoch(self, epoch):
        batch_time = AverageMeterics()
        losses = AverageMeterics()
        top1 = AverageMeterics()

        self.model._fit()
        end_time = time.time()
        for i, (X, y) in enumerate(self.train_loader):
            y_hat = self.model._forward(X)

            loss = self.criterion(y_hat, y)

            for param in self.model.params:
                param.grad = np.zeros_like(param.data)

            self.model._backward(self.criterion.grad)
            self.optimizer.step()
            precision = accuracy(y_hat, y)
            losses.update(loss, X.shape[0])
            top1.update(precision, X.shape[0])

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            if (i % self.print_freq == 0) or (i == len(self.train_loader) - 1):
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch + 1, i, len(self.train_loader) - 1, batch_time=batch_time,
                    loss=losses, top1=top1))
        print('EPOCH: {epoch} {flag} Results: Accuracy {top1.avg:.3f} Loss: {losses.avg:.4f}'.format(epoch=epoch + 1,
                                                                                                   flag='train',
                                                                                                   top1=top1,
                                                                                                   losses=losses))
        self.train_loss.append(losses.avg)
        self.train_precision.append(top1.avg)

    def validate(self, epoch):
        batch_time = AverageMeterics()
        losses = AverageMeterics()
        top1 = AverageMeterics()

        self.model._fit(mode='test')

        end = time.time()
        for i, (X, y) in enumerate(self.valid_loader):
            # compute output
            y_hat = self.model._forward(X)
            loss = self.criterion(y_hat, y)

            # measure accuracy and record loss
            precision = accuracy(y_hat, y)
            losses.update(loss, X.shape[0])
            top1.update(precision, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % self.print_freq == 0) or (i == len(self.valid_loader) - 1):
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(self.valid_loader) - 1, batch_time=batch_time, loss=losses,
                    top1=top1))

        print('EPOCH: {epoch} {flag} Results: Accuracy {top1.avg:.3f} Loss: {losses.avg:.4f}'.format(epoch=epoch + 1,
                                                                                                   flag='val',
                                                                                                   top1=top1,
                                                                                                   losses=losses))
        self.valid_loss.append(losses.avg)
        self.valid_precision.append(top1.avg)

        return top1.avg

class Dataloader(object):
    def __init__(self, X, y, batch_size, shuffle=True, seed=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.index = np.arange(X.shape[0])

    def __iter__(self):
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(self.index)
        self.n = 0
        return self

    def __next__(self):
        if self.n >= len(self.index):
            raise StopIteration

        index = self.index[self.n:self.n + self.batch_size]
        batch_X = self.X[index]
        batch_y = self.y[index]
        self.n += self.batch_size

        return batch_X, batch_y

    def __len__(self):
        """
            num of batch
        """
        return (len(self.index) + self.batch_size - 1) // self.batch_size  # ceiling
def build_model(layers):
    model = MLP()
    str2obj = {'linear': FCLayer, 'relu': ReLU, 'dropout': Dropout}
    for i in layers:
        model.add_layer(str2obj[i['type']](**i['params']))
    return model


layers=[
    {'type': 'linear','params':{'name':'fc1','n_in':128,'n_out':256}},
    {'type': 'dropout', 'params': {'name': 'dropout1', 'drop_rate': 0.3}},
    {'type':'relu', 'params': {'name': 'relu1'}},
    {'type':'linear', 'params': {'name': 'fc2', 'n_in':256,'n_out':128}},
    {'type': 'dropout', 'params': {'name': 'dropout2', 'drop_rate': 0.3}},
    {'type': 'relu', 'params': {'name': 'relu2'}},
    {'type': 'linear', 'params': {'name': 'fc3', 'n_in': 128, 'n_out': 10}},
]
batch_size=1024
config={'layers': layers,'lr': 0.1,'batch_size': batch_size,'momentum': 0.9,'weight_decay': 5e-4,'seed': 0,'epoch': 200,
    'optimizer': 'sgd',     # adam, sgd
    'scheduler': None,      # cos, None
    'pre-process': 'standardization',      # min-max, standardization, None
    'print_freq': 50000 // batch_size // 5
}
np.random.seed(config['seed'])
dir_path='E:\\Postgraduate\\25S1\\COMP5329\\Assignment\\Assignment1\\Assignment1-Dataset\\'
train_file='train_data.npy'
train_label_file='train_label.npy'
test_file='test_data.npy'
test_label_file='test_label.npy'
train_data=np.load(dir_path+train_file)
train_label=np.load(dir_path+train_label_file)
train_X=pre_processing(train_data,config['pre-process'])
train_dataloader=Dataloader(train_X, train_label, config['batch_size'], shuffle=True, seed=config['seed'])
test_X=np.load(dir_path+test_file)
test_label=np.load(dir_path+test_label_file)
test_X=pre_processing(test_X,config['pre-process'])
test_dataloader=Dataloader(test_X, test_label, config['batch_size'], shuffle=False, seed=config['seed'])

# model = build_model(config['layers'])
model=MLP_V2()
trainer=Trainer(config,model,train_dataloader,test_dataloader)
trainer.train()