import torch, time, os, argparse
import pandas as pd
import numpy as np
from torch import optim
from dataloader import read_dataset, JobFairDataset, splitTrainTestImgs
from customOptims.adamw import AdamW
from customOptims.radam import RAdam
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.utils import seed_everything, get_lr, to_np, soft_voting
from utils.lr_scheduler import defineLRScheduler
from backbone_networks import initialize_model
from utils.metrics import Evaluator
from skorch.net import NeuralNet
from sklearn.model_selection import cross_val_score
from skorch.callbacks import Freezer
from skorch.callbacks import EpochScoring
from skorch.callbacks import Checkpoint
from skorch.helper import predefined_split

current_path = os.getcwd()
DATA_PATH = os.path.join(current_path, 'dataset')
IMG_PATH = os.path.join(current_path, 'dataset/faces_images')
df_train = pd.read_csv(os.path.join(DATA_PATH, 'train_vision.csv'), skipinitialspace=True)
df_test = pd.read_csv(os.path.join(DATA_PATH, 'test_vision.csv'), skipinitialspace=True)

class Trainer(object):
    def __init__(self, args):
        if args.mode == 'train':  ### training mode 일때는 여기만 접근
            print('Training Start...')

            self.args = args
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print('Total Epoches:', args.epochs)

            # Define Evaluator (F1, Acc_class 등의 metric 계산을 정의한 클래스)
            self.evaluator = Evaluator(self.args.class_num)

            # Define Criterion
            weight = None # Calculate class weight when dataset is strongly imbalanced. (see pytorch deeplabV3 code's main_local.py)
            self.criterion = nn.CrossEntropyLoss()




            # 아래 부분 수정할것!!
            # Load dataset (df_train_img_path is for training, df_test_img_path is for calculate leaderboard score.)
            # df_train_img_path, df_test_img_path = splitTrainTestImgs(IMG_path=IMG_PATH, df_train=df_train, df_test=df_test)

            # Pytorch Data loader
            # data, labels, length_data, length_labels = read_dataset(df_train_img_path)
            X_train, X_test, Y_train, Y_test = train_test_split(self.train_dataset, self.labels, test_size=0.1,
                                                                random_state=2019)

            self.train_dataset = JobFairDataset(args, mode='train', )
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers)
            self.validation_dataset = JobFairDataset(args, mode='val', data=Y_train, labels=Y_test, len_data=len(Y_train),
                                                     len_label=len(Y_test))
            self.validation_loader = DataLoader(self.validation_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers)

            print('Dataset class : ', self.args.class_num)
            print('Train/Val dataset length : ' + str(len(self.train_dataset)) + str(len(self.validation_dataset)))

            # 위 부분 수정할것!!

            # Define network
            input_channels = 3 if args.use_additional_annotation else 2  # use_additional_annotation = True이면 3
            model = initialize_model(args.backbone, use_pretrained=False, input_channels=None, num_classes=None)

            # Define Optimizer
            if args.optim.lower() == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
            elif args.optim.lower() == 'adamw':
                optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay)
            elif args.optim.lower() == 'radam':
                optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay)
            else:
                print("Wrong optimizer args input.")
                raise NotImplementedError
            # Define learning rate scheduler
            self.scheduler = defineLRScheduler(args, optimizer, len(self.train_dataset))

            self.model, self.optimizer = model, optimizer
            model.to(self.device)

            # Print parameters to be optimized/updated.
            print("Params to learn:")
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)

            freezer = Freezer('conv*')

            # define skorch_model for grid search
            skorch_model = NeuralNet(
                self.model, criterion=self.criterion, criterion__padding=16, batch_size=32, max_epochs=20,
                optimizer=self.optimizer, optimizer__momentum=args.momentum, iterator_train__shuffle=True, iterator_train__num_workers=4,
                iterator_valid__shuffle=False, iterator_valid__num_workers=4, train_split=predefined_split(val_ds),
                callbacks=[freezer, self.scheduler, Checkpoint(f_params='best_params.pt')],device='cuda', )

            # 5-fold grid cross validation
            scores = cross_val_score(skorch_model, X_data, y_data, cv=5, scoring="accuracy")



            # Train the model
            self.training(args.epoch)

    def training(self, epochs):
        self.model.train() # Train모드로 전환
        num_img_tr = len(self.train_dataset)
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                image, target = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(image) # 각 뷰포인트마다 2개의 softmax결과 * 4개 = 8개의 softmax (python set으로 반환됨)
                output = soft_voting(outputs) # voting해서 하나의 클래스만 남기도록 하는 부분 추가 (set의 voting)

                loss_LMLO = self.criterion(outputs[VIEWS.LMLO], labels)
                loss_RMLO = self.criterion(outputs[VIEWS.RMLO], labels)
                loss_LCC = self.criterion(outputs[VIEWS.LCC], labels)
                loss_RCC = self.criterion(outputs[VIEWS.RCC], labels)
                loss_VIEWS = [loss_LMLO, loss_RMLO, loss_LCC, loss_RCC]
                loss = self.criterion(output, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                predict_vector = np.argmax(to_np(output), axis=1)
                label_vector = to_np(labels)
                bool_vector = predict_vector == label_vector
                accuracy = bool_vector.sum() / len(bool_vector)

                log_batch = 'Epoch {}  Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f}'.format(
                    int(epoch), int(batch_idx), len(self.train_dataset), float(loss.item()), float(accuracy))
                if batch_idx % 10 == 0: # 10스텝마다 출력
                    print(log_batch)

                total_loss += loss.item()
                total_correct += bool_vector.sum()

            log_epoch = 'Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f}'.format(
                epoch, epochs, total_loss / num_img_tr, total_correct / num_img_tr)
            if epoch / 2 == 0:
                print(log_epoch)

            if epoch / 5 == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.evaluator.reset()  # metric이 정의되어 있는 evaluator클래스 초기화 (confusion matrix 초기화 수행)
                    length_val_dataloader = len(self.validation_dataset)
                    print("Start epoch validation...")

                    for item in self.validation_loader:
                        images = item['image'].to(device)
                        labels = item['label'].to(device)

                        outputs = self.model(images)  # 각 뷰포인트마다 2개의 softmax결과 * 4개 = 8개의 softmax (python set으로 반환됨)
                        output = soft_voting(outputs)  # voting해서 하나의 클래스만 남기도록 하는 부분 추가 (set의 voting)

                        predict_vector = np.argmax(to_np(output), axis=1)
                        label_vector = to_np(labels)
                        bool_vector = predict_vector == label_vector
                        accuracy = bool_vector.sum() / len(bool_vector)

                        log_validacc = 'Validation Acc of the model on {} images : {}'.format(length_val_dataloader, accuracy)
                        print(log_validacc)


def main():
    # Set base parameters (dataset path, backbone name etc...)
    parser = argparse.ArgumentParser(description="This code is for testing various octConv+ResNet.")
    parser.add_argument('--backbone', type=str, default='oct_resnet26',
                        choices=['resnet101', 'resnet152' # Original ResNet 
                            'resnext50_32x4d', 'resnext101_32x8d', # Modified ResNet
                            'oct_resnet26', 'oct_resnet50', 'oct_resnet101', 'oct_resnet152', 'oct_resnet200', # OctConv + Original ResNet
                            'senet154', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', # Squeeze and excitation module based models
                            'efficientnetb3', 'efficientnetb4', 'efficientnetb5'], # EfficientNet models
                        help='Set backbone name')
    parser.add_argument('--dataset', type=str, default='KHD_NSML',
                        choices=['local', 'KHD_NSML'],
                        help='Set dataset path. `local` is for testing via local device, KHD_NSML is for testing via NSML server. ')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Set CPU threads for pytorch dataloader')
    parser.add_argument('--checkname', type=str, default=None,
                        help='Set the checkpoint name. if None, checkname will be set to current dataset+backbone+time.')

    # Set hyper params for training network.
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Set k_folds params for stratified K-fold cross validation.')
    parser.add_argument('--distributed', type=bool, default=None,
                        help='Whether to use distributed GPUs. (default: None)')
    parser.add_argument('--sync_bn', type=bool, default=None,
                        help='Whether to use sync bn (default: auto)')
    parser.add_argument('--freeze_bn', type=bool, default=False,
                        help='Whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='Set loss func type. `ce` is crossentropy, `focal` is focal entropy from DeeplabV3.')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='Set max epoch. If None, max epoch will be set to current dataset`s max epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--class_num', type=int, default=None,
                        help='Set class number. If None, class_num will be set according to dataset`s class number.')
    parser.add_argument('--use_pretrained', type=bool, default=False) # ImageNet pre-trained model 사용여부
    parser.add_argument('--use_additional_annotation', type=bool, default=True, help='Whether use additional annotation') # 데이터셋에 악성 종양에 대한 세그먼트 어노테이션이 있는 경우 True

    # Set optimizer params for training network.
    parser.add_argument('--lr', type=float, default=None,
                        help='Set starting learning rate. If None, lr will be set to current dataset`s lr.')
    parser.add_argument('--lr_scheduler', type=str, default='WarmupCosineWithHardRestartsSchedule',
                        choices=['StepLR', 'MultiStepLR', 'WarmupCosineSchedule', 'WarmupCosineWithHardRestartsSchedule'],
                        help='Set lr scheduler mode: (default: WarmupCosineSchedule)')
    parser.add_argument('--optim', type=str, default='RAdam',
                        choices=['SGD', 'ADAM', 'AdamW', 'RAdam'],
                        help='Set optimizer type. (default: RAdam)')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='Set momentum value for pytorch`s SGD optimizer. (default: 0.9)')

    # Set params for CUDA, seed and logging
    parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=2019, metavar='S', help='random seed (default: 2019)')

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    parser.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    parser.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args = parser.parse_args()
    print('cuDNN version : ', torch.backends.cudnn.version())

    # default settings for epochs, lr and class_num of dataset.
    if args.epochs is None:
        epoches = {'local': 150, 'KHD_NSML': 100, }
        args.epochs = epoches[args.dataset]

    if args.lr is None:
        lrs = {'local': 0.1, 'KHD_NSML': 0.1,}
        args.lr = lrs[args.dataset]

    if args.class_num is None:
        # local은 KaKR 3rd 자동차 차종분류 데이터셋인 경우 192개의 차종 클래스
        # KHD_NSML은 정상(normal), 양성(benign), 악성(malignant) 3개의 클래스
        class_nums = {'local': 192, 'KHD_NSML': 3, }
        args.class_num  = class_nums[args.dataset]

    if args.checkname is None:
        now = datetime.now()
        args.checkname = str(args.dataset) + '-' + str(args.backbone) + ('%s-%s-%s' % (now.year, now.month, now.day))

    print(args)
    torch.manual_seed(args.seed)

    # Define trainer. (Define dataloader, model, optimizer etc...)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)

    trainer.training(args.epoch)
    # if not trainer.args.no_val and args.epoch % args.eval_interval == (args.eval_interval - 1):
    #     trainer.validation(args.epoch)


if __name__ == "__main__":
    try:
        from apex import amp
        from apex.parallel import DistributedDataParallel as DDP
        from apex.parallel import convert_syncbn_model
        has_apex = True
    except ImportError:
        from torch.nn.parallel import DistributedDataParallel as DDP
        has_apex = False
    cudnn.benchmark = True

    SEED = 20191129
    seed_everything(SEED) # 하이퍼파라미터 테스트를 위해 모든 시드 고정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main()