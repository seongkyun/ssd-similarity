import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.fpn_mobilenetv1_ssd import create_fpn_mobilenetv1_ssd
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from ssd_teacher_net import build_ssd

# For tensorboard
from tensorboardX import SummaryWriter
summary = SummaryWriter(log_dir='/home/han/study/vb_ssd_2/runs/retrain_mb_ssd')
global iter_cnt
iter_cnt = 0

global glb_feature_student
def Get_features4student(self, input, output):
    global glb_feature_student
    glb_feature_student = output
    return None
global glb_feature_teacher
def Get_features4teacher(self, input, output):
    global glb_feature_teacher
    glb_feature_teacher = output
    return None
global glb_grad
def Get_grad(self, ingrad, outgrad):
    global glb_grad
    glb_grad = outgrad
    return None

# python train_ssd.py --datasets ~/data/VOC0712/VOC2007/ ~/data/VOC0712/VOC2012/ --validation_dataset ~/data/VOC0712/VOC2007/ --net vgg16-ssd --batch_size 24 --num_epochs 400 --scheduler cosine --lr 0.001 --t_max 200

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', 
                    default=['/home/han/data/VOC0712/VOC2007', '/home/han/data/VOC0712/VOC2012'],
                    help='Dataset directory path')
parser.add_argument('--validation_dataset', default="/home/han/data/VOC0712/VOC2007/",
                    help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")


parser.add_argument('--teacher_model', default="/home/han/study/vb_ssd_2/models/VOC_SSD300_VGG16_mAP_77.2.pth", 
                    help="The teacher network trained weight file.")
#parser.add_argument('--teacher_net', default="vgg16-ssd",
#                    help="The teacher network architecture, it can be mb1-ssd, mb1-lite-ssd or vgg16-ssd.")
parser.add_argument('--student_net', default="mb1-ssd",
                    help="The student network architecture, it can be mb1-ssd, mb1-lite-ssd or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',  
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')


args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
batch_size = args.batch_size


def train_new(loader, teacher_net, student_net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    global iter_cnt # to use tensorboard

    global glb_feature_student
    global glb_feature_teacher
    global glb_grad

    teacher_net.eval()
    student_net.train()
   
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    running_mse_loss = 0.0

    criterion_mse = torch.nn.MSELoss()

    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        dummy_out = teacher_net(images)
        confidence, locations = student_net(images)

        emb_teacher = torch.tensor(glb_feature_teacher, requires_grad=False, device=DEVICE)
        emb_student = torch.nn.Upsample(size=(19,19), mode='bilinear')(torch.tensor(glb_feature_student
                                    , requires_grad=True, device=DEVICE))
        
        mse_loss = 10 * criterion_mse(emb_student, emb_teacher)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        
        #loss = regression_loss + classification_loss
        loss = regression_loss + classification_loss + mse_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        running_mse_loss += mse_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            avg_mse_loss = running_mse_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}" +
                f"Average Mse Loss: {avg_mse_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            running_mse_loss = 0.0

        if iter_cnt % 10 == 0:
            summary.add_scalar('loss/loss', loss.item(), iter_cnt)
            summary.add_scalar('loss/loss_regression', regression_loss.item(), iter_cnt)
            summary.add_scalar('loss/loss_classification', classification_loss.item(), iter_cnt)
            summary.add_scalar('loss/loss_mse', mse_loss.item(), iter_cnt)
            summary.add_scalars('loss/loss', {"loss_regression": regression_loss.item(),
                                            "loss_classification": classification_loss.item(),
                                            "loss_mse": mse_loss.item(),
                                            "loss": loss.item()}, iter_cnt)
        iter_cnt += 1

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    global iter_cnt # to use tensorboard

    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        #_, confidence, locations = net(images)

        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0



def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            #_, confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def freezenet(net, freeze_base_net, freeze_net):
    if freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
        return net, params

if __name__ == '__main__':
    global glb_feature_student
    global glb_feature_teacher
    global glb_grad
    glb_feature_student = torch.tensor(torch.zeros(batch_size, 1024, 19, 19)
                        , requires_grad=True, device=DEVICE)
    glb_feature_teacher = torch.tensor(torch.zeros(batch_size, 1024, 10, 10)
                        , requires_grad=False, device=DEVICE)
    glb_grad = torch.tensor(torch.zeros(batch_size, 1024, 10, 10)
                        , requires_grad=False, device=DEVICE)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    timer = Timer()

    logging.info(args)
    
    student_create_net = create_mobilenetv1_ssd
    config = mobilenetv1_ssd_config
    
    # for student net
    '''
    if args.student_net == 'vgg16-ssd':
        
        config = vgg_ssd_config
    elif args.student_net == 'mb1-ssd':
        student_create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.student_net == 'mb1-ssd-lite':
        student_create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)        
    '''

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                 transform=train_transform, target_transform=target_transform,
                 dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)

        else:
            raise ValueError(f"Dataset tpye {args.dataset_type} is not supported.")
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="validation")
        logging.info(val_dataset)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Build network.")
    #net = create_net(num_classes)

    #teacher_net = teacher_create_net(num_classes)
    teacher_net = ssd_net = build_ssd('train', 300, num_classes)
    student_net = student_create_net(num_classes)

    try:
        teacher_net.load_weights(args.teacher_model)
    except FileNotFoundError:
        print('ERROR::No pretrained teacher network file found!')
        sys.exit(1)

    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    
    #teacher_net, teacher_params = freezenet(teacher_net, args.freeze_base_net, args.freeze_net)
    student_net, params = freezenet(student_net, args.freeze_base_net, args.freeze_net)

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        student_net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        student_net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        student_net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    student_net.to(DEVICE)
    teacher_net.to(DEVICE)
    
    # parameter setting
    for param in student_net.parameters():
        param.requires_grad=True
    for param in teacher_net.parameters():
        param.requires_grad=False

    # Register hooking
    l_student_conv_1024 = student_net.base_net[13][3]
    l_student_bn = student_net.base_net[13][4]
    l_student_relu = student_net.base_net[13][5]

    l_teacher_conv_1024 = teacher_net.vgg[33]
    l_teacher_relu = teacher_net.vgg[34]

    l_student_relu.register_forward_hook(Get_features4student)
    l_teacher_relu.register_forward_hook(Get_features4teacher)
    l_student_relu.register_backward_hook(Get_grad)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    
    print("Training starting")
    for epoch in range(last_epoch + 1, args.num_epochs):

        scheduler.step()
        #train(train_loader, net, criterion, optimizer,
        #      device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        
        train_new(train_loader, teacher_net, student_net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, student_net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            print('Epoch: %d, val_loss: %.4f, val_regression_loss: %.4f, val_classification_loss: %.4f' 
                % (epoch, val_loss, val_regression_loss, val_classification_loss))
            for param_group in optimizer.param_groups:
                lr_print = param_group['lr']
            print('current learning rate: ', lr_print)    
            model_path = os.path.join(args.checkpoint_folder, f"retrain_mobile_net-Epoch-{epoch}-Loss-{val_loss}.pth")
            student_net.save(model_path)
            logging.info(f"Saved model {model_path}")

