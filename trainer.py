import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from models.matching import Matching
from utils.transformnvidia import ImgWtLossSoftNLL, RelaxedBoundaryLossToTensor
from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

def sliding_window(sequence, winSize, step=1, stride=1, drop_last=False):
    """Returns a generator that will iterate through
    the defined chunks of input sequence. Input sequence
    must be sliceable.
    usage:
        a =np.arange(16)
        for i in sliding_window(a,8,step=1):
            print(i)
    """

    # Verify the inputs
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):  # noqa: E721
        print("(type(winSize) == type(0))", (type(winSize) == type(0)))  # noqa: E721
        raise Exception("type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("step must not be larger than winSize.")
    if winSize * stride > len(sequence):
        raise Exception(
            "winSize*stride ={}*{} must not be larger than sequence length={}.".format(winSize, stride, len(sequence))
        )

    n = len(sequence)
    for i in range(0, n, step):
        last = min(i + winSize * stride, n)
        ret = sequence[i:last:stride]
        if not drop_last or len(ret) == winSize:
            yield ret
        if len(ret) != winSize:
            return


def multi_vid_batch_loss(criterion_metric, batch, targets, num_vid_example):
    """ multiple view-pair in batch, metric loss for multi example for frame , only 2 view support"""
    batch_size = batch.size(0)
    emb_view0, emb_view1 = batch[: batch_size // 2], batch[batch_size // 2 :]
    t_view0, t_view1 = targets[: batch_size // 2], targets[batch_size // 2 :]
    batch_example_vid = emb_view0.size(0) // num_vid_example
    slid_vid = lambda x: sliding_window(x, winSize=batch_example_vid, step=batch_example_vid)
    loss = torch.zeros(1).cuda()
    # compute loss for each video
    for emb_view0_vid, emb_view1_vid, t0, t1 in zip(
        slid_vid(emb_view0), slid_vid(emb_view1), slid_vid(t_view0), slid_vid(t_view1)
    ):
        if isinstance(criterion_metric,NpairLoss):
            loss += criterion_metric(emb_view0_vid, emb_view1_vid, t0)
        else:
            loss += criterion_metric(torch.cat((emb_view0_vid, emb_view1_vid)), torch.cat((t0, t1)))
    return loss

def _pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else res.clamp(min=eps).sqrt()


class LiftedStruct(nn.Module):
    """Lifted Structured Feature Embedding
    https://arxiv.org/abs/1511.06452
    based on: https://github.com/vadimkantorov/metriclearningbench
    see also: https://gist.github.com/bkj/565c5e145786cfd362cffdbd8c089cf4
    """

    def forward(self, embeddings, labels, margin=1.0, eps=1e-4):
        loss = torch.zeros(1)
        if torch.cuda.is_available():
            loss = loss.cuda()
        # L_{ij} = \log (\sum_{i, k} exp\{m - D_{ik}\} + \sum_{j, l} exp\{m - D_{jl}\}) + D_{ij}
        # L = \frac{1}{2|P|}\sum_{(i,j)\in P} \max(0, J_{i,j})^2
        d = _pdist(embeddings, squared=False, eps=eps)
        # pos mat  1 where labels are same for distance mat
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)

        neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
        loss += torch.sum(F.relu(pos.triu(1) * ((neg_i + neg_i.t()).log() + d)).pow(2)) / (pos.sum() - len(d))
        return loss

def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


class NpairLoss(nn.Module):
    """the multi-class n-pair loss
        from: https://github.com/ChaofWang/Npair_loss_pytorch/blob/master/Npair_loss.py
    """

    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        '''
            target are class labels
        '''
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss

class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True
        self.examples_per_seq = config['train_loader']['args']['examples_per_seq']
        # self.metric_criterion = losses.LiftedStructureLoss().to(self.device)
        # self.metric_criterion = losses.NPairsLoss().to(self.device)
        self.metric_criterion = losses.NCALoss().to(self.device)


    def _train_epoch(self, epoch):
        self.logger.info('\n')

        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target, input_a, mkpt0, mkpt1,m_cnt) in enumerate(tbar):
            # self._valid_epoch(epoch) # DEBUG
            self.data_time.update(time.time() - tic)
            #data, target = data.to(self.device), target.to(self.device)
            self.lr_scheduler.step(epoch=epoch-1)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            output = self.model(data)
            # assert output.size()[2:] == target.size()[1:], "output {}, target {}".format(output.shape,target.shape)
            # assert output.size()[1] == self.num_classes
            loss = self.loss(output, target)
            emb = self.model(torch.cat((data,input_a)),True)

            # loss_sg = self.loss(output_a, target_a_gule)
            n = data.size(0)
            label_positive_pair = np.arange(n)
            # labels = torch.from_numpy(label_positive_pair).to(self.device)
            labels_full = torch.from_numpy(np.concatenate([label_positive_pair, label_positive_pair])).to(self.device)
            # loss_metric = self.metric_criterion(emb,labels)[0]
            # loss_metric = self.metric_criterion(emb[:n],emb[n:],labels)
            loss_metric = multi_vid_batch_loss(self.metric_criterion, emb, labels_full, num_vid_example=self.examples_per_seq)[0]

            if isinstance(self.loss, torch.nn.DataParallel):
                loss_metric = loss_metric.mean()
                loss = loss.mean()
            loss+= loss_metric*0.1
            # loss_metric=torch.tensor(0)
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)
                self.writer.add_scalar(f'{self.wrt_mode}/loss_metric', loss_metric, self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()

            # save_image(torch.cat((data,input_a)),"saved/img{}-{}.png".format(epoch,batch_idx))

            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | m {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                                                epoch,
                                                self.total_loss.average,loss_metric.detach().cpu().numpy(),
                                                pixAcc, mIoU,
                                                self.batch_time.average, self.data_time.average))

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]:
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
                **seg_metrics}

        #if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def relax_labels(self,lbl):
        ignore_label=255
        rlbl_tensor = RelaxedBoundaryLossToTensor(ignore_label,self.num_classes)
        lbl_relaxed = rlbl_tensor(lbl)
        lbl_relaxed = torch.from_numpy(lbl_relaxed).byte()
        return lbl_relaxed

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target, data_a, mkpt0, mkpt1,m_cnt) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                # rm backgroud form loss with lable zero
                seg_metrics_no_bg = eval_metrics(output, target, self.num_classes, rm_class_lable=[2])
                self._update_seg_metrics(*seg_metrics, *seg_metrics_no_bg)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics(no_bg=True)
            for k, v in list(seg_metrics.items())[:-1]:
                if not isinstance(v,dict ):
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }
        del output, data, data_a
        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

        self.total_correct_no_bg = 0
        self.total_inter_no_bg = 0
        self.total_union_no_bg = 0
        self.total_label_no_bg =0

    def _update_seg_metrics(self, correct, labeled, inter, union, correct_no_bg=None,labeled_no_bg=None, inter_no_bg=None, union_no_bg=None):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

        if correct_no_bg is not None:
            self.total_correct_no_bg +=correct_no_bg
            self.total_label_no_bg += labeled_no_bg
            self.total_inter_no_bg += inter_no_bg
            self.total_union_no_bg += union_no_bg

    def _get_seg_metrics(self, no_bg = False):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        ret =  {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }
        if no_bg:
            pixAcc_no_bg = 1.0 * self.total_correct_no_bg / (np.spacing(1) + self.total_label_no_bg)
            IoU_no_bg = 1.0 * self.total_inter / (np.spacing(1) + self.total_union_no_bg)
            mIoU_no_bg = IoU_no_bg.mean()
            ret.update({
                "Pixel_Accuracy_no_bg": np.round(pixAcc_no_bg, 3),
                "Mean_IoU_no_bg": np.round(mIoU_no_bg, 3),
                "Class_IoU_no_bg": dict(zip(range(self.num_classes), np.round(IoU_no_bg, 3)))
            })

        return ret
