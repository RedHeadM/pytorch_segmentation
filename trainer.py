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
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import os

class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True, model_staudet=None,pseudo_loader=None):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger,model_staudet = model_staudet)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        # self.restore_transform = transforms.Compose([
            # local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            # transforms.ToPILImage()])
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()])
        self.pseudo_loader=pseudo_loader

        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

        self._no_pseudo= True
        self.label_relax=False
        if self.label_relax:
            wt_bound = 1.0
            self.label_relax_loss = ImgWtLossSoftNLL(classes=self.num_classes, ignore_index=255, upper_bound=wt_bound)

        self.encoder_momentum = 0.9

    def ias_thresh(self,conf_dict, c, alpha, w=None, gamma=1.0):
        if w is None:
            w = np.ones(c)
        # threshold
        cls_thresh = np.ones(c,dtype = np.float32)
        for idx_cls in np.arange(0, c):
            if conf_dict[idx_cls] != None:
                arr = np.array(conf_dict[idx_cls])
                cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * w[idx_cls] ** gamma))
        return cls_thresh


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        form lightning-bolts
        """
        for param_q, param_k in zip(self.model.parameters(), self.model_staudet.parameters()):
            em = self.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    def _create_pseudo_label(self, net_pseudo, epoch):
        self.logger.info('\n')

        self.model.eval()
        if self.model_staudet:
            self.model_staudet.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.pseudo_loader, ncols=130)
        PSEUDO_PL_ALPHA=0.2

        beta = 0.9
        num_classes = self.train_loader.dataset.num_classes #fg.MODEL.PREDICTOR.NUM_CLASSES
        gamma = 0.8

        cls_thresh = np.ones(num_classes)*0.9
        pseudo_save_dir = self.train_loader.dataset.pseudo_dir


        with torch.no_grad():
            # TODO not aumentations

            for batch_idx, (data, target, input_a, _, comm_name, frame_idx) in enumerate(tbar):
                tbar.update(1)
                b = input_a
                images = Variable(b.cuda())
                names = comm_name
                logits = nn.Softmax(dim=1)(net_pseudo(images))
                # originsize
                # logits = F.interpolate(logits, size=origin_size[::-1], mode="bilinear", align_corners=True)

                max_items = logits.max(dim=1)
                label_pred = max_items[1].data.cpu().numpy()
                logits_pred = max_items[0].data.cpu().numpy()

                logits_cls_dict = {c: [cls_thresh[c]] for c in range(num_classes)}
                for cls in range(num_classes):
                    logits_cls_dict[cls].extend(logits_pred[label_pred == cls].astype(np.float16))

                # instance adaptive selector
                tmp_cls_thresh = self.ias_thresh(logits_cls_dict, alpha=PSEUDO_PL_ALPHA, c=num_classes, w=cls_thresh, gamma=gamma)
                cls_thresh = beta*cls_thresh + (1-beta)*tmp_cls_thresh
                cls_thresh[cls_thresh>=1] = 0.999

                np_logits = logits.data.cpu().numpy()
                dd=True
                for _i, (name,frame) in enumerate(zip(names,frame_idx.cpu().numpy())):
                    name = os.path.splitext(os.path.basename(name))[0]
                    # save pseudo label
                    logit = np_logits[_i].transpose(1,2,0)
                    label = np.argmax(logit, axis=2)
                    logit_amax = np.amax(logit, axis=2)
                    label_cls_thresh = np.apply_along_axis(lambda x: [cls_thresh[e] for e in x], 1, label)
                    ignore_index = logit_amax < label_cls_thresh
                    label[ignore_index] = 255
                    pseudo_label_name = name + str(frame)+'_pseudo_label.png'
                    pseudo_color_label_name = name + str(frame)+'_pseudo_color_label.png'
                    pseudo_label_path = os.path.join(pseudo_save_dir, pseudo_label_name)
                    pseudo_color_label_path = os.path.join(pseudo_save_dir, pseudo_color_label_name)
                    img = Image.fromarray(label.astype(np.uint8)).convert('P').save(pseudo_label_path)
                    palette = self.train_loader.dataset.palette
                    img = colorize_mask(label, palette)
                    if dd:
                        self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', np.array([np.asarray(img)]), epoch)
                        dd= False
                    img = img.save(pseudo_color_label_path)
                    # assert False
        tbar.close()

    def _train_epoch(self, epoch):
        if epoch >5:
            use_pseudo = True
            # if epoch %1 ==0 or self._no_pseudo:
                # self._create_pseudo_label(self.model,epoch)
            self._no_pseudo = False
        else:
            use_pseudo = False

        continue_tain=True
        # use_pseudo = True

        self.logger.info('\n')

        self.model.train()
        if self.model_staudet:
            self.model_staudet.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        val_visual=[]
        tbar = tqdm(self.train_loader, ncols=130)

        for batch_idx, (data, target, input_a, _, comm_name,frame_idx) in enumerate(tbar):

            self._momentum_update_key_encoder()
            # self._valid_epoch(epoch) # DEBUG
            self.data_time.update(time.time() - tic)
            # data, target = data.to(self.device), target.to(self.device)
            # target = target.to(self.device)
            self.lr_scheduler.step(epoch=epoch-1)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            if use_pseudo:
                with torch.no_grad():
                    output_a = self.model_staudet(input_a)
                    pseudo_label = torch.softmax(output_a.detach(), dim=1)
                    max_probs, targets_u_w = torch.max(pseudo_label, dim=1)
                    targets_u_w[max_probs<0.9]= 255
                    # max_probs[max_probs<0.9] = 0

                output = self.model(input_a)
                loss = self.loss(output_a, targets_u_w)*0.9
                if continue_tain:
                    output = self.model(data)
                    loss += self.loss(output, target)
            else:
                output = self.model(data)
                loss = self.loss(output, target)
            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
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

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()

            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                                                epoch, self.total_loss.average,
                                                pixAcc, mIoU,
                                                self.batch_time.average, self.data_time.average))

            # LIST OF IMAGE TO VIZ (15 images)
            visual_num=5
            if use_pseudo:
                if len(val_visual) < visual_num:
                    target_np = targets_u_w.data.cpu().numpy()
                    output_np = output_a.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(),input_a[0].data.cpu(), target_np[0], output_np[0]])
                elif len(val_visual) == visual_num:
                    # WRTING & VISUALIZING THE MASKS
                    val_img = []
                    palette = self.train_loader.dataset.palette
                    for d_l,d, t, o in val_visual:
                        d = self.restore_transform(d)
                        d_l = self.restore_transform(d_l)
                        t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                        d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                        vimg = [self.viz_transform(x) for x in [d_l,d, t, o]]
                        val_img.extend(vimg)
                    val_img = torch.stack(val_img, 0)
                    val_img = make_grid(val_img.cpu(), nrow=4, padding=5)
                    self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

        # METRICS TO TENSORBOARD
        self.wrt_step = (epoch) * len(self.val_loader)
        self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
        seg_metrics = self._get_seg_metrics(no_bg=True)
        for k, v in list(seg_metrics.items())[:-1]:
            if not isinstance(v,dict ):
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

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
        if self.model_staudet:
            self.model_staudet.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target, data_a, mkpt0, mkpt1,m_cnt) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                # LOSS
                if self.model_staudet:
                    output = self.model_staudet(data)
                else:
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
