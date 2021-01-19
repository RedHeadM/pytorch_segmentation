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


        # supergure
        config_match = {
            'superpoint': {
                'nms_radius': config['superpoint']['nms_radius'],
                'keypoint_threshold': config['superpoint']['keypoint_threshold'],
                'max_keypoints': config['superpoint']['max_keypoints']
            },
            'superglue': {
                'weights': config['superglue']['weights'],
                'sinkhorn_iterations': config['superglue']['sinkhorn_iterations'],
                'match_threshold': config['superglue']['match_threshold'],
            }
        }
        resize=config['superglue']['resize']
        if len(resize) == 2 and resize[1] == -1:
            resize = resize[0:1]
        if len(resize) == 2:
            print('Will resize to {}x{} (WxH)'.format(
                resize[0], resize[1]))
        elif len(resize) == 1 and resize[0] > 0:
            print('Will resize max dimension to {}'.format(resize[0]))
        elif len(resize) == 1:
            print('Will not resize images')
        else:
            raise ValueError('Cannot specify more than two integers for --resize')

        self.matching = Matching(config_match).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']


    def _get_glue_mask(self,target,mkpt0, mkpt1,m_cnt, ignore_idx=255):
        target_adapt = torch.full_like(target,ignore_idx)
        for i in range(target.size(0)):
        # for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            # rm padding
            m=mkpt1[i,:m_cnt[i]]
            # m=torch.stack(mkpt1[i]).to(torch.long)
            test=target_adapt[i,[2,3],[3,4]]
            m[0] = torch.tensor([[4,4]])
            target_adapt[i,m[:,0],m[:,1]]=target[i,m[:,0],m[:,1]]
        return target_adapt

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
            if self.config['arch']['type'][:3] == 'PSP':
                assert output[0].size()[2:] == target.size()[1:]
                assert output[0].size()[1] == self.num_classes
                loss = self.loss(output[0], target)
                loss += self.loss(output[1], target) * 0.4
                output = output[0]
            else:
                assert output.size()[2:] == target.size()[1:], "output {}, target {}".format(output.shape,target.shape)
                assert output.size()[1] == self.num_classes
                loss = self.loss(output, target)

            output_a = self.model(input_a)
            target_a_gule= self._get_glue_mask(target,mkpt0, mkpt1,m_cnt)
            loss_sg = self.loss(output_a, target_a_gule)
            print('loss_sg: {}'.format(loss_sg))
            print('loss: {}'.format(loss))
            loss+= loss_sg *0.9
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
            for batch_idx, (data, target,data_a, mkpt0, mkpt1,m_cnt) in enumerate(tbar):
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
