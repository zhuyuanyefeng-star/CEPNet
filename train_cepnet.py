import random
import torch
import torch.nn as nn
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from tqdm import tqdm
import os
import os.path as ops
import numpy as np
import time

# 自定义模块
from utils.Adan import Adan                    # Adan优化器（比Adam更稳定/收敛更快）
from utils.data import SirstDataset, IRSTD1K_Dataset   # 数据集加载
from utils.lr_scheduler import adjust_learning_rate, create_lr_scheduler  # 学习率调度
from model.loss import SoftIoULoss, criterion, criterion_bce_softiou, LCFocalIoULoss, criterion_LC  # 损失函数
from model.metrics import IoUMetric, nIoUMetric, PD_FA  # 评价指标
from model.cepnet import CEPNet, CEPNet_M, CEPNet_L       # 三种规模模型

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ========================= 参数解析 =========================
def parse_args():
    parser = ArgumentParser(description='Train CEPNet for infrared small target detection')

    parser.add_argument('--img_size', type=int, default=512, help='输入图像尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--warm_up_epochs', type=int, default=10, help='预热轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')

    parser.add_argument('--dataset', type=str, default='sirst', help='数据集: sirst 或 IRSTD-1k')
    parser.add_argument('--mode', type=str, default='S', help='模型规模: L / M / S')
    parser.add_argument('--ablation', type=str, default='CEPNet', choices=['A0', 'A1', 'CEPNet'],
                        help='A0 backbone, A1 +CGDAF, CEPNet +CGDAF+LPP-FHS')
    parser.add_argument('--seed', type=int, default=2026, help='random seed')

    parser.add_argument('--resume', default='', help='断点续训路径')
    parser.add_argument('--start_epoch', default=1, type=int, help='起始epoch')

    parser.add_argument('--amp', default=True, help='是否使用混合精度训练')

    args = parser.parse_args()
    if args.dataset.lower() == 'irstd-1k':
        args.dataset = 'IRSTD-1k'
    return args


# ========================= 训练器类 =========================
class Trainer(object):

    def __init__(self, args):
        self.args = args
        set_seed(args.seed)

        # ---------- 1. 数据集加载 ----------
        if args.dataset == 'sirst':
            self.train_set = SirstDataset(args, mode='train')
            self.val_set = SirstDataset(args, mode='val')
        elif args.dataset == 'IRSTD-1k':
            self.train_set = IRSTD1K_Dataset(args, mode='train')
            self.val_set = IRSTD1K_Dataset(args, mode='val')
        else:
            NameError

        # DataLoader（多线程加载 + pin_memory加速GPU拷贝）
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        self.train_data_loader = Data.DataLoader(
            self.train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=8, pin_memory=True, worker_init_fn=seed_worker, generator=generator)

        self.val_data_loader = Data.DataLoader(
            self.val_set, batch_size=args.batch_size,
            num_workers=8, pin_memory=True, worker_init_fn=seed_worker, generator=generator)

        # ---------- 2. 模型选择 ----------
        assert args.mode in ['L', 'M', 'S']
        if args.mode == 'L':
            self.net = CEPNet_L(ablation=args.ablation)
        elif args.mode == 'M':
            self.net = CEPNet_M(ablation=args.ablation)
        elif args.mode == 'S':
            self.net = CEPNet(ablation=args.ablation)

        self.net = self.net.cuda()  # 移到GPU

        # 混合精度（节省显存+加速）
        self.scaler = torch.amp.GradScaler('cuda') if args.amp else None

        # ---------- 3. 损失函数 ----------
        self.criterion = criterion_bce_softiou  # BCE on all outputs + light main-output SoftIoU

        # ---------- 4. 优化器 ----------
        self.optimizer = Adan(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        # ---------- 5. 学习率调度 ----------
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer,
            len(self.train_data_loader),
            args.epochs,
            warmup=True,
            warmup_epochs=args.warm_up_epochs
        )

        # ---------- 6. 断点续训 ----------
        if args.resume:
            checkpoint = torch.load(args.resume, weights_only=False)
            self.net.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

            if self.args.amp:
                self.scaler.load_state_dict(checkpoint["scaler"])

        # ---------- 7. 指标 ----------
        self.iou_metric = IoUMetric()
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)

        self.best_iou = 0
        self.best_nIoU = 0
        self.best_PD = 0
        self.best_FA = 1

        self.PD_FA = PD_FA(args.img_size)

        # ---------- 8. 保存路径 ----------
        if args.resume:
            folder_name = os.path.abspath(
                os.path.dirname(os.path.abspath(os.path.dirname(args.resume) + os.path.sep + "."))
                + os.path.sep + ".")
        else:
            folder_name = '%s_bs%s_lr%s' % (
                f"{args.ablation}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))}",
                args.batch_size, args.learning_rate)

        # SIRST
        if self.train_set.__class__.__name__ == 'SirstDataset':
            self.save_folder = ops.join('results_sirst_cepnet/', folder_name)
            self.save_pth = ops.join(self.save_folder, 'checkpoint')

            os.makedirs(self.save_pth, exist_ok=True)

        # IRSTD-1k
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            self.save_folder2 = ops.join('results_IRSTD-1k_cepnet/', folder_name)
            self.save_pth2 = ops.join(self.save_folder2, 'checkpoint')

            os.makedirs(self.save_pth2, exist_ok=True)

        # ---------- 9. TensorBoard ----------
        if self.train_set.__class__.__name__ == 'SirstDataset':
            self.writer = SummaryWriter(log_dir=self.save_folder)
        else:
            self.writer = SummaryWriter(log_dir=self.save_folder2)

        self.writer.add_text(folder_name, 'Args:%s, ' % args)

        print('Args: %s' % args)

    # ========================= 训练 =========================
    def training(self, epoch):

        losses = []
        self.net.train()

        tbar = tqdm(self.train_data_loader)
        for i, (data, labels) in enumerate(tbar):
            data, labels = data.cuda(), labels.cuda()

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.net(data)
                # 最关键的一步：把输入图像 data 传给 input_img 参数，让 Loss 能够提取物理先验
                loss = self.criterion(output, labels)

            self.optimizer.zero_grad()

            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()  # 更新学习率

            losses.append(loss.item())

            # tqdm显示
            tbar.set_description(
                'Epoch:%3d, lr:%f, train loss:%f'
                % (epoch, self.optimizer.param_groups[0]['lr'], np.mean(losses))
            )

        # TensorBoard记录
        self.writer.add_scalar('Losses/train loss', np.mean(losses), epoch)
        self.writer.add_scalar('Learning rate/', self.optimizer.param_groups[0]['lr'], epoch)

    # ========================= 验证 =========================
    def validation(self, epoch):

        # 重置指标
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        eval_losses = []
        self.net.eval()

        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):

            with torch.no_grad():
                output = self.net(data.cuda())
                output = output.cpu()

            loss = self.criterion(output, labels)
            eval_losses.append(loss.item())

            # 更新指标
            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))

            tbar.set_description(
                'Epoch:%3d, eval loss:%f, IoU:%f, nIoU:%f, Fa:%.8f, Pd:%.5f'
                % (epoch, np.mean(eval_losses), IoU, nIoU, Fa, Pd)
            )

        # ---------- 保存模型 ----------
        pkl_name = 'Epoch-%03d_IoU-%.4f_nIoU-%.4f_Fa_%.8f_Pd_%.5f.pth' % (
            epoch, IoU, nIoU, Fa, Pd)

        save_file = {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": self.args
        }

        if self.args.amp:
            save_file["scaler"] = self.scaler.state_dict()

        save_pth = self.save_pth if hasattr(self, 'save_pth') else self.save_pth2
        os.makedirs(save_pth, exist_ok=True)

        # 多指标最优保存
        if IoU > self.best_iou:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_iou = IoU

        if nIoU > self.best_nIoU:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_nIoU = nIoU

        if Pd > self.best_PD:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_PD = Pd

        if Fa < self.best_FA:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_FA = Fa

        # TensorBoard记录
        self.writer.add_scalar('Eval/IoU', IoU, epoch)
        self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        self.writer.add_scalar('Eval/Pd', Pd, epoch)
        self.writer.add_scalar('Eval/Fa', Fa, epoch)


# ========================= 主函数 =========================
if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)

    for epoch in range(args.start_epoch, args.epochs + 1):
        trainer.training(epoch)
        trainer.validation(epoch)

    print('Best IoU: %.5f, best nIoU: %.5f, Best Pd: %.5f, best Fa: %.5f' %
          (trainer.best_iou, trainer.best_nIoU, trainer.best_PD, trainer.best_FA))
