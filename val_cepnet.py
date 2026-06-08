import torch
import torchvision.transforms as transforms
import cv2
import torch.utils.data as Data

from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
# 导入自定义的评估指标类（IoU, nIoU, Pd, Fa, ROC曲线等）
from model.metrics import IoUMetric, nIoUMetric, PD_FA, ROCMetric2
# 导入数据集读取类
from utils.data import SirstDataset, IRSTD1K_Dataset
# 导入模型定义
from model.cepnet import CEPNet, CEPNet_M, CEPNet_L

# 自动选择运行设备：有显卡用CUDA，没显卡用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """解析命令行参数，方便在终端修改配置"""
    parser = ArgumentParser(description='CEPNet 模型验证脚本')

    parser.add_argument('--img_size', type=int, default=512, help='输入图像的尺寸')
    parser.add_argument('--batch_size', type=int, default=1, help='测试时的批大小，通常设为1')
    parser.add_argument('--dataset', type=str, default='sirst', help='选择数据集: sirst 或 IRSTD-1k')
    parser.add_argument('--mode', type=str, default='S', help='模型规模: L(大), M(中), S(小)')
    parser.add_argument('--ablation', type=str, default='CEPNet', choices=['A0', 'A1', 'CEPNet'],
                        help='A0 backbone, A1 +CGDAF, CEPNet +CGDAF+LPP-FHS')
    parser.add_argument('--checkpoint', type=str, default=r'Epoch-337_IoU-0.7921_nIoU-0.7710_Fa_0.00000883_Pd_0.97222.pth', help='训练好的模型权重文件路径')

    args = parser.parse_args()
    if args.dataset.lower() == 'irstd-1k':
        args.dataset = 'IRSTD-1k'
    return args


class Val:
    def __init__(self, args, load_path: str):
        self.args = args

        # --- 1. 数据准备 ---
        if args.dataset == 'sirst':
            self.val_set = SirstDataset(args, mode='val')
        elif args.dataset == 'IRSTD-1k':
            self.val_set = IRSTD1K_Dataset(args, mode='val')
        else:
            raise NameError("不支持的数据集类型")

        # 使用 DataLoader 加载验证集，负责多线程读取数据
        self.val_data_loader = Data.DataLoader(self.val_set, batch_size=args.batch_size)

        # --- 2. 模型初始化 ---
        assert args.mode in ['L', 'M', 'S']
        if args.mode == 'L':
            self.net = CEPNet_L(ablation=args.ablation)
        elif args.mode == 'M':
            self.net = CEPNet_M(ablation=args.ablation)
        elif args.mode == 'S':
            self.net = CEPNet(ablation=args.ablation)

        # 加载训练好的权重文件 (.pth)
        # 加上 weights_only=False，显式允许加载非权重对象
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        # 提取模型参数状态（注意：这里假设权重保存时使用了 {'net': state_dict} 的格式）
        self.net.load_state_dict(checkpoint['net'])

        # 将模型移动到 GPU 或 CPU 上
        self.net.to(device)

        # --- 3. 评估器初始化 ---
        self.iou_metric = IoUMetric()  # 计算常规像素级 IoU
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)  # 计算归一化 nIoU
        self.PD_FA = PD_FA(args.img_size)  # 计算 Pd (检测率) 和 Fa (虚警率)
        self.ROC = ROCMetric2(1, bins=10)  # 计算 ROC 曲线数据（TPR/FPR）

    def test_model(self):
        """执行完整的验证集测试流程"""
        # 重置所有评估指标的计数器
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        # 将模型切换到评估模式（关闭 Dropout, 固定 BatchNorm）
        self.net.eval()

        print(f"正在设备 {next(self.net.parameters()).device} 上运行测试...")

        # 使用 tqdm 显示进度条
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):
            # 将数据送入 GPU 并进行前向推理
            with torch.no_grad():  # 测试阶段不需要计算梯度，节省显存
                output = self.net(data.cuda())
                output = output.cpu()  # 将预测结果转回 CPU 准备计算指标

            # 更新各项指标计算器的数值
            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)

            # 维度转换以匹配 ROC 计算要求：[B, C, H, W] -> [H, W, C]
            output2 = output.squeeze(0).permute(1, 2, 0)
            labels2 = labels.squeeze(0)
            self.ROC.update(output2, labels2)

            # 实时从指标类中提取当前平均分，用于进度条展示
            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))
            # 获取 ROC 相关的 TPR, FPR, Recall, Precision
            ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()

            # 动态更新进度条描述信息，实时看到指标变化
            tbar.set_description('IoU:%f, nIoU:%f, Fa:%.10f, Pd:%.10f'
                                 % (IoU, nIoU, Fa, Pd))

        # 返回最终汇总的整个数据集的评分
        return IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate


if __name__ == "__main__":
    # 解析命令行输入的参数
    args = parse_args()

    # 实例化验证类
    value = Val(args, load_path=args.checkpoint)

    # 启动测试流程
    IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate = value.test_model()

    # 在控制台打印最终的汇总成绩
    print('\n' + '=' * 30)
    print('最终验证集评估结果:')
    print(f'IoU: {IoU}')
    print(f'nIoU: {nIoU}')
    print(f'Fa (虚警率): {Fa}')
    print(f'Pd (检测率): {Pd}')
    print(f'TPR (真阳性率): {ture_positive_rate}')
    print(f'FPR (假阳性率): {false_positive_rate}')
    print('=' * 30)
