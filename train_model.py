import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import parameters
from Dataset.DataSet_zhenzhongju import MyDataset
from models.model_zhenzhognju import Informer
from test_model_alltask import test_model
from utils.utils import dis, criterion, criterion_dis

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, default='informer',
                    help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, default='dizhen', help='data')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='dizhen.csv', help='data file')
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
# 以小时为单位重采样
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# 输入长度
parser.add_argument('--seq_len', type=int, default=800, help='input sequence length of Informer encoder')
# 预测时与标签重合的一部分
parser.add_argument('--label_len', type=int, default=799, help='start token length of Informer decoder')
# 预测未来的长度
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
# 输入序列的维度
parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=9, help='decoder input size')
# 预测多少个值
parser.add_argument('--c_out', type=int, default=1, help='output size')
# 隐藏层特征数
parser.add_argument('--d_model', type=int, default=800, help='dimension of model')
# 注意力头数
parser.add_argument('--n_heads', type=int, default=5, help='num of heads')

parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')

parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
# 计算注意力时对q采样的因字数
parser.add_argument('--factor', type=int, default=400, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
# 每层注意力之间要不要进行pooling操作
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=False)
parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
# 总体训练多少多少次，epoch为24 itr为2，则训练48次
parser.add_argument('--itr', type=int, default=20, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--train_epochs_final', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
# 连续多少次没更新就停止
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--l1_lambda', type=float, default='0.01', help='L1 regularization_loss ')



args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# 模型加载及初始化
model_dict = {
    'informer': Informer
}

device = torch.device('cpu')
if args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else args.devices
    device = torch.device('cuda:{}'.format(args.gpu))
    print('Use GPU: cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')
    print('Use CPU')

if args.model == 'informer' or args.model == 'informerstack':
    e_layers = args.e_layers if args.model == 'informer' else args.s_layers
    model = model_dict[args.model](
        # args.enc_in,
        # args.dec_in,
        800,
        800,
        args.c_out,
        9,
        args.label_len,
        args.pred_len,
        args.factor,
        args.d_model,
        args.n_heads,
        e_layers,  # self.args.e_layers,
        args.d_layers,
        args.d_ff,
        args.dropout,
        args.attn,
        args.embed,
        args.freq,
        args.activation,
        args.output_attention,
        args.distil,
        args.mix,
        device
    ).float()
if args.use_multi_gpu and args.use_gpu:
    model = nn.DataParallel(model, device_ids=args.device_ids)
model = model.to("cuda")
model_optim = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.1)


fw = open("models/model_0409_with200_x100_alltask_noarrive_zhenzhognju.txt", "a")

test_loader = torch.utils.data.DataLoader(
    MyDataset(dataPath=r'date-azi-test3.txt', args=args,
              position_encode=True, label_type='location'),
    batch_size=256, shuffle=False)# 训练集
print("修改学习率")
total_dis = 0.25

model=torch.load("models/model_aiz.pkl")
for epoch in range(1):

    train_loss_epicenter_dis = []
    train_loss_depth = []
    train_epicenter_dis_mean = []

    train_depth_mean = []
    train_dis_mean = []
    train_arrive_mean = []
    train_arrive_loss = []
    train_ml_mean = []
    train_ml_loss = []
    train_aiz_loss = []
    train_aiz_mean = []
    model.train()
    print("训练:第" + str(epoch) + "轮")
    fw.write("\n训练:第" + str(epoch) + "轮\n")

    for i, (batch_x, batch_y, batch_y_lat, batch_y_lng, epicenter_dis, epicenter_dis_norm,label_depth,label_depth_norm, label_ml, label_arrivetime, label_arrivetime_norm,
            receive_location, label_aiz, label_aiz_norm) in enumerate(train_loader):
        label_arrivetime_norm, epicenter_dis_norm= label_arrivetime_norm.to("cuda"), epicenter_dis_norm.to("cuda")
        label_aiz, label_aiz_norm = label_aiz.to("cuda"), label_aiz_norm.to("cuda")
        batch_x, batch_y, batch_y_lat, batch_y_lng, epicenter_dis, label_depth, label_ml, label_arrivetime, receive_location = batch_x.to(
            "cuda"), batch_y.to("cuda"), batch_y_lat.to("cuda"), batch_y_lng.to("cuda"), epicenter_dis.to("cuda"), label_depth.to("cuda"), label_ml.to("cuda"), label_arrivetime.to("cuda"), receive_location.to("cuda")

        model_optim.zero_grad()
        # dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
        outputs_arrive, outputs_epicenter_dis, outputs_aiz, outputs_depth, outputs_ml = model(batch_x.float(), batch_x.float(),
                                                                                 batch_y_lat.float(),
                                                                                 batch_y_lng.float(),
                                                                                 receive_location=receive_location,
                                                                                 receive_type=True)
        loss_arrivetime = criterion(outputs_arrive.float().squeeze(), label_arrivetime.float().squeeze(), "arrivetime", None)
        loss_ml = criterion(outputs_ml.float(), label_ml.float(), "arrivetime", None)
        loss_epicenter_dis = criterion(outputs_epicenter_dis.float().squeeze(), epicenter_dis.float().squeeze(), "arrivetime",
                                    None)
        loss_depth = criterion((outputs_depth).float().squeeze(), (label_depth).float().squeeze(), "arrivetime", None)
        loss_aiz = criterion((outputs_aiz).float().squeeze(), (label_aiz).float().squeeze(), "arrivetime", None)

        loss_total = (7*(loss_epicenter_dis+10*loss_aiz) + 3*(loss_arrivetime + loss_ml + loss_depth))
        loss_total.backward()
        model_optim.step()

        outputs_depth = torch.exp(outputs_depth)
        label_depth = torch.exp(label_depth)
        outputs_ml = outputs_ml*1.6+2.2
        label_ml = label_ml*1.6+2.2

        outputs_arrive = outputs_arrive*label_arrivetime_norm[:,0].unsqueeze(1) + label_arrivetime_norm[:,1].unsqueeze(1)
        label_arrivetime = label_arrivetime*label_arrivetime_norm[:,0].unsqueeze(1) + label_arrivetime_norm[:,1].unsqueeze(1)
        epicenter_dis = epicenter_dis*epicenter_dis_norm[:,0].unsqueeze(1)+epicenter_dis_norm[:,1].unsqueeze(1)
        outputs_epicenter_dis = outputs_epicenter_dis*epicenter_dis_norm[:,0].unsqueeze(1)+epicenter_dis_norm[:,1].unsqueeze(1)
        label_aiz = label_aiz * label_aiz_norm[:, 0].unsqueeze(1) + label_aiz_norm[:, 1].unsqueeze(1)
        outputs_aiz = outputs_aiz * label_aiz_norm[:, 0].unsqueeze(1) + label_aiz_norm[:, 1].unsqueeze(1)

        outputs_arrive[:,0] = 0

        train_epicenter_dis_mean.append(torch.mean(torch.abs(((outputs_epicenter_dis)).squeeze() - ((epicenter_dis)).squeeze())).item())
        train_depth_mean.append(torch.mean(torch.abs(((outputs_depth)).squeeze() - ((label_depth)).squeeze())).item())
        train_arrive_mean.append(torch.mean(torch.abs((outputs_arrive).squeeze() - (label_arrivetime).squeeze())).item())
        train_ml_mean.append(torch.mean(torch.abs((outputs_ml) - (label_ml))).item())
        train_aiz_mean.append(torch.mean(torch.abs((outputs_aiz) - (label_aiz))).item())
        train_loss_epicenter_dis.append((loss_epicenter_dis).item())
        train_loss_depth.append((loss_depth).item())
        train_ml_loss.append((loss_ml).item())
        train_aiz_loss.append((loss_aiz).item())
        train_arrive_loss.append(loss_arrivetime.item())

        if (i + 1) % 10 == 0:
            # 计算真实震级和预测震级
            print("\titers: {0}, epoch: {1} | loss_epicenter_dis: {2:.5f} | loss_aiz: {10:.5f} | loss_ml: {3:.5f} | loss_depth: {4:.5f} | loss_arrivetime: {5:.5f} | ml_mean: {6:.3f}| depth_mean: {7:.3f} | arrivetime_mean: {8:.3f} | dis_mean: {9:.3f} | aiz_mean: {11:.3f}".format(i + 1, epoch + 1,sum(train_loss_epicenter_dis) / len(train_loss_epicenter_dis), sum(train_ml_loss) / len(train_ml_loss), sum(train_loss_depth) / len(train_loss_depth),sum(train_arrive_loss) / len(train_arrive_loss),sum(train_ml_mean) / len(train_ml_mean), sum(train_depth_mean) / len(train_depth_mean), sum(train_arrive_mean) / len(train_arrive_mean), sum(train_epicenter_dis_mean) / len(train_epicenter_dis_mean), sum(train_aiz_loss) / len(train_aiz_loss), sum(train_aiz_mean) / len(train_aiz_mean)))

            #print("\tdis:"+str(sum(train_dis_r2)/len(train_dis_r2))+"  depth:"+str(sum(train_depth_r2)/len(train_depth_r2))+"  arrive:"+str(sum(train_arrive_r2)/len(train_arrive_r2))+"  ml:"+str(sum(train_ml_r2)/len(train_ml_r2)))
            fw.write("\titers: {0}, epoch: {1} | loss_epicenter_dis: {2:.5f} | loss_aiz: {10:.5f} | loss_ml: {3:.5f} | loss_depth: {4:.5f} | loss_arrivetime: {5:.5f} | ml_mean: {6:.3f}| depth_mean: {7:.3f} | arrivetime_mean: {8:.3f} | dis_mean: {9:.3f} | aiz_mean: {11:.3f}".format(i + 1, epoch + 1,sum(train_loss_epicenter_dis) / len(train_loss_epicenter_dis), sum(train_ml_loss) / len(train_ml_loss), sum(train_loss_depth) / len(train_loss_depth),sum(train_arrive_loss) / len(train_arrive_loss),sum(train_ml_mean) / len(train_ml_mean), sum(train_depth_mean) / len(train_depth_mean), sum(train_arrive_mean) / len(train_arrive_mean), sum(train_epicenter_dis_mean) / len(train_epicenter_dis_mean), sum(train_aiz_loss) / len(train_aiz_loss), sum(train_aiz_mean) / len(train_aiz_mean)))

    torch.save(model,"models/model_aiz.pkl")
    if (epoch + 1) % 1 == 0:
        total_dis = test_model(model, test_loader, epoch, total_dis)
