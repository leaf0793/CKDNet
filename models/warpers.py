import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# from models.submodule import disparity_regression


class Loss_warper(nn.Module):
    def __init__(self, model=None, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.model = model
        # self.right_pad = -1
        # self.bot_pad = -1
        self.T_model = None
        self.eval_losses = [
            self.eval_epe,
            self.D1_metric,
        ]

    def forward(self, L, R, gt):
        L, R, bot_pad, right_pad = self.pad_img(L, R)
        output = self.model(L, R)
        output = self.unpad_img(output, bot_pad, right_pad)
        mask = (gt > 0) & (gt < self.maxdisp)

        if self.training:
            loss_disp = self.loss_disp(output, gt, mask)
        else:
            loss_disp = [_(output[0], gt, mask) for _ in self.eval_losses]

        return loss_disp, 0
    

    def pad_img(self, L, R, base=32):
        if self.training is True:
            # if self.T_model:
            #     self.T_model.Regression.set_full_shape((L.shape[2], L.shape[3]))
            # self.model.Regression.set_full_shape((L.shape[2], L.shape[3]))
            return L, R, 0, 0
        else:
            bot_pad = int(
                base-L.shape[2] % base) if int(base-L.shape[2] % base) != base else 0
            right_pad = int(
                base-L.shape[3] % base) if int(base-L.shape[3] % base) != base else 0
            # self.model.Regression.set_full_shape(
            #     (L.shape[2]+bot_pad, L.shape[3]+right_pad))
            L = F.pad(
                L, (0, right_pad, 0, bot_pad),
                "constant", 0
            )
            R = F.pad(
                R, (0, right_pad, 0, bot_pad,),
                "constant", 0
            )
            return L, R, bot_pad, right_pad

    def unpad_img(self, output, bot_pad, right_pad):
        if self.training is True:
            return output
        else:
            # output = list(output)
            for i in range(len(output)):
                if output[i] is not None:
                    output[i] = output[i][:, :-bot_pad,:] if bot_pad > 0 else output[i]
                    output[i] = output[i][:, :, :-
                                          right_pad] if right_pad > 0 else output[i]
            return output

    def eval_epe(self, preds, gt, mask):

        loss = F.l1_loss(preds[mask], gt[mask], reduction='mean')
        return loss

    def loss_disp(self, preds, gt, mask):
        mask = torch.squeeze(mask, dim=1)
        gt = torch.squeeze(gt, dim=1)

        if isinstance(preds, list) or isinstance(preds, tuple):
            weights = [0.5, 0.7, 1.0]
            loss1 = []
            for weights, output in zip(weights, preds):
                output = torch.squeeze(output, dim=1)
                loss1.append(
                    weights*(
                        F.smooth_l1_loss(
                            output[mask], gt[mask], reduction='mean')
                    )
                )
            loss = sum(loss1)
        else:
            loss = F.smooth_l1_loss(preds[mask], gt[mask], reduction='mean')
        return loss
    
    def D1_metric(self, D_es, D_gt, mask):
        D_es, D_gt = D_es[mask], D_gt[mask]
        E = torch.abs(D_gt - D_es)
        err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
        return torch.mean(err_mask.float())

    def losses_head(self, out, gt, mask):
        return 0
        out = torch.unsqueeze(out, dim=1)
        out = F.interpolate(
            out, (self.maxdisp, gt.shape[-2], gt.shape[-1]), mode='trilinear', align_corners=False)  # up to 1
        out = torch.squeeze(out, dim=1)
        #--------loss2--------
        out2 = disparity_regression(out, self.maxdisp)
        # return F.smooth_l1_loss(
        #         out2[mask], gt[mask], reduction='mean')
        # out2 = torch.unsqueeze(out2, dim=1)
        alarm = torch.abs(out2 - gt)
        mmask = (alarm >= 1.2).logical_and(mask)
        if(len(mmask[mmask == True])):
            loss2 = F.smooth_l1_loss(
                out2[mmask], gt[mmask], reduction='mean')
        else:
            loss2 = 0
        gt = gt//1
        # gt = torch.div(gt, 1, rouding_mode='trunc')
        gt = gt.type('torch.cuda.LongTensor')
        gt[~mask] = -1

        # loss = F.nll_loss(F.log_softmax(out,dim=1),gt.squeeze(1), ignore_index=-1)
        loss = F.nll_loss(out,gt.squeeze(1), ignore_index=-1)
        # loss_temp = F.nll_loss(F.log_softmax(out,dim=1), gt.squeeze(1), ignore_index=-1)
        return loss+loss2


class KD_warper(Loss_warper):
    def __init__(self, teacher, student, maxdisp=192, naive=False,KDlossOnly=False):
        super().__init__(maxdisp=maxdisp)
        self.T_model = teacher
        self.model = student
        self.naive = naive
        self.KDlossOnly = KDlossOnly
        self.Temperature = 0.5
        self.weight_loss = 1
        if naive:
            self.ah1 = nn.Sequential(
                nn.ReLU6(True),
                nn.Conv2d(32, 16, 1, 1, 0),
            )
            self.ah2 = nn.Sequential(
                nn.ReLU6(True),
                nn.Conv2d(64, 16, 1, 1, 0),
            )
            self.ah3 = nn.Sequential(
                nn.ReLU6(True),
                nn.Conv2d(128, 32, 1, 1, 0),
            )
            self.ah4 = nn.Sequential(
                nn.ReLU6(True),
                nn.Conv2d(128, 32, 1, 1, 0),
            )
        for param in self.T_model.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = True


    def forward(self, L, R, gt):
        self.T_model.eval()
        
        L, R, bot_pad, right_pad = self.pad_img(L, R)
        KL_S, S_output = self.model(L, R)
        S_output = self.unpad_img(S_output, bot_pad, right_pad)

        mask = (gt > 0) & (gt < self.maxdisp)

        if self.model.training:
            if self.KDlossOnly:
                loss_disp = 0
            else:
                loss_disp = self.loss_disp(S_output, gt, mask)

        else:
            loss_disp = [_(S_output[0], gt, mask)
                         for _ in self.eval_losses]
        if self.model.training:
            # loss_kd=0
            with torch.no_grad():
                KL_T, T_output = self.T_model(L, R)
            loss_kd = self.Knowledge_distill(KL_T, KL_S)
            # return self.weight_loss*loss_disp, 0, (1-self.weight_loss)*loss_kd
            return loss_disp, 0, loss_kd
        else:
            # loss_kd = 0
            return loss_disp, 0


    def Knowledge_distill(self, KL_T, KL_S):
        # return 0
        """
        #* KL_T contains:
        (
            (feature_L:(x,l2,l3,l4), feature_R:(x,l2,l3,l4)),
            (Volume),
            (head),
            (out3, pred3)
        )
        #* KL_S contains:
        (
            (feature_L:(x,l2), feature_R:(x,l2)),
            (Volume),
            (head),
            (out3, pred3)
        )
        """
        KL_T_1, KL_T_2, KL_T_3, KL_T_4 = KL_T
        KL_S_1, KL_S_2, KL_S_3, KL_S_4 = KL_S
        loss = 0
        # feature_T_L, feature_T_R = KL_T_1
        # ! ---------------------------------------------------
        # KL_T = (x, l2, l3, l4)
        # KL_S = (x, l2)
        if self.naive:
            for i in range(2):
                KL_T_1_new = [
                    self.ah1(KL_T_1[i][0]),
                    self.ah2(KL_T_1[i][1]),
                    self.ah3(KL_T_1[i][2]),
                    self.ah4(KL_T_1[i][3])
                ]
                KL_S_1_new = [
                    KL_S_1[i][0],
                    KL_S_1[i][1][:, :16, :, :],
                    KL_S_1[i][1][:, 16:48, :, :],
                    KL_S_1[i][1][:, 48:, :, :]
                ]

                for feature_L_T, feature_L_S in zip(KL_T_1_new, KL_S_1_new):
                    # F.avg_pool2d()
                    loss += F.smooth_l1_loss(
                        feature_L_T, feature_L_S, reduction='mean')
        else:
            # for feature_L_T, feature_L_S in zip(KL_T_1[0], KL_S_1[0]):
            #     # loss += F.smooth_l1_loss(
            #     #     feature_L_T, feature_L_S, reduction='mean')
            #     loss += F.mse_loss(
            #         feature_L_S, feature_L_T, reduction='mean')
            # for feature_R_T, feature_R_S in zip(KL_T_1[1], KL_S_1[1]):
            #     # loss += F.smooth_l1_loss(
            #     #     feature_R_T, feature_R_S, reduction='mean')
            #     loss += F.mse_loss(
            #         feature_R_S, feature_R_T, reduction='mean')
            None
        # ----------------------------------------------------
        # ! ---------------------------------------------------
        # if self.Temperature is not None:
        #     KL_S_4 = list(KL_S_4)
        #     KL_T_4 = list(KL_T_4)
        #     KL_S_3 /= self.Temperature
        #     KL_T_3 /= self.Temperature
        #     KL_S_4[1] /= self.Temperature
        #     KL_T_4[1] /= self.Temperature
        loss += F.mse_loss(
            KL_S_2, KL_T_2,  reduction='mean')

        # loss += F.kl_div(
        #     F.log_softmax(KL_S_3, dim=1), F.log_softmax(KL_T_3, dim=1), reduction='batchmean', log_target=True)
        loss += F.mse_loss(
            KL_S_3, KL_T_3, reduction='mean')

        loss += F.mse_loss(
            KL_S_4[0], KL_T_4[0],  reduction='mean')
        # loss += F.kl_div(F.log_softmax(KL_S_4[1], dim=1),
        #                  F.log_softmax(KL_T_4[1], dim=1), reduction='batchmean', log_target=True)
        # loss += F.mse_loss(KL_S_4[1], KL_T_4[1], reduction='mean')

        # if self.Temperature is not None:
        #     loss *= self.Temperature*self.Temperature
        return loss
        
def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)   #C和w*h转置
    gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t相乘
    return gram

class KD_warper2(Loss_warper):
    def __init__(self, teacher, student, maxdisp=192, naive=False,KDlossOnly=False):
        super().__init__(maxdisp=maxdisp)
        self.T_model = teacher
        self.model = student
        self.naive = naive
        self.KDlossOnly = KDlossOnly
        if self.KDlossOnly:
            print("KDlossOnly is True")
        self.Temperature = 1
        self.weight_loss = 1
        # if naive:
        #     self.ah1 = nn.Sequential(
        #         nn.ReLU6(True),
        #         nn.Conv2d(32, 16, 1, 1, 0),
        #     )
        #     self.ah2 = nn.Sequential(
        #         nn.ReLU6(True),
        #         nn.Conv2d(64, 16, 1, 1, 0),
        #     )
        #     self.ah3 = nn.Sequential(
        #         nn.ReLU6(True),
        #         nn.Conv2d(128, 32, 1, 1, 0),
        #     )
        #     self.ah4 = nn.Sequential(
        #         nn.ReLU6(True),
        #         nn.Conv2d(128, 32, 1, 1, 0),
        #     )
        for param in self.T_model.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = True


    def forward(self, L, R, gt):
        self.T_model.eval()
        mask = (gt > 0) & (gt < self.maxdisp)
        
        L, R, bot_pad, right_pad = self.pad_img(L, R)
        KL_S, S_output = self.model(L, R, self.Temperature)
        if self.KDlossOnly is False or self.training is False:
            S_output = self.unpad_img(S_output, bot_pad, right_pad)
        # if 
            

        # if self.model.training:
        #     if self.KDlossOnly:
        #         loss_disp = 0
        #     else:
        #         loss_disp = self.loss_disp(S_output, gt, mask)
        # else:
        #     loss_disp = [_(S_output[0], gt, mask)
        #                  for _ in self.eval_losses]

        if self.model.training:
            loss_disp = self.loss_disp(S_output, gt, mask) if not self.KDlossOnly else 0
            with torch.no_grad():
                KL_T, _ = self.T_model(L, R, self.Temperature)
            loss_kd = self.Knowledge_distill(KL_T, KL_S)
            
            return loss_disp, 0, loss_kd
        else:
            gt=torch.squeeze(gt,dim=1)
            mask=torch.squeeze(mask,dim=1)
            loss_disp = [_(S_output[0], gt, mask)
                         for _ in self.eval_losses]
            return loss_disp, 0


    def Knowledge_distill(self, KL_T, KL_S):
        # return 0
        """
        #* KL_T contains:
        (
            (feature_L:(x,l2,l3,l4), feature_R:(x,l2,l3,l4)),
            (Volume),
            (head),
            (out3, pred3)
        )
        #* KL_S contains:
        (
            (feature_L:(x,l2), feature_R:(x,l2)),
            (Volume),
            (head),
            (out3, pred3)
        )
        """
        KL_T_1, KL_T_2, KL_T_3 = KL_T
        KL_S_1, KL_S_2, KL_S_3 = KL_S
        loss = 0
        if 1:
            loss += F.l1_loss(
                KL_S_1, KL_T_1,  reduction='mean')
            loss += F.l1_loss(
                KL_S_2, KL_T_2,  reduction='mean')
            loss += F.l1_loss(
                KL_S_3, KL_T_3,  reduction='mean')
        else:
            eps = 1e-16
            loss += F.kl_div(
                torch.log(KL_S_1+eps), KL_T_1+eps, reduction='batchmean')
            loss += F.kl_div(
                torch.log(KL_S_2+eps), KL_T_2+eps, reduction='batchmean')
            loss += F.kl_div(
                torch.log(KL_S_3+eps), KL_T_3+eps, reduction='batchmean')
            
        '''
        kl_div = y * log(y_p/y)  
         y * log(y_p/y)  == (y+eps) * log((y_p+eps)/(y+eps))
        '''
        
        # 
        # feature_T_L, feature_T_R = KL_T_1
        # ! ---------------------------------------------------
        # KL_T = (x, l2, l3, l4)
        # KL_S = (x, l2)

        # ----------------------------------------------------
        # ! ---------------------------------------------------
        # if self.Temperature is not None:
        #     KL_S_4 = list(KL_S_4)
        #     KL_T_4 = list(KL_T_4)
        #     KL_S_3 /= self.Temperature
        #     KL_T_3 /= self.Temperature
        #     KL_S_4[1] /= self.Temperature
        #     KL_T_4[1] /= self.Temperature
        # loss += F.mse_loss(
        #     KL_S_2, KL_T_2,  reduction='mean')
        # loss += F.kl_div(
        #     F.log_softmax(KL_S_3, dim=1), F.log_softmax(KL_T_3, dim=1), reduction='batchmean', log_target=True)
        # loss += F.mse_loss(
        #     KL_S_3, KL_T_3, reduction='mean')
        
        # gram_matrix_T = gram_matrix(KL_T_4[1]).detach_()
        # gram_matrix_S = gram_matrix(KL_S_4[1])
        # loss += F.mse_loss(gram_matrix_S, gram_matrix_T)
        
        # KL_S = F.softmax(KL_S_4[1] / self.Temperature, dim=1)
        # KL_T = F.softmax(KL_T_4[1] / self.Temperature, dim=1)

        # loss += F.mse_loss(
        #     KL_S, KL_T,  reduction='mean')
        # return input
        
        # loss += F.kl_div(F.log_softmax(KL_S_4[1], dim=1),
        #                  F.log_softmax(KL_T_4[1], dim=1), reduction='batchmean', log_target=True)
        # loss += F.mse_loss(KL_S_4[1], KL_T_4[1], reduction='mean')

        # if self.Temperature is not None:
        #     loss *= self.Temperature*self.Temperature
        return loss
        
