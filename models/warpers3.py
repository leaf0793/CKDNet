import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.warpers import Loss_warper, KD_warper2


def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)   #C和w*h转置
    gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t相乘
    return gram

class KD_warper4(KD_warper2):
    def __init__(self, teacher, student, maxdisp=192, KDlossOnly=True):
        super().__init__(maxdisp=maxdisp, teacher=teacher, student=student)
        # self.T_model = teacher
        # self.model = student

        self.KDlossOnly = KDlossOnly
        if self.KDlossOnly:
            print("KDlossOnly is True")
        self.Temperature = 1
        self.weight_loss = 1
        # self.Layers = self.construct_convs()
        # self.Layers = [_.cuda() for _ in self.Layers]
        # print(self.Layers)

        for param in self.T_model.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = True
            
            
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
        KL_S_1 = KL_S[0]
        loss = 0
        if 1:
            loss += F.l1_loss(
                KL_S_1, KL_T_1,  reduction='mean')
            # loss += F.l1_loss(
            #     KL_S_2, KL_T_2,  reduction='mean')
            # loss += F.l1_loss(
            #     KL_S_3, KL_T_3,  reduction='mean')
        else:
            eps = 1e-16
            loss += F.kl_div(
                torch.log(KL_S_1+eps), KL_T_1+eps, reduction='batchmean')
            # loss += F.kl_div(
            #     torch.log(KL_S_2+eps), KL_T_2+eps, reduction='batchmean')
            # loss += F.kl_div(
            #     torch.log(KL_S_3+eps), KL_T_3+eps, reduction='batchmean')
            
        '''
        kl_div = y * log(y_p/y)  
         y * log(y_p/y)  == (y+eps) * log((y_p+eps)/(y+eps))
        '''

        return loss
        

    
    def forward(self, L, R, gt):
        self.T_model.eval()
        mask = (gt > 0) & (gt < self.maxdisp)
        
        L, R, bot_pad, right_pad = self.pad_img(L, R)
        KL_S, S_output = self.model(L, R, self.Temperature)
        S_output = self.unpad_img(S_output, bot_pad, right_pad)

        if self.model.training:
            if self.KDlossOnly:
                loss_disp = 0
            else:
                loss_disp = self.loss_disp(S_output, gt, mask)
            with torch.no_grad():
                KL_T, T_output = self.T_model(L, R, self.Temperature)
            loss_kd = self.Knowledge_distill(KL_T, KL_S)
            # loss_kd += self.Knowledge_distill_feature(KD_T, KD_S, self.Temperature)
            return loss_disp, 0, loss_kd
        else:
            gt=torch.squeeze(gt,dim=1)
            mask=torch.squeeze(mask,dim=1)
            loss_disp = [_(S_output[0], gt, mask)
                         for _ in self.eval_losses]
            return loss_disp, 0
     
    # def Knowledge_distill_feature(self, KD_T, KD_S, Temperature):
    #     # distilled_T = []
    #     # loss = [0.001]*12
    #     loss = []
    #     for i in range(len(KD_T)-3):
    #         # alpha = 1
    #         distilled_T = self.Layers[i](KD_T[i])
    #         loss.append(
    #             F.l1_loss(KD_S[i]/Temperature, distilled_T/Temperature)
    #         )
    #     loss = torch.stack(loss, dim=0)
    #     return torch.mean(loss)
