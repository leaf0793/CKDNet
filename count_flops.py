import torch

import torch.nn as nn

class Loss_warper(nn.Module):
    def __init__(self, cd):
        super().__init__()
        all_conv = []
        for _ in cd:
            C_in, C_out = _
            if C_in != C_out:
                all_conv.append(nn.Conv2d(C_in, C_out, 1))
            else:
                all_conv.append(nn.Identity())
        print(all_conv)
        # all_conv = [
        #     nn.Conv2d(_[0], _[1]) for _ in cd
        # ]
        

def test():
    model1 = PSMnet(192, percentage=1).cuda(0)
    model2 = PSMnet(192, percentage=0.5).cuda(0)
    model1_list = []
    model2_list = []
    for m in model1.modules():
        if isinstance(m, torch.nn.Conv2d):
            model1_list.append(m)
    for m in model2.modules():
        if isinstance(m, torch.nn.Conv2d):
            model2_list.append(m)
    if len(model1_list) == len(model2_list):
        print(1)



def test2():
    from models.student2.stackhourglass import PSMNet_KD as PSMnet
    input = torch.FloatTensor(1, 3, 480, 640).fill_(1.0).cuda(0)
    # import 

    model = PSMnet(192, percentage=0.5).cuda(0)
    model2 = PSMnet(192, percentage=0.3).cuda(0)
    res = model(input,input)
    res2 = model2(input,input)
    kd_res = res[2]
    kd_res2 = res2[2]
    cd = []
    for i in range(len(kd_res)):
        cd.append([kd_res[i].shape[1], kd_res2[i].shape[1]])
    Loss = Loss_warper(cd)
    print(1)
    # for m in model.modules():


if __name__ == '__main__':
    # test2()
    from models.student1.stackhourglass import PSMNet_KD as PSMnet
    # from models.student_short.stackhourglass import PSMNet_KD as PSMnet
    input = torch.FloatTensor(1, 3, 480, 640).fill_(1.0).cuda(0)
    # import 
    # for per in [1]:
    for per in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
        model = PSMnet(192, percentage=per).cuda(0)
        # for m in model.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         print(m)
        model.eval()
        with torch.no_grad():
            from thop import clever_format, profile

            flops, params = profile(model, inputs=(
                input, input), verbose=False)
            flops, params = clever_format([flops, params], "%.3f")
            print(f"per:{per}, {flops} {params}")