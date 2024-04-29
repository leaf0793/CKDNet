# CKDNet

Submission for the submission 1592: CKDNet: a cascaded knowledge distillation framework for lightweight stereo matching

## To Train expert mode:
python train.py

## Knowledge distillation from Expert to teacher:
python train_KD_expert2teacher.py --slim_pecentage 0.5

## Knowledge distillation from Expert to teacher:
python train_KD_teacher2student.py --teacher_slim_pecentage 0.5 --slim_pecentage 0.3