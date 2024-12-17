import os
import matplotlib.pyplot as plt
import json

with open('/media/ubuntu/Seagate Basic/work_dirs/09-07/20230915_155646.log.json','r') as file:
    # json_info=json.loads(file)
    json_info=file.readlines()

acc_seg,loss_seg,loss=[],[],[]
iters=[]

for line in json_info[1:]:
    dict_line=eval(line.replace('\n',''))
    if dict_line['decode.acc_seg']<90:
        continue
    acc_seg.append(dict_line['decode.acc_seg'])
    loss_seg.append(dict_line['decode.loss_seg'])
    loss.append(dict_line['loss'])

    iters.append(dict_line['iter'])

fig,ax=plt.subplots(1,3)
ax[0].plot(iters,acc_seg)
ax[0].set_title('Acc Seg')

ax[1].plot(iters,loss_seg)
ax[1].set_title('Loss Seg')

ax[2].plot(iters,loss)
ax[2].set_title('Loss')


plt.savefig('loss.png')