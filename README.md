# deepfill_v2-pp
implement deepfill version 2 with paddlepaddle framework

## pre-requisites
1. python3.6 above
2. paddlepaddle 2.0
3. opencv-python

## How to use it
Download pretrained weights of generator from following link()

    链接：https://pan.baidu.com/s/1a352rmZaEaNk8PMaZObf0Q 
    提取码：sidy 

- train model

    >python train.py [--baseroot {path to image folder}] [--epoch {total epoch}]
    
- resume training

    >python train.py --resume_epoch {num} [--baseroot {path to image folder}] [--epoch {total epoch}]


- evaluate model

    >python eval.py [--load_path {path to generator state_dict}] [--baseroot {path to image folder} ]


- inpaint images

    >python inpaint.py [--load_path {path to generator state_dict}] [--baseroot {path to image folder}] [--maskroot {mask to image folder}] 

## Todo
- well trained weight for datasets(need huge GPU resource for trainning)
- sketch
