# deepfill_v2-pp
implement deepfill version 2 with paddlepaddle framework

## pre-requisites
1. python3.6 above
2. paddlepaddle 2.0
3. opencv-python

## How to use it
- train model

    >python train.py
    
    or 
    >python train.py --baseroot {path to image folder} 


- evaluate model

    >python eval.py --load_path {path to generator state_dict} --baseroot {path to image folder} 


- inpaint images

    >python inpaint.py --load_path {path to generator state_dict} --baseroot {path to image folder} --maskroot {mask to image folder} 

