import argparse
import os
import utils
import paddle
from paddle.io import DataLoader
import dataset

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--load_path', type=str, default='./archive/no_train_eval_WGAN/models/deepfillv2_LSGAN_epoch40_batchsize5', help='path of model weights')
    parser.add_argument('--test_path', type=str, default='./test', help='training samples path that is a folder')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type=str, default="0", help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--load_name', type=str, default='', help='load model name')
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--in_channels', type=int, default=4, help='input RGB image + 1 channel mask')
    parser.add_argument('--out_channels', type=int, default=3, help='output RGB image')
    parser.add_argument('--latent_channels', type=int, default=64, help='latent channels')
    parser.add_argument('--pad_type', type=str, default='zero', help='the padding type')
    parser.add_argument('--activation', type=str, default='lrelu', help='the activation type')
    parser.add_argument('--norm', type=str, default='in', help='normalization type')
    parser.add_argument('--init_type', type=str, default='xavier', help='the initialization type')
    parser.add_argument('--init_gain', type=float, default=0.02, help='the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type=str, default=r"", help='the training folder')
    parser.add_argument('--maskroot', type=str, default=r"", help='the training folder')
    parser.add_argument('--mask_type', type=str, default='free_form', help='mask type')
    parser.add_argument('--imgsize', type=int, default=256, help='size of image')
    parser.add_argument('--margin', type=int, default=10, help='margin of image')
    parser.add_argument('--mask_num', type=int, default=15, help='number of mask')
    parser.add_argument('--bbox_shape', type=int, default=30, help='margin of image for bbox mask')
    parser.add_argument('--max_angle', type=int, default=4, help='parameter of angle for free form mask')
    parser.add_argument('--max_len', type=int, default=40, help='parameter of length for free form mask')
    parser.add_argument('--max_width', type=int, default=10, help='parameter of width for free form mask')
    opt = parser.parse_args()
    print(opt)

    '''
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------

    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # configurations
    load_folder = os.path.dirname(opt.load_path)
    test_folder = opt.test_path
    if not os.path.exists(load_folder):
        os.makedirs(load_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Enter main function
    generator = utils.create_generator(opt)
    state_dict = paddle.load(opt.load_path)
    generator.set_state_dict(state_dict)
    testset = dataset.ValidationSet_with_Known_Mask(opt)  # img,mask,image_name
    dataloader = DataLoader(testset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    generator.eval()
    for batch_idx, (img, mask, _) in enumerate(dataloader):
        img = paddle.to_tensor(img)
        mask = paddle.to_tensor(mask)

        # Generator output
        first_out, second_out = generator(img, mask)

        # forward propagation
        second_out_wholeimg = img * (1 - mask) + second_out * mask  # in range [0, 1]

        ### Sample data
        masked_img = img * (1 - mask) + mask
        mask = paddle.concat((mask, mask, mask), 1)
        img_list = [img, mask, masked_img, first_out, second_out, second_out_wholeimg]
        name_list = ['raw', 'mask', 'masked_img', 'first_out', 'second_out','second_out_wholeimg']
        utils.save_sample_png(sample_folder=test_folder, sample_name='index%d' % (batch_idx + 1), img_list=img_list,
                              name_list=name_list, pixel_max_cnt=255)