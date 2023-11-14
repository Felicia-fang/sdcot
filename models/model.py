""" Util for model initialization

Author: Zhao Na
Date: September, 2020
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from ap_helper import parse_prediction_to_pseudo_bboxes, parse_groundtruths,parse_prediction_dynamic_to_pseudo_bboxes


def init_detection_model(args, dataset_config, model_name1=None):
    model = create_detection_model(args, dataset_config)
    if args.model_checkpoint_path is not None and model_name1 == 'static':
        model = load_detection_model(model, args.model_checkpoint_path)
    elif args.model_checkpoint_path is not None and model_name1 == 'dynamic':
        if os.path.isfile(args.model_dynamic_checkpoint_path):
            model = load_detection_model(model, args.model_dynamic_checkpoint_path, model_name='teacher')
        else:
            model = load_detection_model(model, args.model_checkpoint_path)
    else:
        raise ValueError('Detection model checkpoint path must be given!')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device


def generate_pseudo_bboxes(base_detector, dynamic_detector, device, pseudo_config_dict, point_cloud,instance_bboxes,plist):
    ''' generate pseudo bounding boxes w.r.t. base classes
    Args:
        base_detector: nn.Module, model
        pseudo_config_dict: dict
        point_cloud: numpy array, shape (num_point, pc_attri_dim)
    Returns:
        pseudo_bboxes: numpy array, shape (num_valid_detections, 8)
    '''
    point_cloud_tensor = torch.from_numpy(point_cloud.astype(np.float32)).to(device).unsqueeze(0)
    with torch.no_grad():
        end_points = base_detector(point_cloud_tensor)
    pseudo_bboxes = parse_prediction_to_pseudo_bboxes(end_points, pseudo_config_dict, point_cloud)

    point_cloud_tensor = torch.from_numpy(point_cloud.astype(np.float32)).to(device).unsqueeze(0)
    with torch.no_grad():
        end_points = dynamic_detector(point_cloud_tensor)    
    pseudo_bboxes_dynamic = parse_prediction_dynamic_to_pseudo_bboxes(end_points, pseudo_config_dict, point_cloud,plist)
    # print("instence",instance_bboxes)
    #for novel class in gt_bboxes, compare it with pseudo_bboxes_dynamic, 
    # print("pseudo_bboxes_dynamic",pseudo_bboxes_dynamic)

    if pseudo_bboxes is None or pseudo_bboxes_dynamic is None:
        return pseudo_bboxes
    gt_bbox_novel = []
    for ins in instance_bboxes:
        x=ins[0]
        y=ins[1]
        z=ins[2]
        l=ins[3]
        w=ins[4]
        h=ins[5]
        cla=-1
        heading=0
        probility=1
        gt_bbox_novel.append([x,y,z,l,w,h,heading,cla,probility])
    
    combined_boxes = np.concatenate((pseudo_bboxes, pseudo_bboxes_dynamic,gt_bbox_novel), axis=0)

    # combined_boxes = np.concatenate((pseudo_bboxes, pseudo_bboxes_dynamic), axis=0)
    x1 = combined_boxes[:, 0]
    y1 = combined_boxes[:, 1]
    z1 = combined_boxes[:, 2]
    x2 = combined_boxes[:, 0] + combined_boxes[:, 3]  # x2 = x1 + l
    y2 = combined_boxes[:, 1] + combined_boxes[:, 4]  # y2 = y1 + w
    z2 = combined_boxes[:, 2] + combined_boxes[:, 5]  # z2 = z1 + h
    heading = combined_boxes[:, 6]
    score = combined_boxes[:, 8]  # confidence score
    cls = combined_boxes[:, 7]  # class
    # add logits

    # each row is a bounding box: x1, y1, z1, x2, y2, z2, score, cls
    boxes = np.column_stack((x1, y1, z1, x2, y2, z2, heading, score, cls))
    # print("boxes",boxes)

    # do nms
    overlap_threshold = 0.5 
    picked_indices = nms_3d_faster_samecls(boxes, overlap_threshold, old_type=False)

    # pick out the selected boxes
    selected_boxes = boxes[picked_indices]

    # convert to original format
    x1_selected = selected_boxes[:, 0]
    y1_selected = selected_boxes[:, 1]
    z1_selected = selected_boxes[:, 2]
    l_selected = selected_boxes[:, 3] - selected_boxes[:, 0]
    w_selected = selected_boxes[:, 4] - selected_boxes[:, 1]
    h_selected = selected_boxes[:, 5] - selected_boxes[:, 2]
    heading_selected = selected_boxes[:, 6]
    class_selected = selected_boxes[:, 8]

    # concatenate the selected boxes
    original_format_boxes = np.column_stack((x1_selected, y1_selected, z1_selected, l_selected, w_selected, h_selected, heading_selected, class_selected))

    
    # print("Original Format Boxes:")
    # print(original_format_boxes)
    # find the indices of -1 class
    indices = np.where(class_selected == -1)

    # delete the -1 class
    original_format_boxes = np.delete(original_format_boxes, indices, axis=0)
    # print("Original Format Boxes11111111111111:")
    # print(original_format_boxes)
    # import pdb; pdb.set_trace()
    return original_format_boxes


def create_detection_model(args, dataset_config):
    from votenet import VoteNet
    model = VoteNet(dataset_config.num_class,
                    dataset_config.num_heading_bin,
                    dataset_config.mean_size_arr,
                    input_feature_dim=args.num_input_channel,
                    num_proposal=args.num_target,
                    vote_factor=args.vote_factor,
                    sampling=args.cluster_sampling)
    return model


def check_state_dict_consistency(loaded_state_dict, model_state_dict):
    """check consistency between loaded parameters and created model parameters
    """
    valid_state_dict = {}
    for k in loaded_state_dict:
        if k in model_state_dict:
            if loaded_state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}'.format(
                    k, model_state_dict[k].shape, loaded_state_dict[k].shape))
                valid_state_dict[k] = model_state_dict[k]
            else:
                valid_state_dict[k] = loaded_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))

    for k in model_state_dict:
        if not (k in loaded_state_dict):
            print('No param {}.'.format(k))
            valid_state_dict[k] = model_state_dict[k]

    return valid_state_dict


def load_detection_model(model, model_path, model_name=None, optimizer=None, lr=None, lr_step=None, lr_rate=None):
    start_epoch = 0
    
    if model_name is None:
        
        checkpoint_filename = os.path.join(ROOT_DIR, model_path, 'checkpoint.tar')
        checkpoint = torch.load(checkpoint_filename) #Load all tensors onto the CPU
    else:
        checkpoint_filename = os.path.join(ROOT_DIR, model_path, model_name+'_checkpoint.tar')
        checkpoint = torch.load(checkpoint_filename)
    print('loaded {}, epoch {}'.format(checkpoint_filename, checkpoint['epoch']))
    state_dict_ = checkpoint['model_state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    import pdb; pdb.set_trace()
    state_dict = check_state_dict_consistency(state_dict, model_state_dict)
    model.load_state_dict(state_dict, strict=True)

    # resume optimizer parameters
    if optimizer is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= lr_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
        return model, optimizer, start_epoch
    else:
        return model


def save_model(log_dir, epoch, model, model_name=None, optimizer=None):
    # Save checkpoint
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict() # with nn.DataParallel() the net is added as a submodule of DataParallel
    else:
        model_state_dict = model.state_dict()

    save_dict = {'model_state_dict': model_state_dict}

    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()

    if model_name is None:
        # if model_name is not specified, the saved model is the detection model
        save_dict['epoch'] = epoch + 1  # after training one epoch, the start_epoch should be epoch+1
        torch.save(save_dict, os.path.join(log_dir, 'checkpoint.tar'))
    else:
        # otherwise the saved model is the meta-weight generator
        save_dict['epoch'] = epoch + 1
        torch.save(save_dict, os.path.join(log_dir, model_name + '_checkpoint.tar'))