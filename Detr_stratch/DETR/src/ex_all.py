import pickle
import numpy as np
from numpy import dtype

from models import detr
from models import matcher
from mindspore.common.tensor import Tensor
from mindspore import Model
# from mindvision.engine.callback import LossMonitor
import mindspore.nn as nn
from mindspore import context, ops
from mindspore import load_checkpoint, load_param_into_net

from datasets import coco, cocopanoptic
from models.position_encoding import PositionEmbeddingSine
from models.backbone import build_backbone
from models.resnet import resnet50, resnet101
from models.transformer import MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, build_transformer
from models.detr import bulid_detr
from models.segmentation import DETRsegm
from models.matcher import build_matcher, build_criterion, box_cxcywh_to_xyxy, box_iou, generalized_box_iou

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")  

def ex_coco():
    dataset = coco.build(img_set='val', batch=1, shuffle=False)
    n = 0
    for d in dataset.create_dict_iterator():
        for k, v in d.items():
            print(n, k, v.shape)
            if 'size' in k:
                print(v)
        n += 1
        if n > 1:
            break

def ex_coco_pano():
    dataset = cocopanoptic.build(img_set='val', batch=2, shuffle=False)
    n = 0
    for d in dataset.create_dict_iterator():
        for k, v in d.items():
            print(n, k, v.shape)
        n += 1
        if n > 10:
            break

def ex_position_encoding():
    p = PositionEmbeddingSine()
    print(p)
    m = np.ones((2, 3, 4)).astype(np.float32)
    m = Tensor(m)
    print(m)
    print(type(m))
    print(p(m).shape)  

def ex_backbone():
    with open('./sample_np.pkl', 'rb') as f:
        s = pickle.load(f)
    x = Tensor(s['img'].astype(np.float32))
    mask = Tensor(s['mask'].astype(np.float32))
    net = build_backbone(resnet='resnet50', return_interm_layers=True, is_dilation=False)
    x, mask, pos = net(x, mask)
    print('x:', x[-1][0][0][0])

def ex_resnet():
    with open('./sample_np.pkl', 'rb') as f:
        s = pickle.load(f)
    x = Tensor(s['img'].astype(np.float32))
    print('x:', x.shape)
    net = resnet50(return_interm_layers=True, is_dilation=True)
    x = net(x)
    print('x:', x[-1].shape, x[-1][0][0][0])

def ex_multhead():
    k = v = Tensor(np.ones((20, 2, 256)).astype(np.float32))
    q = Tensor(np.ones((30, 2, 256)).astype(np.float32))
    mha = MultiHeadAttention(n_head=8, d_model=256)
    mask = Tensor(np.ones((2, k.shape[0])).astype(np.float32))
    output, attn = mha(q, k, v, mask)
    print('output shape:', output.shape, 'attn shape:', attn.shape)

def ex_encoder_layer():
    tel = TransformerEncoderLayer()
    src = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    src_mask = Tensor(np.ones((2, 10)).astype(np.float32))
    pos = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    out = tel(src=src, src_padding_mask=src_mask, pos=pos)
    print('out shape:', out.shape)

def ex_encoder():
    tel = TransformerEncoderLayer()
    src = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    src_mask = Tensor(np.ones((2, 10)).astype(np.float32))
    pos = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    encoder = TransformerEncoder(encoder_layer=tel, num_layers=2, norm=None)
    out = encoder(src=src, src_padding_mask=src_mask, pos=pos)
    print('out shape:', out.shape)

def ex_decoder_layer():
    src = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    src_mask = Tensor(np.ones((2, 10)).astype(np.float32))
    pos = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    tgt = Tensor(np.ones((100, 2, 256)).astype(np.float32))
    query_pos = Tensor(np.ones((100, 2, 256)).astype(np.float32))
    tdl = TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward=2048,
                                  dropout=0.1, activation="relu")
    out = tdl(tgt, src, memory_padding_mask=src_mask, pos=pos, query_pos=query_pos)
    print('out shape:', out.shape)

def ex_decoder():
    src = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    src_mask = Tensor(np.ones((2, 10)).astype(np.float32))
    pos = Tensor(np.ones((10, 2, 256)).astype(np.float32))
    tgt = Tensor(np.ones((100, 2, 256)).astype(np.float32))
    query_pos = Tensor(np.ones((100, 2, 256)).astype(np.float32))
    tdl = TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward=2048,
                                  dropout=0.1, activation="relu")
    decoder = TransformerDecoder(decoder_layer=tdl, num_layers=2, norm=None, return_intermediate=True)
    hs = decoder(tgt, src, memory_padding_mask=src_mask, pos=pos,
                 query_pos=query_pos)
    print('hs shape:', hs.shape)

def ex_transformer():
    tf = build_transformer()
    src = Tensor(np.ones((2, 256, 2, 5)).astype(np.float32))
    mask = Tensor(np.zeros((2, 2, 5)).astype(np.float32))
    pos = Tensor(np.ones((2, 256, 2, 5)).astype(np.float32))
    query_embed = Tensor(np.ones((100, 256)).astype(np.float32))
    hs, memory = tf(src=src, mask=mask, query_embed=query_embed, pos_embed=pos)
    print('hs shape:', hs.shape, 'memory shape:', memory.shape)

def ex_detr():
    with open('sample_np.pkl', 'rb') as f:
        s = pickle.load(f)
    x = Tensor(s['img'].astype(np.float32))
    mask = Tensor(s['mask'].astype(np.float32))
    net = bulid_detr(return_interm_layers=True)
    out = net(x, mask)
    for k, v in out.items():
        print(k, v.shape)
    print(out.keys())

def ex_segmentation():
    x = Tensor(np.ones((2, 3, 1066, 1201)).astype(np.float32))
    mask = Tensor(np.zeros((2, 1066, 1201)).astype(np.float32))
    net = bulid_detr(num_classes=250, return_interm_layers=True)
    model = DETRsegm(net, freeze_detr=True)
    out = model(x, mask)
    for k, v in out.items():
        print(k, v.shape)

def pkl_read():
    with open('sample_coco_np.pkl', 'rb') as f:
        s = pickle.load(f)
    target = dict()
    print('========= target ==========')
    for k, v in s['target'][-1].items():
        target[k] = Tensor(v)
        print(k, v.shape)
    with open('output_np.pkl', 'rb') as f:
        output = pickle.load(f)
    print('========= output ==========')
    print(output.keys())
    print('pred_logits:', output['pred_logits'].shape, type(output['pred_logits']))
    print('pred_boxes:', output['pred_boxes'].shape)
    print('aux_outputs len:', len(output['aux_outputs']))
    output['pred_logits'] = Tensor(output['pred_logits'])
    output['pred_boxes'] = Tensor(output['pred_boxes'])
    print('size:', target['size'], 'orig_size:', target['orig_size'])
    print('boxes:', target['boxes'], type(target['boxes']))
    return output, [target]

def ex_iou():
    box1 = Tensor(np.array([[0.5205, 0.6888, 0.9556, 0.5955],
        [0.2635, 0.2472, 0.4989, 0.4764],
        [0.3629, 0.7329, 0.4941, 0.5106],
        [0.6606, 0.4189, 0.6789, 0.7815],
        [0.3532, 0.1326, 0.1180, 0.0969],
        [0.2269, 0.1298, 0.0907, 0.0972],
        [0.3317, 0.2269, 0.1313, 0.1469]]).astype(np.float32))
    box2 = Tensor(np.array([[0.3532, 0.1326, 0.1180, 0.0969],
        [0.2269, 0.1298, 0.0907, 0.0972],
        [0.3317, 0.2269, 0.1313, 0.1469]]).astype(np.float32))
    b1 = box_cxcywh_to_xyxy(box1)
    b2 = box_cxcywh_to_xyxy(box2)
    print('b1:', b1.shape, 'b2:', b2.shape)
    iou, union = box_iou(b1, b2)
    print('iou:', iou.shape, 'union:', union.shape)
    giou = generalized_box_iou(b1, b2)
    print('giou:', giou.shape)

def ex_matcher():
    hm = build_matcher()
    print(hm)
    output, target = pkl_read()
    indices = hm(output, target)
    print(indices)
    for i, (src, _) in enumerate(indices):
        print(i, src, _)

def ex_criterion():
    with open('loss_input.pkl', 'rb') as f:
        s = pickle.load(f)
    outputs = {k:Tensor(v) for k, v in s['out'].items()}
    targets = s['tgts']
    criterion = build_criterion(is_segmentation=True)
    outs = criterion(outputs, targets)
    for k, v in outs.items():
        print(k, v)

def ex_one_sample():
    is_segmentation = False
    with open('sample_coco_np.pkl', 'rb') as f:
        s = pickle.load(f)
    x = Tensor(s['img'].astype(np.float32))
    mask = Tensor(s['mask'].astype(np.float32))
    print('x:', x.shape, 'mask:', mask.shape)
    target = [{k: Tensor(v) for k, v in t.items()} for t in s['target']]
    for t in target:
        for k, v in t.items():
            print(k, v.shape)
        break
    print('构建detr网络......')
    if is_segmentation:
        net = detr.bulid_detr(resnet='resnet50', return_interm_layers=is_segmentation,
                              num_classes=250, is_dilation=False)
        net = DETRsegm(net, freeze_detr=False)
    else:
        net = detr.bulid_detr(resnet='resnet50', return_interm_layers=False, num_classes=91, is_dilation=False)
    param_dict = load_checkpoint('./detr/resume/resnet50.ckpt')
    load_param_into_net(net, param_dict)
    net.set_train(False)
    out = net(x, mask)
    for k, v in out.items():
        print(k, v.shape, v[0][0])
    print(out.keys())

def main():
    ex_coco()
    # ex_coco_pano()
    # ex_position_encoding()
    ex_backbone()
    ex_resnet()
    ex_multhead()
    ex_encoder_layer()
    ex_encoder()
    ex_decoder_layer()
    ex_decoder()
    ex_transformer()
    ex_detr()
    ex_segmentation()
    ex_iou()
    ex_matcher()
    ex_criterion()
    ex_one_sample()

if __name__ == '__main__':
    main()
