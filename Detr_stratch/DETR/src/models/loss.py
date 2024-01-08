#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 03:04:55 2024

@author: farshid
"""


# ======================================================================
"""Connecting the network to the loss function."""
import mindspore.nn as nn


class CustomWithLossCell(nn.Cell):
    """
    Connecting the network to the loss function.
    Notes:
        The number of columns in the dataset is not 2, then the forward network
        needs to be connected to the loss function.
    
    Args:
        net (Cell): Network.
        loss_fn (Cell): Loss function.
    
    Return:
        loss value.
    """

    def __init__(self, net, loss_fn):
        super(CustomWithLossCell, self).__init__()
        self.net = net
        self._loss_fn = loss_fn

    def construct(self, img, img_mask, img_id, masks, cats,
                  bbox, size, orig_size, iscrowd, area, len_list):
        """ Connecting the network to the loss function. """
        tgt = [img_id, size, orig_size, iscrowd, area]
        output = self.net(img, img_mask)
        tgt = []
        begin = 0
        for i in range(img.shape[0]):
            if len_list[i][0] < 2:
                end = begin + int(len_list[i][0])
            else:
                end = begin + len_list[i][0]
            target = {'labels': cats[0][begin:end],
                      'boxes': bbox[0][begin:end],
                      'masks': masks[0][begin:end]}
            begin = end
            tgt.append(target)
        loss = self._loss_fn(output, tgt)
        weight_dict = self._loss_fn.weight_dict
        losses = 0
        for k in loss.keys():
            if k in weight_dict:
                losses += loss[k] * weight_dict[k]
        return losses
