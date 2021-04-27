# Copyright (C) 2016-2018 Alibaba Group Holding Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-
import mxnet as mx
import xdl
import numpy as np
import time
import datetime

import os
import ctypes
from xdl.python.utils.collections import READER_HOOKS, get_collection
from xdl.python.utils.metrics import add_metrics
from xdl.python.training.train_session import QpsMetricsHook, MetricsPrinterHook
from utils import Auc, Dist

'''
Usage:
python mtl.py --run_mode=local --ckpt_dir=./ckpt --task_index=0 --config=config.json
'''

batch_size = 5000
embed_size = 12
train_batch_num = 8000
test_batch_num = 8000

lr = 0.001
is_debug = 0

user_fn=['101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124', '125', '126', '127', '128', '129']
ad_fn=['205', '206', '207', '210', '216', '508', '509', '702', '853', '301']

def reader(name, files, epochs, batch_size, label_dim, user_fn, ad_fn,
           file_type=xdl.parsers.pb, fs_type=xdl.fs.local, enable_state=False):
    data_io = xdl.DataIO(name, file_type=file_type, fs_type=fs_type,
                         enable_state=enable_state)
    data_io.epochs(epochs)
    data_io.threads(len(files))
    data_io.batch_size(batch_size)
    data_io.label_count(label_dim)

    for fn in user_fn:
        data_io.feature(name=fn, type=xdl.features.sparse, table=1)

    for fn in ad_fn:
        data_io.feature(name=fn, type=xdl.features.sparse)

    for path in files:
        data_io.add_path(path)

    data_io.startup()
    return data_io


def run(is_training, files):

    data_io = reader("mmoe", files, 2, batch_size, 2, user_fn, ad_fn)
    batch = data_io.read()

    user_embs = list()
    for fn in user_fn:
        emb = xdl.embedding('u_'+fn, batch[fn],
                            xdl.TruncatedNormal(stddev=0.001),
                            embed_size, 1000, 'sum', vtype='hash')
        user_embs.append(emb)

    ad_embs = list()
    for fn in ad_fn:
        emb = xdl.embedding('a_'+fn, batch[fn],
                            xdl.TruncatedNormal(stddev=0.001),
                            embed_size, 1000, 'sum', vtype='hash')
        ad_embs.append(emb)

    var_list = model(is_training)(ad_embs, user_embs,
                                  batch["indicators"][0],
                                  batch["label"])
    keys = ['loss', 'ctr_prop', 'ctcvr_prop', 'ulike_prob', 'ctr_label',
            'ctcvr_label', 'cvr_label']
    run_vars = dict(zip(keys, list(var_list)))

    hooks = []
    if is_training:
        train_op = xdl.Adam(lr).optimize()
        hooks = get_collection(READER_HOOKS)
        if hooks is None:
            hooks = []
        if xdl.get_task_index() == 0:
            ckpt_hook = xdl.CheckpointHook(1000, max_to_keep=100)
            hooks.append(ckpt_hook)

        run_vars.update({None: train_op})

    if is_debug > 1:
        print("=========gradients")
        grads = xdl.get_gradients()
        grads_keys = grads[''].keys()
        grads_keys.sort()
        for key in grads_keys:
            run_vars.update({"grads {}".format(key): grads[''][key]})

    hooks.append(QpsMetricsHook())
    log_format = "lstep[%(lstep)s] gstep[%(gstep)s] " \
                 "lqps[%(lqps)s] gqps[%(gqps)s]"
    hooks.append(MetricsPrinterHook(log_format, 100))

    ckpt = xdl.get_config("checkpoint", "ckpt")
    print('ckpt: ', ckpt)
    from xdl.python.training.saver import Saver
    saver = Saver()
    #saver.save('./esmm_init.ckp')
    if ckpt is not None and len(ckpt) > 0:
        if int(xdl.get_task_index()) == 0:
            #from xdl.python.training.saver import Saver
            #saver = Saver()
            print("restore from %s" % ckpt)
            saver.restore(ckpt)
        else:
            time.sleep(120)

    sess = xdl.TrainSession(hooks)

    if is_training:
        itr = 1
        ctr_auc = Auc('ctr')
        ctcvr_auc = Auc('ctcvr')
        cvr_auc = Auc('cvr')

        while not sess.should_stop():
            print('iter=', itr)
            values = sess.run(run_vars.values())
            if not values:
                continue
            value_map = dict(zip(run_vars.keys(), values))
            print('loss=', value_map['loss'])
            ctr_auc.add(value_map['ctr_prop'], value_map['ctr_label'])
            ctcvr_auc.add(value_map['ctcvr_prop'], value_map['ctcvr_label'])
            cvr_auc.add(value_map['ulike_prob'], np.clip(value_map['cvr_label'] + value_map['ctcvr_label'], 0, 1))

            itr += 1
            if itr > train_batch_num:
                break
        ctr_auc.show()
        ctcvr_auc.show()
        cvr_auc.show()
    else:
        print('+++++++++++++++++++test')
        ctr_test_auc = Auc('ctr')
        ctcvr_test_auc = Auc('ctcvr')
        cvr_test_auc = Auc('cvr')
        
        ctr_dist = Dist('ctr')
        ctcvr_dist = Dist('ctcvr')
        cvr_dist = Dist('cvr')

        count = 0
        mean_ctr_prop = 0
        mean_ctcvr_prop = 0
        mean_ulike = 0
        for i in xrange(test_batch_num):
            print('iter=', i+1)
            values = sess.run(run_vars.values())
            value_map = dict(zip(run_vars.keys(), values))
            print('test_loss=', value_map['loss'])
            ctr_test_auc.add(value_map['ctr_prop'], value_map['ctr_label'])
            ctcvr_test_auc.add(value_map['ctcvr_prop'],
                    value_map['ctcvr_label'])
            cvr_test_auc.add(value_map['ulike_prob'], np.clip(value_map['cvr_label'] + value_map['ctcvr_label'], 0, 1))
            #cvr_test_auc.add_with_filter(value_map['cvr_prop'],
            #        value_map['cvr_label'],
            #        np.where(value_map['ctr_label'] == 1))
            mean_ctr_prop += np.sum(value_map['ctr_prop'])
            mean_ctcvr_prop += np.sum(value_map['ctcvr_prop'])
            mean_ulike += np.sum(value_map['ulike_prob'])
            count += len(value_map['ctr_prop'])
            ctr_dist.add(value_map['ctr_prop'])
            ctcvr_dist.add(value_map['ctcvr_prop'])
            cvr_dist.add(value_map['ulike_prob'])
        mean_ctr_prop /= count
        mean_ctcvr_prop /= count
        mean_ulike /= count
        print('mean_ctr_prop: ', mean_ctr_prop)
        print('mean_ctcvr_prop: ', mean_ctcvr_prop)
        print('mean_ulike: ', mean_ulike)

        ctr_test_auc.show()
        ctcvr_test_auc.show()
        cvr_test_auc.show()
        ctr_dist.show()
        ctcvr_dist.show()
        cvr_dist.show()

class Identity(mx.init.Initializer):
    def __init__(self, init_value=None):
        super(Identity, self).__init__(init_value=init_value)

def batch_norm(tag, data, fix_gamma=False, eps=1e-3, momentum=0.9):
    gamma = mx.sym.var(name='%s_bn_gamma' % tag, init=mx.init.Constant(1.0))
    beta = mx.sym.var(name='%s_bn_beta' % tag, init=mx.init.Constant(0.))
    moving_mean = mx.sym.var(name='%s_bn_moving_mean' %
                             tag, init=mx.init.Constant(0.))
    moving_var = mx.sym.var(name='%s_bn_moving_var' %
                            tag, init=mx.init.Constant(0.))
    return mx.sym.BatchNorm(data=data, gamma=gamma, beta=beta,
            moving_mean=moving_mean, moving_var=moving_var,
            eps=eps, momentum=momentum, fix_gamma=fix_gamma,
            name='%s_bn' % tag)

def fc(tag, data, in_dim, out_dim, active='prelu', use_bn=False):
    init_stddev = 1.0
    init_mean = 0.0
    init_value = (init_stddev * np.random.randn(out_dim, in_dim).\
            astype(np.float32) + init_mean) / np.sqrt(in_dim)

    weight = mx.sym.var(name='%s_weight' % tag,
            init=Identity(init_value=init_value.tolist()))
    bias = mx.sym.var(name='%s_bias' % tag, init=mx.init.Constant(0.1))
    dout = mx.sym.FullyConnected(data=data, weight=weight,
            bias=bias, num_hidden=out_dim, name=tag)

    if use_bn:
        dout = batch_norm(tag, dout)

    if active == 'sigmoid':
        out = mx.symbol.Activation(
            data=dout, act_type="sigmoid", name=('%s_sigmoid' % tag))
    elif active == 'relu':
        out = mx.symbol.Activation(
            data=dout, act_type="relu", name=('%s_relu' % tag))
    elif active == 'prelu':
        gamma = mx.sym.var(name=('%s_prelu_gamma' % tag),
                           init=mx.init.Constant(-0.25))
        out = mx.symbol.LeakyReLU(data=dout, act_type='prelu',
                gamma=gamma, slope=0.25, name=('%s_prelu' % tag))
    elif active == 'dice':
        out = batch_norm(tag, dout, fix_gamma=True, eps=1e-4, momentum=0.99)
        prop = mx.symbol.Activation(data=out, act_type="sigmoid",
                name=('%s_dice_sigmoid' % tag))
        dice_gamma = mx.sym.var(name='%s_dice_gamma' % tag,
                shape=((1, out_dim)), init=mx.init.Constant(-0.25))
        out = mx.symbol.broadcast_mul(lhs=dice_gamma,
                rhs=((1.0 - prop) * dout)) + prop * dout
    elif active == '':  # no active function in the last layer
        out = dout
    else:
        print("Unknown activation type %s." % active)
        exit(1)

    return out


def model(is_training, device_type="cpu"):
    @xdl.mxnet_wrapper(is_training=is_training, device_type=device_type)
    def _model(ad_embs, user_embs, indicator, label):
        indicator = mx.sym.BlockGrad(indicator)
        din_ad = mx.sym.concat(*ad_embs)
        din_user = mx.sym.concat(*user_embs)

        din_user = mx.sym.take(din_user, indicator)
        din = mx.sym.concat(din_user, din_ad)

        feature_size = len(ad_embs)+len(user_embs)
        act = 'prelu'

        ############## expert
        expert_num = 2
        expert_out = list()
        #hidden_size = [256, 128, 64]
        for i in range(expert_num):
            expert_fc1 = fc('expert_{}_fc1'.format(i), din, feature_size*embed_size, 256, act)
            expert_fc2 = fc('expert_{}_fc2'.format(i), expert_fc1, 256, 128, act)
            expert_fc3 = fc('expert_{}_fc3'.format(i), expert_fc2, 128, 64, act)
            expert_out.append(expert_fc3)

        ############## like
        ulike_fc1 = fc('ulike_fc1', din, feature_size*embed_size, 200, act)
        ulike_fc2 = fc('ulike_fc2', ulike_fc1, 200, 80, act)
        ulike_out = fc('ulike_out', ulike_fc2, 80, 1, '')

        ############## ctr
        # gate
        ctr_gate = fc('ctr_gate', din, feature_size*embed_size, expert_num, '')
        ctr_gate = mx.symbol.softmax(data=ctr_gate, axis=1)
        ctr_gate_out = list()
        for i, eo in enumerate(expert_out):
            w = mx.symbol.slice_axis(data=ctr_gate, axis=1, begin=i, end=i+1)
            out = mx.symbol.broadcast_mul(lhs=eo, rhs=w)
            ctr_gate_out.append(out)
        ctr_gate_out = mx.symbol.concat(*ctr_gate_out, dim=1)

        ctr_fc1 = fc('ctr_fc1', ctr_gate_out, 64 * len(expert_out), 80, act)
        ctr_out = fc('ctr_out', ctr_fc1, 80, 2, '')

        ############## cvr
        # gate
        cvr_gate = fc('cvr_gate', din, feature_size*embed_size, expert_num, '')
        cvr_gate = mx.symbol.softmax(data=cvr_gate, axis=1)
        cvr_gate_out = list()
        for i, eo in enumerate(expert_out):
            w = mx.symbol.slice_axis(data=cvr_gate, axis=1, begin=i, end=i+1)
            out = mx.symbol.broadcast_mul(lhs=eo, rhs=w)
            cvr_gate_out.append(out)
        cvr_gate_out = mx.symbol.concat(*cvr_gate_out, dim=1)

        cvr_fc1 = fc('cvr_fc1', cvr_gate_out, 64 * len(expert_out), 80, act)
        cvr_out = fc('cvr_out', cvr_fc1, 80, 2, '')


        ctr_clk = mx.symbol.slice_axis(data=label, axis=1, begin=0, end=1)
        ctr_label = mx.symbol.concat(*[1 - ctr_clk, ctr_clk], dim=1)

        ctcvr_buy = mx.symbol.slice_axis(data=label, axis=1, begin=1, end=2)
        ctcvr_label = mx.symbol.concat(*[1 - ctcvr_buy, ctcvr_buy], dim=1)

        ### user_like prob
        ulike_prob = mx.symbol.sigmoid(data=ulike_out, axis=1)

        ### ctr_prob
        ctr_out = mx.symbol.sigmoid(data=ctr_out, axis=1)
        ctr_lclk = mx.symbol.slice_axis(data=ctr_out, axis=1, begin=0, end=1)
        ctr_uclk = mx.symbol.slice_axis(data=ctr_out, axis=1, begin=1, end=2) * 0.0100

        ctr_prop_one = ctr_lclk * ulike_prob + ctr_uclk * (1 - ulike_prob)
        ctr_prop = mx.symbol.concat(*[1 - ctr_prop_one, ctr_prop_one], dim=1)

        ### cvr_prob
        cvr_prop = mx.symbol.sigmoid(data=cvr_out, axis=1)
        cvr_lclk = mx.symbol.slice_axis(data=cvr_prop, axis=1, begin=0, end=1)
        cvr_uclk = mx.symbol.slice_axis(data=cvr_prop, axis=1, begin=1, end=2) * 0.0001

        cvr_prop_one = cvr_lclk * ulike_prob + cvr_uclk * (1 - ulike_prob)
        cvr_prop = mx.symbol.concat(*[1 - cvr_prop_one, cvr_prop_one], dim=1)

        #ctr_prop_one =mx.symbol.slice_axis(data=ctr_prop, axis=1,
        #                                   begin=1, end=2)
        #cvr_prop_one =mx.symbol.slice_axis(data=cvr_prop, axis=1,
        #                                   begin=1, end=2)

        ############## new version
        #ctcvr_prop_one = ctr_prop_one * cvr_prop_one
        ctcvr_prop_one = cvr_prop_one
        ctcvr_prop = mx.symbol.concat(*[1 - ctcvr_prop_one, \
            ctcvr_prop_one], dim=1)

        loss_r = mx.symbol.MakeLoss(-mx.symbol.sum_axis( \
            #mx.symbol.broadcast_mul( \
            #        lhs=mx.symbol.log(cvr_prop) * ctcvr_label, \
            #        rhs=ctr_clk) +  \
            mx.symbol.log(ctr_prop) * ctr_label + \
            mx.symbol.log(ctcvr_prop) * ctcvr_label, \
            axis=1, keepdims=True), \
            normalization="null", \
            grad_scale = 1.0 / batch_size, \
            name="esmm_loss")
        ctr_loss = - mx.symbol.sum(mx.symbol.log(ctr_prop) * ctr_label ) \
                / batch_size
        ctcvr_loss = - mx.symbol.sum(mx.symbol.log(ctcvr_prop) * ctcvr_label) \
                / batch_size

        cnt_cvr_sample = mx.symbol.sum_axis(ctr_clk)
        cnt_ctcvr_sample = mx.symbol.sum_axis(ctcvr_buy)

        cvr_loss = - mx.symbol.sum(mx.symbol.sum_axis( \
            mx.symbol.log(cvr_prop) * ctcvr_label, \
            axis=1, keepdims=True) * ctr_clk) / cnt_cvr_sample

        return loss_r, mx.sym.BlockGrad(ctr_lclk * ulike_prob), \
               mx.sym.BlockGrad(cvr_lclk * ulike_prob), \
               mx.sym.BlockGrad(ulike_prob), mx.sym.BlockGrad(ctr_clk), \
               mx.sym.BlockGrad(ctcvr_buy), mx.sym.BlockGrad(ctcvr_buy)
    return _model

if __name__ == '__main__':
    is_training = xdl.get_config("is_training")
    files = xdl.get_config("files")
    run(is_training, files)
