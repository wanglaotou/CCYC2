#!/usr/bin.python
# -*- coding: utf-8 -*-

import re, os, sys, inspect, shutil, commands
from ConfigParser import SafeConfigParser
from operator import itemgetter
from pynvml import *
pfolder = os.path.realpath(os.path.abspath (os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"..")))
if pfolder not in sys.path:
        sys.path.insert(0, pfolder)
reload(sys)
sys.setdefaultencoding('utf8')

def import_caffe(caffe_root, log_path=""):
  caffe_dir = os.path.join(caffe_root, 'python')
  if caffe_dir not in sys.path: 
    sys.path.insert(0, caffe_dir)
  #os.environ['GLOG_logtostderr'] = '0'
  #os.environ['FLAGS_alsologtostderr'] = '1'
  #os.environ['GLOG_stderrthreshold'] = '1'
  #os.environ['GLOG_log_dir'] = os.path.os.path.realpath(log_path)
  
  import caffe
  if log_path != "":
    caffe.set_log_dir(os.path.os.path.join(log_path, 'pycaffe'))

  return caffe

def auto_select_gpu():
  nvmlInit()
  deviceCount = nvmlDeviceGetCount()
  gpu_wei = 0.75
  min_score = 200
  ret = -1
  for i in xrange(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    gr = nvmlDeviceGetUtilizationRates(handle)
    score = gr.gpu * gpu_wei + gr.memory * (1 - gpu_wei)
    if score < min_score:
      min_score = score
      ret = i
  nvmlShutdown()
  return ret

def set_caffe_env(template_root, proto_root, snapshot_prefix, train_sample_path, train_root_path, validate_sample_path, validate_root_path):
  train_proto = os.path.join(template_root, 'train_val.prototxt')
  with open(train_proto, 'r') as train_fd:
    content = train_fd.read()
    content = re.sub(r'path-to-training-sample', train_sample_path, content)
    content = re.sub(r'path-to-training-root', train_root_path, content)
    content = re.sub(r'path-to-validate-sample', validate_sample_path, content)
    content = re.sub(r'path-to-validate-root', validate_root_path, content)
    with open(os.path.join(proto_root, 'train_val.prototxt'), 'w') as proto_fd:
      print >>proto_fd, content

  solver_proto = os.path.join(template_root, 'solver.prototxt')
  with open(solver_proto, 'r') as solver_fd:
    content = solver_fd.read()
    content = re.sub(r'net-train-val', os.path.join(proto_root, 'train_val.prototxt'), content)
    content = re.sub(r'snapshot-prefix', snapshot_prefix, content)
    with open(os.path.join(proto_root, 'solver.prototxt'), 'w') as proto_fd:
      print >>proto_fd, content
  idx = 1
  name_proto = 'solver_ft%d.prototxt'%idx
  solver_prototxt = os.path.join(template_root, name_proto)
  while os.path.exists(solver_prototxt):
    with open(solver_prototxt, 'r') as solver_fd:
      content = solver_fd.read()
      content = re.sub(r'net-train-val', os.path.join(proto_root, 'train_val.prototxt'), content)
      snapshot_prefix_tmp = snapshot_prefix+'_ft%d'%idx
      content = re.sub(r'snapshot-prefix', snapshot_prefix_tmp, content)
      with open(os.path.join(proto_root, name_proto), 'w') as proto_fd:
        print >>proto_fd, content
    idx += 1
    name_proto = 'solver_ft%d.prototxt'%idx
    solver_prototxt = os.path.join(template_root, name_proto)

  deploy_proto = os.path.join(template_root, 'deploy.prototxt')
  shutil.copyfile(deploy_proto, os.path.join(proto_root, 'deploy.prototxt'))

def get_max_iter(sovler_proto):
  with open(sovler_proto, 'r') as solver_fd: 
    content = solver_fd.read()
    r = re.compile("^max_iter: (.*)$", re.M)
    ret = r.findall(content)
    if len(ret) > 0:
      max_iter = int(ret[0])
    else:
      raise Exception('no max_iter')
  return max_iter

def get_iter_from_name(name, suffix):
  r = re.compile('(\d*).'+suffix)
  ret = r.findall(name)
  if len(ret)>0:
    it = int(ret[0])
    return it
  else:
    return None
  
def get_max_iter_trained_4resume(model_root):
  name_list = os.listdir(model_root)
  iter_list = []
  for name in name_list:
    it = get_iter_from_name(name, 'solverstate')
    if it is not None:
      iter_list.append(it)
  if len(iter_list)>0:
    iter_max_trained = max(iter_list) 
    return iter_max_trained
  else:
    return None
  
def caffe_train_validate(caffe_root, model_root, proto_root, log_path, net_name, model_path_end, pretrained_model=None, do_finetune=False, gpu_id_set=-1):
  caffe = import_caffe(caffe_root, log_path)
  caffe.set_mode_gpu()
  if gpu_id_set<=-1:
    gpu_id = auto_select_gpu()
    if gpu_id < 0:
      raise Exception('No GPU card')
    else:
      print "select gpu#%d" % gpu_id
      with open(os.path.join(log_path, 'gpu_%d'%gpu_id), 'w') as gpu_fd:
        print >> gpu_fd, "select gpu#%d" % gpu_id
  else:
    gpu_id = gpu_id_set
    print "use gpu#%d" % gpu_id
    with open(os.path.join(log_path, 'gpu_%d'%gpu_id), 'w') as gpu_fd:
      print >> gpu_fd, "select gpu#%d" % gpu_id
  caffe.set_device(gpu_id)
  solver_prototxt = os.path.join(proto_root, 'solver.prototxt')
  solver = caffe.get_solver(solver_prototxt)
  if pretrained_model is not None:    
    solver.net.copy_from(pretrained_model)
    print 'finetune from ', pretrained_model
  else:
    #to see if need resume
    iter_max_trained = get_max_iter_trained_4resume(model_root)
    if iter_max_trained is not None:
      model_4resume = os.path.join(model_root, "%s_iter_%d.solverstate" % (net_name, iter_max_trained))
      solver.restore(model_4resume)
      print 'resume from ', model_4resume
  solver.solve()
  max_iter = get_max_iter(solver_prototxt)
  net_model = os.path.join(model_root, "%s_iter_%d.caffemodel" % (net_name, max_iter))
  if do_finetune:
    idx = 1
    solver_prototxt = os.path.join(proto_root, 'solver_ft%d.prototxt'%idx) #the exist of this file means that it needs finetune
    while os.path.exists(solver_prototxt) and os.path.exists(net_model):
      solver = caffe.get_solver(solver_prototxt)
      solver.net.copy_from(net_model)
      solver.solve()
      max_iter = get_max_iter(solver_prototxt)
      net_model = os.path.join(model_root, "%s_ft%d_iter_%d.caffemodel" % (net_name, idx, max_iter))
      idx += 1
      solver_prototxt = os.path.join(proto_root, 'solver_ft%d.prototxt'%idx)
  if os.path.exists(net_model):
    shutil.copyfile(net_model, model_path_end)
    
