#!/usr/bin.python
# -*- coding: utf-8 -*-

import os, sys, inspect, shutil, random, luigi, cv2, stat
from ConfigParser import SafeConfigParser
pfolder = os.path.realpath(os.path.abspath (os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"..")))
if pfolder not in sys.path:
  sys.path.insert(0, pfolder)
reload(sys)
sys.setdefaultencoding('utf8')
from util.caffe_exe import *
from evaluation.eval import *
from meta import *

class Net(luigi.Task):
  conf = luigi.Parameter()
  
  def __init__(self, *args, **kwargs):
    luigi.Task.__init__(self, *args, **kwargs)
    parser = SafeConfigParser()
    parser.read(self.conf)    
    self.caffe_root = parser.get("basic", "caffe_root")
    workspace = parser.get("basic", "workspace")
    self.net_name   = parser.get('basic', 'net_name')    
    self.template_root = os.path.join(os.path.dirname(self.conf), 'net_structure')

    workspace = os.path.join(workspace, self.net_name)
    self.proto_path = os.path.join(workspace, 'proto')
    self.model_path = os.path.join(workspace, 'model')
    self.log_path = os.path.join(workspace, 'log')
    self.model_path_end = os.path.join(self.model_path, '%s.caffemodel' % self.net_name)
    self.snapshot_prefix = os.path.join(self.model_path, self.net_name)
    
    if parser.has_option('common', 'gpu_id'):
      self.gpu_id_set = parser.getint('common', 'gpu_id')
    else:
      self.gpu_id_set = -1 #auto select
    if parser.has_option('common', 'can_resume'):
      self.can_resume = parser.getboolean('common', 'can_resume')
    else:
      self.can_resume = False    
    self.do_finetune = parser.getboolean('common', 'finetune')

    #finetune from pretrained model
    if parser.has_option('common', 'pretrained'):
      self.pretrained_model = parser.get('common', 'pretrained')
    else:
      self.pretrained_model = None
    
  def requires(self):
    reqs = {'spl_train': ExternalSampleTrain(self.conf), \
            'root_train': ExternalRootTrain(self.conf), \
            'spl_validate': ExternalSampleValidate(self.conf), \
            'root_validate': ExternalRootValidate(self.conf)}
    return reqs

  def output(self):
    return {"model": luigi.LocalTarget(self.model_path_end)}
  
  def run(self):
    if os.path.exists(self.proto_path):
      if not self.can_resume:
        shutil.rmtree(self.proto_path)
        os.makedirs(self.proto_path)
    else:
      os.makedirs(self.proto_path)
    if os.path.exists(self.model_path):
      if not self.can_resume:
        shutil.rmtree(self.model_path)
        os.makedirs(self.model_path)
    else:
      os.makedirs(self.model_path)
    if os.path.exists(self.log_path):
      if not self.can_resume:
        shutil.rmtree(self.log_path)
        os.makedirs(self.log_path)
    else:
      os.makedirs(self.log_path)
    set_caffe_env(self.template_root, self.proto_path, \
                  self.snapshot_prefix, \
                  self.input()['spl_train'].path, \
                  self.input()['root_train'].path, \
                  self.input()['spl_validate'].path, \
                  self.input()['root_validate'].path)
    caffe_train_validate(self.caffe_root, self.model_path, self.proto_path, self.log_path, self.net_name, \
                  self.model_path_end, self.pretrained_model, self.do_finetune, self.gpu_id_set)

class EvalOnTestset(luigi.Task):
  conf = luigi.Parameter()
  test_set = luigi.Parameter()

  def __init__(self, *args, **kwargs):
    luigi.Task.__init__(self, *args, **kwargs)
    parser = SafeConfigParser()
    parser.read(self.conf)
    self.parser = parser
    self.workspace = parser.get("basic", "workspace")

    #result path
    self.net_name = parser.get('basic', 'net_name')
    self.nm_res = os.path.basename(self.workspace.rstrip(os.sep)) + '_result'
    self.path_res = os.path.join(self.workspace, self.nm_res, self.net_name)

    #for test set
    self.annot = parser.get(self.test_set, 'annot')
    self.data  = parser.get(self.test_set, 'data')
    self.mre_thre = parser.getfloat(self.test_set, 'mre_thre')
    self.is_src_img = parser.getboolean(self.test_set, 'is_src_img')
    self.is_mean_value = parser.getboolean(self.test_set, 'is_mean_value')
    if parser.has_option(self.test_set, 'scale'):
      self.scale = parser.getfloat(self.test_set, 'scale')
    else:
      self.scale = 0.0078125
    if parser.has_option(self.test_set, 'mean_value'):      
      self.mean_value = parser.get(self.test_set, 'mean_value')
    else:
      self.mean_value = '127.5 127.5 127.5'
    self.is_test_half = parser.getboolean(self.test_set, 'is_test_half')
    self.is_test_roi = parser.getboolean(self.test_set, 'is_test_roi')
    self.is_save_dmap = parser.getboolean(self.test_set, 'is_save_dmap')
    nm_file = self.test_set
    self.nm_py = 'eval.py'    
    self.end_file = os.path.join(self.path_res, nm_file)    

  def requires(self):
    reqs = {}
    reqs['net'] = Net(conf=self.conf)
    return reqs

  def output(self):
    return luigi.LocalTarget(self.end_file) 
  
  def run(self):
    #save_path
    conf = self.conf
    path_res = self.path_res
    if not os.path.exists(path_res):
      os.makedirs(path_res)
    #eval
    path_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'evaluation')
    path_py = os.path.join(path_folder, self.nm_py)
    cmd = '{0} -annot="{1}" -data="{2}" -conf="{3}" -o="{4}" -show=0 -thre="{5}" -issrc="{6}" -if_ms="{7}" -mean {8} -scale="{9}" \
           -half="{10}" -roi="{11}" -save="{12}"'.format(path_py, self.annot, self.data, conf, self.end_file, self.mre_thre, \
           int(self.is_src_img), int(self.is_mean_value), self.mean_value, self.scale, int(self.is_test_half), int(self.is_test_roi), int(self.is_save_dmap))
    os.chmod(path_py, stat.S_IRWXU|stat.S_IRWXG|stat.S_IROTH)
    os.system(cmd)

class TrainEval(luigi.Task):
  conf = luigi.Parameter()

  def __init__(self, *args, **kwargs):
    luigi.Task.__init__(self, *args, **kwargs)
    parser = SafeConfigParser()
    parser.read(self.conf)
    self.parser = parser
    workspace = parser.get("basic", "workspace")    
    self.path_end = os.path.join(workspace, 'end_train')
    self.test_sets = [set_name.strip() for set_name in parser.get("basic", "test_sets").split(',')]
    do_retrain = parser.getboolean('common', 'retrain')
    if do_retrain and os.path.exists(workspace):
      shutil.rmtree(workspace)

  def requires(self):
    reqs = {}
    reqs['net'] = Net(conf=self.conf)
    for set_name in self.test_sets:
      reqs[set_name] = EvalOnTestset(conf=self.conf, test_set=set_name)
    return reqs

  def output(self):
    return luigi.LocalTarget(self.path_end) 
    
  def run(self):
    with open(self.path_end, 'w') as fd_end:
      print >>fd_end, 'trian over'
    print '=====TrainEval over======'
    return

 
if __name__ == "__main__":
    luigi.run()
