#!/usr/bin.python
# -*- coding: utf-8 -*-

import os, sys, inspect, shutil, random
from ConfigParser import SafeConfigParser
from util.caffe_exe import *
pfolder = os.path.realpath(os.path.abspath (os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"..")))
if pfolder not in sys.path:
        sys.path.insert(0, pfolder)
reload(sys)
sys.setdefaultencoding('utf8')

class Model:
  def __init__(self, conf):
    self.parser = SafeConfigParser()
    self.parser.read(conf)
    self.workspace = self.parser.get("basic", "workspace")
    caffe_root = self.parser.get("basic", "caffe_root") 
    self.net_name   = self.parser.get('basic', 'net_name')
    gpu_id_set = self.parser.getint('common', 'gpu_id')    
      
    self.caffe = import_caffe(caffe_root)
    self.caffe.set_mode_gpu()
    if gpu_id_set<=-1:
      gpu_id = auto_select_gpu()
      if gpu_id < 0:
        raise Exception('No GPU card')
      else:
        print "select gpu#%d" % gpu_id
    else:
      gpu_id = gpu_id_set
      print "use gpu#%d" % gpu_id
    self.caffe.set_device(gpu_id)

    self.net = self.__load_model(self.net_name)

  def __load_model(self, net_name):
    net_root = os.path.join(self.workspace, net_name)
    proto = os.path.join(net_root, 'proto', 'deploy.prototxt')
    caffe_model = os.path.join(net_root, 'model', '%s.caffemodel' % net_name)
    if os.path.exists(proto) and os.path.exists(caffe_model):
      return self.caffe.Net(proto, caffe_model, self.caffe.TEST)
    else:
      return None

