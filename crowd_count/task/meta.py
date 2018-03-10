#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, inspect, shutil, random, luigi, cv2
from ConfigParser import SafeConfigParser
pfolder = os.path.realpath(os.path.abspath (os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"..")))
if pfolder not in sys.path:
	sys.path.insert(0, pfolder)
reload(sys)
sys.setdefaultencoding('utf8')

class ExternalSampleTrain(luigi.ExternalTask):
  conf = luigi.Parameter()
  
  def __init__(self, *args, **kwargs):
    luigi.ExternalTask.__init__(self, *args, **kwargs)
    parser = SafeConfigParser()
    parser.read(self.conf)
    self.path = parser.get("meta", "samples_train")

  def output(self):
    return luigi.LocalTarget(self.path)
    
class ExternalRootTrain(luigi.ExternalTask):
  conf = luigi.Parameter()
  
  def __init__(self, *args, **kwargs):
    luigi.ExternalTask.__init__(self, *args, **kwargs)
    parser = SafeConfigParser()
    parser.read(self.conf)
    self.path = parser.get("meta", "root_train")

  def output(self):
    return luigi.LocalTarget(self.path)

class ExternalSampleValidate(luigi.ExternalTask):
  conf = luigi.Parameter()
  
  def __init__(self, *args, **kwargs):
    luigi.ExternalTask.__init__(self, *args, **kwargs)
    parser = SafeConfigParser()
    parser.read(self.conf)
    self.path = parser.get("meta", "samples_validate")

  def output(self):
    return luigi.LocalTarget(self.path)

class ExternalRootValidate(luigi.ExternalTask):
  conf = luigi.Parameter()
  
  def __init__(self, *args, **kwargs):
    luigi.ExternalTask.__init__(self, *args, **kwargs)
    parser = SafeConfigParser()
    parser.read(self.conf)
    self.path = parser.get("meta", "root_validate")

  def output(self):
    return luigi.LocalTarget(self.path)    


if __name__ == "__main__":
    luigi.run()
