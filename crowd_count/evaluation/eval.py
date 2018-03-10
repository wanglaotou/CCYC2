#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, inspect, argparse, shutil, cv2
pfolder = os.path.realpath(os.path.abspath (os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"..")))
if pfolder not in sys.path:
  sys.path.insert(0, pfolder)
reload(sys)
sys.setdefaultencoding('utf8')
from ConfigParser import SafeConfigParser
from util.model import *
import numpy as np

LEN_LINE_SUMMARIZE = 8

def move_summarize_to_head(file_result):
    with open(file_result, 'r') as fr_in:
        lns = fr_in.readlines()
    lns_summ = lns[-LEN_LINE_SUMMARIZE:]
    for i,ln in enumerate(lns_summ):
        lns.insert(i, ln)
    with open(file_result, 'w') as fw_out:
        fw_out.writelines(lns)

def make_result_img(input_img, gt_dmap, density_map, gt_count, est_count):
    if len(input_img.shape) == 3:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    if gt_dmap is not None:
        if np.max(gt_dmap) != 0:
            gt_dmap = 255*gt_dmap/np.max(gt_dmap)
        if gt_dmap.shape != input_img.shape:
            gt_dmap = cv2.resize(gt_dmap, (input_img.shape[1], input_img.shape[0]))
    if np.max(density_map) != 0:
        density_map = 255*density_map/np.max(density_map)
    if density_map.shape != input_img.shape:
        density_map = cv2.resize(density_map, (input_img.shape[1], input_img.shape[0]))
    
    if gt_dmap is not None:    
        res_img = np.hstack((input_img, gt_dmap, density_map))
    else:
        res_img = np.hstack((input_img, density_map))

    text_arr = np.zeros((res_img.shape[0]/2, res_img.shape[1]))
    result_img = np.vstack((res_img, text_arr))
    cv2.putText(result_img,'gt:'+str(round(gt_count, 3)), (1,res_img.shape[0]+20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    if gt_dmap is None:
        cv2.putText(result_img,'est:'+str(round(est_count, 3)), (input_img.shape[1],res_img.shape[0]+20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    else:
        cv2.putText(result_img,'est:'+str(round(est_count, 3)), (2*input_img.shape[1],res_img.shape[0]+20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    return result_img

def show_results(input_img, gt_dmap, density_map):
    if len(input_img.shape) == 3:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    if np.max(gt_dmap) != 0:
        gt_dmap = 255*gt_dmap/np.max(gt_dmap)
    if np.max(density_map) != 0:
        density_map = 255*density_map/np.max(density_map)
    if density_map.shape[1] != input_img.shape[1]:
         input_img = cv2.resize(input_img, (density_map.shape[1], density_map.shape[0]))
    result_img = np.hstack((input_img, gt_dmap, density_map))
    result_img  = result_img.astype(np.uint8, copy=False)
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)


def img_process(im, net_name, is_test_half=False):
    if is_test_half == True:
        im = cv2.resize(im, (im.shape[1]/2, im.shape[0]/2))
    else:
        if(im.shape[1] > 1300 or im.shape[0] > 1300):
            im = cv2.resize(im, (im.shape[1]/2, im.shape[0]/2))

    if net_name =='mcnn':
        im = cv2.resize(im, (im.shape[1]/4*4, im.shape[0]/4*4))
    elif net_name == 'sacnn':
        im = cv2.resize(im, (im.shape[1]/16*16, im.shape[0]/16*16))
    else:
        im = im

    # if is_mean_value == True:
    #     im = np.array(im, dtype = np.float32)
    #     im -= 127.5
    #     im *= 0.0078125

    if len(im.shape) == 2:
        im = np.reshape(1, im.shape[0], im.shape[1])

    return im


def caffe_forward(model, input_img, if_do_mean_scale, mean_value, scale):
    ## 1> set net img size
    model.net.blobs['data'].reshape(1, input_img.shape[2], input_img.shape[0], input_img.shape[1])
    ## 2> Transformer
    transformer = model.caffe.io.Transformer({'data': model.net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2, 0, 1))
    if if_do_mean_scale:
        mean_data = np.array(mean_value)
        transformer.set_mean('data', mean_data)
        transformer.set_input_scale('data', scale)       
    ## 3> preprocess and Load transformer img to net memory
    model.net.blobs['data'].data[...] = transformer.preprocess('data', input_img)
    ## 4> infer
    out = model.net.forward()
    estcount = model.net.blobs['estcount'].data[0]      
    ## 5> save
    feat = model.net.blobs['estdmap'].data[0]
    estdmap = np.reshape(feat, (feat.shape[1], feat.shape[2]))

    return estcount, estdmap


def get_roi(roipath, h, w):
    roi_list = []
    if(os.path.isfile(roipath)):
        with open(roipath, 'r')as rf:
            for (num, value) in enumerate(rf):
                if num != 0:
                    pts = value.strip().split()
                    pt_x = int(pts[0])
                    pt_y = int(pts[1])
                    roi_list.append([pt_x, pt_y])
    roi_arr = np.array(roi_list)
    x_arr = roi_arr[:,0]
    y_arr = roi_arr[:,1]
    x_min = max(x_arr.min(), 0)
    x_max = min(x_arr.max(), w)
    y_min = max(y_arr.min(), 0)
    y_max = min(y_arr.max(), h)
    return x_min,x_max,y_min,y_max

def get_gt_dmap(dmappath):
    dmap = []
    if os.path.exists(dmappath):
        with open(dmappath, 'r') as filedmap:
            line_id = 0
            while True:
                lines = filedmap.readline()
                if not lines:
                    break
                if line_id !=0:
                    p_tmp = [float(i) for i in lines.strip().split(' ')]
                    dmap.append(p_tmp)
                line_id +=1    
        gt_dmap = np.array(dmap).astype(np.float)
        return gt_dmap
    else:
        return None


def fileMap(conf, path_data, path_annot, path_fn_save, path_fd_dmap, mre_thre, is_src=False, if_do_mean_scale=False, mean_value=[127.5, 127.5, 127.5], 
            scale=0.0078125, is_test_half=False, is_test_roi=False,  is_save_dmap=False, is_show=False):
    total_MAE = 0.0
    total_MSE = 0.0
    total_MRE = 0.0
    total_roi_MAE = 0.0
    total_roi_MSE = 0.0
    total_roi_MRE = 0.0
    total_gtcount = 0
    total_estcount = 0.0
    
    save = file(path_fn_save, "w")
    save.write("#image  #gtcount  #estcount  #abs(gtcount - estcount)  #MRE    #roi_est   #roi_MAE  #roi_MRE " + "\n")
    if is_save_dmap == True:
        if not os.path.exists(path_fd_dmap):
            os.mkdir(path_fd_dmap)
        if mre_thre>0.0:
            path_fd_dmap_err = path_fd_dmap + "_err"
            if not os.path.exists(path_fd_dmap_err):
                os.mkdir(path_fd_dmap_err)

    model = Model(conf)
    with open(path_annot, 'r') as fm:
        map_list = fm.readlines()
        test_num = len(map_list)

        for m_line in map_list:
            pathinx = m_line.strip().split()
            imgpath = os.path.join(path_data, pathinx[0])
            print imgpath
            if not os.path.isfile(imgpath):
                print 'path wrong:',imgpath
                sys.exit()

            ## 1> img_infer
            img = cv2.imread(imgpath)
            im = img_process(img, model.net_name, is_test_half)
            estcount, estdmap = caffe_forward(model, im, if_do_mean_scale, mean_value, scale)

            ## 3> save txt
            gtcount = float(pathinx[1])
            total_gtcount = total_gtcount + gtcount
            mae = round(estcount - gtcount, 3)
            mre = round((np.abs(gtcount - estcount) / (gtcount + 1)), 3)
            save.write(str(pathinx[0])+": "+ str(gtcount)+"  ")
            save.write(str(round(estcount, 3)) + "  ")
            save.write(str(mae) + "  ")
            save.write(str(mre) + "  ")

            ## 1.1 > roi infer
            if is_test_roi == True:
                if is_src == True:
                    # read roiImg
                    scene_name = pathinx[0].split('/')[1]
                    roi_name = "roi_" + scene_name[-2:] + ".txt"
                    roi_partpath = os.path.join('ROI',os.path.join(scene_name, roi_name))
                    roipath = os.path.join(path_data, roi_partpath)
                    print 'roiFile:', roiFile
                    x_min,x_max,y_min,y_max = get_roi(roipath, img.shape[0], img.shape[1])
                    roi_img = img[y_min:y_max, x_min:x_max]
                    roi_im = img_process(roi_img, model.net_name, is_test_half)
                    roi_est, roi_estdmap = caffe_forward(model.net, roi_im, if_do_mean_scale, mean_value, scale)

                    # print 'roi_estcount:',roi_est
                    roi_mae = round(roi_est-gtcount, 3)
                    roi_mre = round((np.abs(gtcount - roi_est) / (gtcount+1)),3)
                    save.write(str(round(roi_est, 3)+"  "))
                    save.write(str(roi_mae)+"  ")
                    save.write(str(roi_mre))

                    total_roi_MAE = total_roi_MAE + np.abs(gtcount - roi_est)
                    total_roi_MSE = total_roi_MSE + np.square(gtcount - roi_est)
                    total_roi_MRE = total_roi_MRE + (np.abs(gtcount - roi_est) / (gtcount+1))
                else:
                    pass
            save.write("\n")

            ## show or save
            if is_show or is_save_dmap:
                imgpath_inx = imgpath.split('/')
                imgname_inx = imgpath_inx[-1].split('.')
                gt_dmapth = imgpath.replace(imgname_inx[-1], 'txt')
                if model.net_name == 'mcnn':
                    if is_src:
                        gt_dmapth = gt_dmapth.replace('Scene', 'Dmap/Dmap4_crowd1_5_src_test')
                    else:
                        gt_dmapth = gt_dmapth.replace('RoiImg', 'Dmap/Dmap4_crowd1_5_test')
                elif model.net_name == 'sacnn':
                    if is_src:
                        gt_dmapth = gt_dmapth.replace('Scene', 'Dmap/Dmap8_crowd1_5_src_test')
                    else:
                        gt_dmapth = gt_dmapth.replace('RoiImg', 'Dmap/Dmap8_crowd1_5_test')
                else:
                    print 'net_name error'
                    sys.exit()

                gtdmap = get_gt_dmap(gt_dmapth)
                result_img = make_result_img(img, gtdmap, estdmap, gtcount, estcount)
                if is_show==True:
                    result_img  = result_img.astype(np.uint8, copy=False)
                    cv2.imshow('Result', result_img)
                    cv2.waitKey(0)

                ## 1.2> dmap save
                if is_save_dmap ==True:
                    imgpath_new = os.path.join(path_fd_dmap, imgpath_inx[-2])
                    if not os.path.exists(imgpath_new):
                        os.mkdir(imgpath_new)
                    fname = imgname_inx[0]+'.png'
                    cv2.imwrite(os.path.join(imgpath_new, fname), result_img)

                    if mre >= mre_thre:
                        imgpath_new = os.path.join(path_fd_dmap_err, imgpath_inx[-2])
                        if not os.path.exists(imgpath_new):
                            os.mkdir(imgpath_new)
                        cv2.imwrite(os.path.join(imgpath_new, fname), result_img)

            total_estcount = total_estcount + estcount
            total_MAE = total_MAE + np.abs(gtcount - estcount)
            total_MSE = total_MSE + np.square(gtcount - estcount)
            total_MRE = total_MRE + (np.abs(gtcount - estcount) / (gtcount+1))

        cv2.destroyAllWindows()

        MAE = total_MAE / test_num
        MSE = np.sqrt(total_MSE / test_num)
        MRE = total_MRE / test_num
        save.write("total_gtcount = %.2f\n"%(total_gtcount))
        save.write("total_MAE = %.2f\n"%(total_MAE))
        save.write("MAE = %.2f\n"%(MAE))
        save.write("MSE = %.2f\n"%(MSE))
        save.write("MRE = %.2f\n"%(MRE))
        save.write("total_roi_MAE = %.2f\n"%(total_roi_MAE))
        save.write("total_roi_MSE = %.2f\n"%(total_roi_MSE))
        save.write("total_roi_MRE = %.2f\n"%(total_roi_MRE))
        save.close()

    move_summarize_to_head(path_fn_save)

if __name__ == "__main__":
    print 'test'
    parser = argparse.ArgumentParser(description='evaluation performace of crowd counting')
    parser.add_argument('-annot', type=str, help='path to annotation for evaluation', \
                      default='/ssd/wangmaorui/data/crowd1_5_test.txt')
    parser.add_argument('-data', type=str, help='path to image for evaluation',\
                      default='/ssd/wangmaorui/data')
    parser.add_argument('-conf', type=str, required=True, help='path to conf file')
    parser.add_argument('-o', type=str, help='path to results face detection', default='./results')
    parser.add_argument('-thre', type=float, help='MRE thre to save', default=0.1)
    parser.add_argument('-issrc', type=int, help='test the results with srcimg or roiimg', default=0)
    parser.add_argument('-if_ms', type=int, help='if do -mean*scale process', default=0)
    parser.add_argument('-mean', type=float, nargs=3, help='img meanvalue', default=[127.5,127.5,127.5])
    parser.add_argument('-scale', type=float, help='img scale', default=0.0078125)
    parser.add_argument('-half', type=int, help='test the results with resize half or not', default=0)
    parser.add_argument('-roi', type=int, help='test the results with roi or not', default=0)
    parser.add_argument('-show', type=int, help='show the results or not', default=0)
    parser.add_argument('-save', type=int, help='save the results or not', default=0)
    args = parser.parse_args()

    conf = args.conf
    path_res   = args.o
    path_data  = args.data
    path_annot = args.annot
    mre_thre = args.thre

    is_src = args.issrc
    if_do_mean_scale  = args.if_ms
    mean_value = args.mean
    scale = args.scale
    is_test_half = args.half
    is_test_roi = args.roi
    is_show = args.show
    is_save_dmap = args.save

    path_fn_save = path_res
    path_fd_dmap = path_res+'_dmap'

    fileMap(conf, path_data, path_annot, path_fn_save, path_fd_dmap, mre_thre, is_src, if_do_mean_scale, mean_value, scale,
            is_test_half, is_test_roi, is_save_dmap, is_show)

