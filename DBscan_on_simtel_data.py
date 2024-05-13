#!/home/burmist/miniconda/envs/eventio/bin/python
# -*- coding: utf-8 -*-

from eventio import SimTelFile
import pandas as pd
import numpy as np
import pickle as pkl
import sys
import time
from sklearn.cluster import DBSCAN

def print_ev(event):
    print("----------------------------------")
    print("event_id         ", event['event_id'])
    print("energy           ", event['mc_shower']['energy'])
    print("xcore            ", event['mc_event']['xcore'])
    print("ycore            ", event['mc_event']['ycore'])
    print("ev_time          ", event['telescope_events'][1]['header']['readout_time'])
    print("nphotons         ", len(event['photons'][0]))
    print("n_pe             ", event['photoelectrons'][0]['n_pe'])
    print("n_pixels         ", (event['photoelectrons'][0]['n_pixels']-np.sum(event['photoelectrons'][0]['photoelectrons']==0)))
    print("wfshape          ", event['telescope_events'][1]['adc_samples'][0].shape)
    print("type             ", type(event['telescope_events'][1]['adc_samples'][0]))
    print("typeadc          ", type(event['telescope_events'][1]['adc_samples'][0][0][0]))
    print("----------------------------------")

def print_ev_info( datafilein = "../simtel_data/proton/data/corsika_run1.simtel.gz", evID=1240202):
    sf = SimTelFile(datafilein)
    for ev in sf:
        if (int(ev['event_id']) == int(evID)):
            print("----------------------------------")
            print("event_id         ", ev['event_id'])
            print("energy           ", ev['mc_shower']['energy'])
            print("xcore            ", ev['mc_event']['xcore'])
            print("ycore            ", ev['mc_event']['ycore'])
            print("ev_time          ", ev['telescope_events'][1]['header']['readout_time'])
            print("nphotons         ", len(ev['photons'][0]))
            print("n_pe             ", ev['photoelectrons'][0]['n_pe'])
            print("n_pixels         ", (ev['photoelectrons'][0]['n_pixels']-np.sum(ev['photoelectrons'][0]['photoelectrons']==0)))
            print("wfshape          ", ev['telescope_events'][1]['adc_samples'][0].shape)
            print("type             ", type(ev['telescope_events'][1]['adc_samples'][0]))
            print("typeadc          ", type(ev['telescope_events'][1]['adc_samples'][0][0][0]))
            print("----------------------------------")
            break
    
    sf.close()

def get_pixel_mapping(datafilein = "../simtel_data/proton/data/corsika_run1.simtel.gz", outmap_csv = 'pixel_mapping.csv'):
    sf = SimTelFile(datafilein)
    #
    n_pixels=float(sf.telescope_descriptions[1]['camera_organization']['n_pixels'])
    n_drawers=float(sf.telescope_descriptions[1]['camera_organization']['n_drawers'])
    pixel_size=float(sf.telescope_descriptions[1]['camera_settings']['pixel_size'][0])
    #
    the_map=np.concatenate((sf.telescope_descriptions[1]['camera_settings']['pixel_x'].reshape(int(n_pixels),1),
                            sf.telescope_descriptions[1]['camera_settings']['pixel_y'].reshape(int(n_pixels),1),
                            sf.telescope_descriptions[1]['camera_organization']['drawer'].reshape(int(n_pixels),1)), axis=1)
    np.savetxt(outmap_csv, the_map, delimiter=' ',fmt='%f')
    #
    print('n_pixels   = ', int(n_pixels))
    print('n_drawers  = ', int(n_drawers))
    print('pixel_size = ', pixel_size)
    #
    # 0.024300
    # 0.023300
    #
    sf.close()

def digi_sum(wf, digi_sum_shape, digi_sum_time_window):
    #digi_sum in time
    if digi_sum_time_window!=3 :
        return 0
    wfp=wf.copy()
    wfp=np.pad(wfp, pad_width=1)
    wfpsl = wfp.copy()
    wfpsr = wfp.copy()
    wfpsl=np.roll(wfpsl, shift=-1,axis=1)
    wfpsr=np.roll(wfpsr, shift=1,axis=1)
    wfp=wfpsl + wfpsr + wfp
    #print(wf.shape)
    #print(wfp.shape)
    #print(wfp)
    digitalsum=np.array([np.sum(wfp[digi_sum_shape[i]],axis=0) for i in np.arange(0,len(digi_sum_shape))])
    digitalsum=digitalsum[:,1:-1]
    return digitalsum

def get_DBSCAN_clusters( digitalsum, pixel_mapping_extended, time_norm, digitalsum_threshold, DBSCAN_eps, DBSCAN_min_samples):
    #
    pix_x=pixel_mapping_extended[:,0].reshape(pixel_mapping_extended.shape[0],1)
    pix_y=pixel_mapping_extended[:,1].reshape(pixel_mapping_extended.shape[0],1)
    pix_x=np.concatenate(([pix_x for i in np.arange(0,digitalsum.shape[1])]), axis=1)
    pix_y=np.concatenate(([pix_y for i in np.arange(0,digitalsum.shape[1])]), axis=1)
    #
    pix_t=np.array([i for i in np.arange(0,digitalsum.shape[1])]).reshape(1,digitalsum.shape[1])
    pix_t=pix_t*time_norm
    pix_t=np.concatenate(([pix_t for i in np.arange(0,digitalsum.shape[0])]), axis=0)
    #
    pix_x=pix_x[digitalsum>digitalsum_threshold]
    pix_y=pix_y[digitalsum>digitalsum_threshold]
    pix_t=pix_t[digitalsum>digitalsum_threshold]  
    #
    pix_x=np.expand_dims(pix_x,axis=1)
    pix_y=np.expand_dims(pix_y,axis=1)
    pix_t=np.expand_dims(pix_t,axis=1)
    #
    #print(pix_x.shape)
    #print(pix_y.shape)
    #print(pix_t.shape)
    #
    X=np.concatenate((pix_x,pix_y,pix_t), axis=1)
    dbscan = DBSCAN( eps = DBSCAN_eps, min_samples = DBSCAN_min_samples)
    clusters = dbscan.fit_predict(X)
    pointID = np.unique(clusters)
    #
    clusters_info=def_clusters_info()
    #
    clusters_info['n_digitalsum_points'] = len(pix_x)
    #
    if len(pointID) > 1 :
        pointID = pointID[pointID>-1]
        #print(pointID)
        clustersID = np.argmax([len(clusters[clusters==clID]) for clID in pointID])            
        #
        clusters_info['n_clusters'] = len(pointID)
        clusters_info['n_points'] = len(clusters[clusters==clustersID])
        clusters_info['x_mean'] = np.mean(pix_x[clusters==clustersID])
        clusters_info['y_mean'] = np.mean(pix_y[clusters==clustersID])
        clusters_info['t_mean'] = np.mean(pix_t[clusters==clustersID])
        #
        clusters_info['channelID'] = get_channelID_from_x_y( pixel_mapping_extended=pixel_mapping_extended, x_val=clusters_info['x_mean'], y_val=clusters_info['y_mean'])
        clusters_info['timeID'] = get_timeID( number_of_time_points=digitalsum.shape[1], time_norm=time_norm, t_val=clusters_info['t_mean'])
        #
        #clustern = np.max([ for clID in np.unique(clusters)[1:]])
        #clustern = 0
        #print(len(cl_ID))
        #print(clustern)
    #
    return clusters_info

def def_clusters_info( n_digitalsum_points=0, n_clusters=0, n_points=0,
                       x_mean=-999.0, y_mean=-999.0, t_mean=-999.0,
                       channelID=-999, timeID=-999):
    clusters_info={'n_digitalsum_points':n_digitalsum_points,
                   'n_clusters':n_clusters,
                   'n_points':n_points,
                   'x_mean':x_mean,
                   'y_mean':y_mean,
                   't_mean':t_mean,
                   'channelID':channelID,
                   'timeID':timeID}
    #
    return clusters_info    

def get_channelID_from_x_y( pixel_mapping_extended, x_val, y_val):
    delta_dim=0.015
    pm=pixel_mapping_extended[pixel_mapping_extended[:,0] > (x_val-delta_dim)]
    pm=pm[pm[:,0] < (x_val+delta_dim)]
    pm=pm[pm[:,1] > (y_val-delta_dim)]
    pm=pm[pm[:,1] < (y_val+delta_dim)]
    #print(len(pm))
    if len(pm)>0:
        return int(pm[0,3])
    return -999

def get_timeID( number_of_time_points, time_norm, t_val):
    if number_of_time_points > 1:        
        pix_t=np.array([i for i in np.arange(0,number_of_time_points)])
        pix_t=np.abs(pix_t*time_norm-t_val)
        return np.argmin(pix_t)

    return -999

def evtloop(datafilein, npecsvIn, nevmax, pixel_mapping_extended, flower_pixID, df_pne):
    #
    sf = SimTelFile(datafilein)
    wf=np.array([], dtype=np.uint16)
    ev_counter=0
    #
    tic = time.time()
    toc = time.time()
    it_cout = 0
    #
    for ev in sf:
        wf=ev['telescope_events'][1]['adc_samples'][0]
        #digi_sum_and_DBSCAN(wf)
        try:
            digitalsum=digi_sum(wf=wf, digi_sum_shape=flower_pixID, digi_sum_time_window=3)
        except:
            digitalsum=np.zeros(wf.shape)
        #
        try:
            clusters_info = get_DBSCAN_clusters( digitalsum = digitalsum, pixel_mapping_extended = pixel_mapping_extended,
                                                 time_norm = 0.05, digitalsum_threshold = 6505,
                                                 DBSCAN_eps = 0.1, DBSCAN_min_samples = 15)
        except:
            clusters_info = def_clusters_info()
            #clusters=np.empty(shape=(0,), dtype=int)
            #clusters=np.array([],dtype=int)
        #
        #
        #print(clusters_info)
        #
        ev_counter=ev_counter+1
        #print('ev_counter = ', ev_counter)
        #print_ev(ev)
        if (ev_counter >= nevmax and nevmax > 0):
            break
        if (it_cout%1000==0) :
            toc = time.time()
            print('{:10d} {:10.2f} s'.format(it_cout, toc - tic))
            tic = time.time()
        it_cout = it_cout + 1
        
    sf.close()
    
def get_flower_pixID(pixel_mapping_neighbors):
    flower_pixID = np.genfromtxt(pixel_mapping_neighbors,dtype=int)
    flower_pixID=flower_pixID+1
    flower_seedID=np.arange(1, flower_pixID.shape[0]+1, 1).reshape(flower_pixID.shape[0],1)
    flower_pixID=np.concatenate((flower_seedID, flower_pixID), axis=1)
    return flower_pixID

def extend_pixel_mapping(pixel_mapping):
    pixel_mapping_extended = pixel_mapping.copy()
    pix_ID=np.array([i for i in np.arange(0,pixel_mapping.shape[0])])
    pix_ID=np.expand_dims(pix_ID,axis=1)
    pixel_mapping_extended=np.concatenate((pixel_mapping_extended,pix_ID), axis=1)
    #print(pixel_mapping_extended)
    #print(pixel_mapping_extended.shape)
    return pixel_mapping_extended
    
def main():
    pass;
    
if __name__ == "__main__":
    if (len(sys.argv)==5):
        #
        simtelIn = str(sys.argv[1])
        npecsvIn = str(sys.argv[2])
        pixel_mapping_csv = str(sys.argv[3])
        pixel_mapping_neighbors_csv = str(sys.argv[4])
        #
        print("simtelIn                    = ", simtelIn)
        print("npecsvIn                    = ", npecsvIn)
        print("pixel_mapping_csv           = ", pixel_mapping_csv)
        print("pixel_mapping_neighbors_csv = ", pixel_mapping_neighbors_csv)
        #
        #print_ev_info()
        #
        df_pne = pd.read_csv(npecsvIn)
        pixel_mapping = np.genfromtxt(pixel_mapping_csv)
        pixel_mapping_extended = extend_pixel_mapping(pixel_mapping) 
        flower_pixID = get_flower_pixID(pixel_mapping_neighbors_csv)
        #
        evtloop( datafilein=simtelIn, npecsvIn=npecsvIn, nevmax=-1,
                 pixel_mapping_extended=pixel_mapping_extended, flower_pixID=flower_pixID, df_pne=df_pne)

