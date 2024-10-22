#!/users/lburmist/miniconda/envs/pyeventio/bin/python
# -*- coding: utf-8 -*-

#/home/burmist/miniconda/envs/eventio/bin/python

from eventio import SimTelFile
import pandas as pd
import numpy as np
import pickle as pkl
import sys
import time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

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

def print_ev_first_ID( datafilein = "../simtel_data/proton/data/corsika_run1.simtel.gz", n_ev=10):
    sf = SimTelFile(datafilein)
    ev_counter = 0
    for ev in sf:
        print("event_id         ", ev['event_id'])
        ev_counter=ev_counter+1
        if (ev_counter >= n_ev and n_ev > 0):
            break
    sf.close()
    
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

def get_number_of_channels_wf_number_of_time_points(simtelIn):
    sf = SimTelFile(simtelIn)
    for ev in sf:
        #print("wfshape          ", ev['telescope_events'][1]['adc_samples'][0].shape)
        number_of_time_points=ev['telescope_events'][1]['adc_samples'][0].shape[1]
        number_of_channels=ev['telescope_events'][1]['adc_samples'][0].shape[0]
        break
    
    sf.close()

    return number_of_channels, number_of_time_points

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

def def_clusters_info( event_ID = -999, n_digitalsum_points=0, n_clusters=0, n_points=0,
                       x_mean=-999.0, y_mean=-999.0, t_mean=-999.0,
                       channelID=-999, timeID=-999):
    clusters_info={'event_ID':event_ID,
                   'n_digitalsum_points':n_digitalsum_points,
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

def get_digital_sum_threshold( digitalsum, thresholds, number_of_digsum_micro_clusters_in_camera):
    th=np.array([np.sum(digitalsum>th) for th in thresholds])
    ths=thresholds[th<number_of_digsum_micro_clusters_in_camera]
    if len(ths)>0:
        return int(ths[0])
    return -999

def save_digital_sum_threshold_hist( data, thresholds, pdf_file_out):
    fig, ax = plt.subplots()
    ax.hist(data, bins=np.linspace(6500, 6550, num=200), edgecolor='black')
    #ax.hist(data, bins=thresholds, edgecolor='black')
    #ax.hist(data, edgecolor='black')
    # Add labels and title
    plt.xlabel('threshold')
    #plt.ylabel('')
    #plt.title('npe')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(True)
    # Show the plot
    #plt.show()
    ax.get_figure().savefig(pdf_file_out)
    ax.clear();
    ax.remove();
    plt.close('all');
    
def evtloop(datafilein, nevmax, pixel_mapping_extended, flower_pixID, plot_optimizing_digi_sum_threshold):
    #
    sf = SimTelFile(datafilein)
    wf=np.array([], dtype=np.uint16)
    ev_counter=0
    #
    tic = time.time()
    toc = time.time()
    it_cout = 0
    #
    digital_sum_threshold=[]
    thresholds=np.linspace(6000, 7000, num=1000)
    #
    event_info_list=[]
    clusters_info_list=[]
    #
    for ev in sf:
        #
        wf=ev['telescope_events'][1]['adc_samples'][0]
        #
        try:
            digitalsum=digi_sum(wf=wf, digi_sum_shape=flower_pixID, digi_sum_time_window=3)
        except:
            digitalsum=np.zeros(wf.shape)
        #
        if plot_optimizing_digi_sum_threshold:
            digital_sum_threshold.append(get_digital_sum_threshold( digitalsum=digitalsum, thresholds=thresholds, number_of_digsum_micro_clusters_in_camera=750))
        #
        #
        try:
            #digitalsum_threshold = 6514 NSB @ 386MHz
            #digitalsum_threshold = 6481 NSB @ 268MHz
            clusters_info = get_DBSCAN_clusters( digitalsum = digitalsum, pixel_mapping_extended = pixel_mapping_extended,
                                                 time_norm = 0.05, digitalsum_threshold = 6481,
                                                 DBSCAN_eps = 0.1, DBSCAN_min_samples = 15)
        except:
            clusters_info = def_clusters_info()
            #clusters=np.empty(shape=(0,), dtype=int)
            #clusters=np.array([],dtype=int)
        #
        #
        clusters_info['event_ID'] = int(ev['event_id'])
        clusters_info_list.append(clusters_info)
        #
        event_info_list.append([ev['event_id'],
                                ev['mc_shower']['energy'],
                                ev['mc_shower']['azimuth'],
                                ev['mc_shower']['altitude'],
                                ev['mc_shower']['h_first_int'],
                                ev['mc_shower']['xmax'],
                                ev['mc_shower']['hmax'],
                                ev['mc_shower']['emax'],
                                ev['mc_shower']['cmax'],
                                ev['mc_event']['xcore'],
                                ev['mc_event']['ycore'],
                                ev['telescope_events'][1]['header']['readout_time'],
                                len(ev['photons'][0]),
                                ev['photoelectrons'][0]['n_pe'],
                                (ev['photoelectrons'][0]['n_pixels']-np.sum(ev['photoelectrons'][0]['photoelectrons']==0)),
                                clusters_info['event_ID'],
                                clusters_info['n_digitalsum_points'],
                                clusters_info['n_clusters'],
                                clusters_info['n_points'],
                                clusters_info['x_mean'],
                                clusters_info['y_mean'],
                                clusters_info['t_mean'],
                                clusters_info['channelID'],
                                clusters_info['timeID']])                               
        
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
    
    if plot_optimizing_digi_sum_threshold:
        save_digital_sum_threshold_hist(np.array(digital_sum_threshold),thresholds, "digital_sum_threshold.pdf")
        print( "digital_sum_threshold = ", np.mean(np.array(digital_sum_threshold)))
        #thresholds=np.linspace(6000, 7000, num=100)

    return  event_info_list, clusters_info_list
        
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

def test_channelID_timeID_getters( pixel_mapping_csv, number_of_time_points):
    time_norm = 0.05
    pixel_mapping = np.genfromtxt(pixel_mapping_csv)
    #
    x_val=0
    y_val=0
    t_val_not_norm=0
    #
    t_val=t_val_not_norm*time_norm
    print("----------")
    print("x         ", x_val)
    print("y         ", y_val)
    print("t         ", t_val_not_norm)
    print("channelID ", get_channelID_from_x_y(pixel_mapping_extended=extend_pixel_mapping(pixel_mapping),
                                               x_val=x_val, y_val=y_val))
    print("timeID    ", get_timeID( number_of_time_points=number_of_time_points,
                                    time_norm=time_norm, t_val=t_val))    
    #
    x_val = -0.328050
    y_val =  0.694470
    t_val_not_norm=37
    #
    t_val=t_val_not_norm*time_norm
    print("----------")
    print("true chID ", 2139)
    print("x         ", x_val)
    print("y         ", y_val)
    print("t         ", t_val_not_norm)
    print("channelID ", get_channelID_from_x_y(pixel_mapping_extended=extend_pixel_mapping(pixel_mapping),
                                               x_val=x_val, y_val=y_val))
    print("timeID    ", get_timeID( number_of_time_points=number_of_time_points,
                                    time_norm=time_norm, t_val=t_val))    
    #
    x_val = 1.044900
    y_val = 0.126270
    t_val_not_norm=50
    #
    t_val=t_val_not_norm*time_norm
    print("----------")
    print("true chID ", 7969)
    print("x         ", x_val)
    print("y         ", y_val)
    print("t         ", t_val_not_norm)
    print("channelID ", get_channelID_from_x_y(pixel_mapping_extended=extend_pixel_mapping(pixel_mapping),
                                               x_val=x_val, y_val=y_val))
    print("timeID    ", get_timeID( number_of_time_points=number_of_time_points,
                                    time_norm=time_norm, t_val=t_val))    
    #
    x_val = 0.558900
    y_val = 0.883870
    t_val_not_norm=47
    #
    t_val=t_val_not_norm*time_norm
    print("----------")
    print("true chID ", 1151)
    print("x         ", x_val)
    print("y         ", y_val)
    print("t         ", t_val_not_norm)
    print("channelID ", get_channelID_from_x_y(pixel_mapping_extended=extend_pixel_mapping(pixel_mapping),
                                               x_val=x_val, y_val=y_val))
    print("timeID    ", get_timeID( number_of_time_points=number_of_time_points,
                                    time_norm=time_norm, t_val=t_val))

def mearge_and_save_data( event_info_list, clusters_info_list, headeroutpkl, headeroutcsv):
    #print(len(event_info_list))
    #print(len(clusters_info_list))
    event_info_arr=np.array(event_info_list)
    #print(len(event_info_arr))
    #print(event_info_arr.shape)
    clusters_info_arr=np.array(clusters_info_list)
    pkl.dump(event_info_arr, open(headeroutpkl, "wb"), protocol=pkl.HIGHEST_PROTOCOL)    
    df = pd.DataFrame({'event_id': event_info_arr[:,0], 
                       'energy': event_info_arr[:,1],
                       'azimuth': event_info_arr[:,2],
                       'altitude': event_info_arr[:,3],
                       'h_first_int': event_info_arr[:,4],
                       'xmax': event_info_arr[:,5],
                       'hmax': event_info_arr[:,6],
                       'emax': event_info_arr[:,7],
                       'cmax': event_info_arr[:,8],
                       'xcore': event_info_arr[:,9],
                       'ycore': event_info_arr[:,10],
                       'ev_time': event_info_arr[:,11],
                       'nphotons': event_info_arr[:,12],
                       'n_pe': event_info_arr[:,13],
                       'n_pixels': event_info_arr[:,14],
                       'cluster_event_ID': event_info_arr[:,15],
                       'n_digitalsum_points': event_info_arr[:,16],
                       'n_clusters': event_info_arr[:,17],
                       'n_points': event_info_arr[:,18],
                       'x_mean': event_info_arr[:,19],
                       'y_mean': event_info_arr[:,20],
                       't_mean': event_info_arr[:,21],
                       'channelID': event_info_arr[:,22],
                       'timeID': event_info_arr[:,23]})
    df.to_csv(headeroutcsv)
    print("mean(event_id - cluster_event_ID) : ", np.mean(event_info_arr[:,0]-event_info_arr[:,15]))
    print("std(event_id - cluster_event_ID)  : ", np.std(event_info_arr[:,0]-event_info_arr[:,15]))
    
def main():
    pass
    
if __name__ == "__main__":
    if (len(sys.argv)==7 and (str(sys.argv[1]) == "-d")):
        #
        simtelIn = str(sys.argv[2])
        headeroutpkl = str(sys.argv[3])
        headeroutcsv = str(sys.argv[4])
        #npecsvIn = str(sys.argv[3])
        pixel_mapping_csv = str(sys.argv[5])
        pixel_mapping_neighbors_csv = str(sys.argv[6])
        #
        print("simtelIn                    = ", simtelIn)
        print("headeroutpkl                = ", headeroutpkl)
        print("headeroutcsv                = ", headeroutcsv)
        #print("npecsvIn                    = ", npecsvIn)
        print("pixel_mapping_csv           = ", pixel_mapping_csv)
        print("pixel_mapping_neighbors_csv = ", pixel_mapping_neighbors_csv)
        #
        #print_ev_info()
        #
        #df_pne = pd.read_csv(npecsvIn)
        pixel_mapping = np.genfromtxt(pixel_mapping_csv)
        pixel_mapping_extended = extend_pixel_mapping(pixel_mapping) 
        flower_pixID = get_flower_pixID(pixel_mapping_neighbors_csv)
        #
        event_info_list, clusters_info_list = evtloop( datafilein=simtelIn, nevmax=-1,
                                                       pixel_mapping_extended=pixel_mapping_extended, flower_pixID=flower_pixID,
                                                       plot_optimizing_digi_sum_threshold=False)
        #
        mearge_and_save_data( event_info_list, clusters_info_list, headeroutpkl, headeroutcsv)
        #        
        #
    elif (len(sys.argv)==4 and (str(sys.argv[1]) == "--test_getters")):
        simtelIn = str(sys.argv[2])
        pixel_mapping_csv = str(sys.argv[3])
        print("simtelIn                    = ", simtelIn)
        print("pixel_mapping_csv           = ", pixel_mapping_csv)
        number_of_channels, number_of_time_points = get_number_of_channels_wf_number_of_time_points(simtelIn)        
        #print("number_of_channels    = ", number_of_channels)
        #print("number_of_time_points = ", number_of_time_points)
        test_channelID_timeID_getters( pixel_mapping_csv=pixel_mapping_csv, number_of_time_points=number_of_time_points)
