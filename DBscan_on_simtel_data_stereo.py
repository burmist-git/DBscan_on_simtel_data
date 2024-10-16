#!/home/burmist/miniconda/envs/eventio/bin/python
# -*- coding: utf-8 -*-

#hostname dpnc02
#/home/burmist/miniconda/envs/eventio/bin/python
#hostname daint
#/users/lburmist/miniconda/envs/pyeventio/bin/python

from eventio import SimTelFile
import pandas as pd
import numpy as np
import pickle as pkl
import sys
import time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

###################################
#
#
_n_max_noise_events=1000
_npe_noise=20
_n_of_time_sample=75
_time_of_one_sample_s=(_n_of_time_sample*1000/1024.0*1.0e-9)
_event_modulo=100
#
#
#
_time_norm_isolated=0.09
_DBSCAN_eps_isolated = 0.11
_DBSCAN_digitalsum_threshold_isolated = 2165
_DBSCAN_min_samples_isolated = 3
#
#
#
_time_norm=0.06
_DBSCAN_eps = 0.11
_DBSCAN_digitalsum_threshold = 2160
_DBSCAN_min_samples = 15
#
#
###################################

def print_setup():
    print("_n_max_noise_events                   = ",_n_max_noise_events)
    print("_npe_noise                            = ",_npe_noise)
    print("_time_of_one_sample_s                 = ",_time_of_one_sample_s)
    print("_event_modulo                         = ",_event_modulo)
    print("_time_norm                            = ",_time_norm)
    print("_DBSCAN_eps                           = ",_DBSCAN_eps)
    print("_DBSCAN_digitalsum_threshold_isolated = ",_DBSCAN_digitalsum_threshold_isolated)
    print("_DBSCAN_min_samples_isolated          = ",_DBSCAN_min_samples_isolated)
    print("_DBSCAN_digitalsum_threshold          = ",_DBSCAN_digitalsum_threshold)
    print("_DBSCAN_min_samples                   = ",_DBSCAN_min_samples)
    
def extend_pixel_mapping( pixel_mapping, channel_list, number_of_wf_time_samples):
    pixel_mapping_extended=pixel_mapping[channel_list[:,0]].copy()
    pixel_mapping_extended=pixel_mapping_extended[:,:-1]
    pixel_mapping_extended=np.expand_dims(pixel_mapping_extended,axis=2)
    pixel_mapping_extended=np.swapaxes(pixel_mapping_extended, 1, 2)
    pixel_mapping_extended=np.concatenate(([pixel_mapping_extended for i in np.arange(0,number_of_wf_time_samples)]), axis=1)
    pixt=np.array([i for i in np.arange(0,number_of_wf_time_samples)]).reshape(1,number_of_wf_time_samples)
    pixt=np.concatenate(([pixt for i in np.arange(0,channel_list.shape[0])]), axis=0)
    pixt=np.expand_dims(pixt,axis=2)
    pixel_mapping_extended=np.concatenate((pixel_mapping_extended,pixt), axis=2)
    #print(pixel_mapping_extended)
    #print(pixel_mapping_extended.shape)
    return pixel_mapping_extended

def get_DBSCAN_clusters( digitalsum, pixel_mapping, pixel_mapping_extended, channel_list, time_norm, digitalsum_threshold, DBSCAN_eps, DBSCAN_min_samples):
    #extend_pixel_mapping( pixel_mapping=pixel_mapping, channel_list=channel_list, number_of_wf_time_samples=digitalsum.shape[1])
    #print(digitalsum.shape)
    #print(pixel_mapping.shape)
    #print(print(pixel_mapping_extended.shape)
    #
    clusters_info=def_clusters_info()
    X=pixel_mapping_extended[digitalsum>digitalsum_threshold] 
    X=X*[[1,1,time_norm]]
    dbscan = DBSCAN( eps = DBSCAN_eps, min_samples = DBSCAN_min_samples)
    clusters = dbscan.fit_predict(X)
    clustersID = np.unique(clusters)
    #
    clusters_info['n_digitalsum_points'] = len(X)
    #
    if (len(clustersID) > 1) :
        clustersID = clustersID[clustersID>-1]
        clustersIDmax = np.argmax([len(clusters[clusters==clID]) for clID in clustersID])
        #
        clusters_info['n_clusters'] = len(clustersID)
        clusters_info['n_points'] = len(clusters[clusters==clustersIDmax])
        #
        clusters_info['x_mean'] = np.mean(X[clusters==clustersIDmax][0])
        clusters_info['y_mean'] = np.mean(X[clusters==clustersIDmax][1])
        clusters_info['t_mean'] = np.mean(X[clusters==clustersIDmax][2])
    #    
    #
    return clusters_info
    
def digital_w_sum( wf, digi_sum_channel_list, channels_blacklist):
    #
    digital_sum_result=None
    for i in np.arange(0,len(digi_sum_channel_list)):
        not_contains_any=np.sum(np.isin(digi_sum_channel_list[i],channels_blacklist))
        if (not_contains_any == 0):
            if digital_sum_result is None:
                digital_sum_result=np.sum(wf[digi_sum_channel_list[i]],axis=0)
                digital_sum_result=np.reshape(digital_sum_result,(1,len(digital_sum_result)))
            else:
                digital_sum_result_tmp=np.sum(wf[digi_sum_channel_list[i]],axis=0)
                digital_sum_result_tmp=np.reshape(digital_sum_result_tmp,(1,len(digital_sum_result_tmp)))
                digital_sum_result=np.concatenate((digital_sum_result,digital_sum_result_tmp))
                #
    return digital_sum_result
                
def digital_sum( wf, digi_sum_channel_list):
    digital_sum_result = np.array([np.sum(wf[digi_sum_channel_list[i]],axis=0) for i in np.arange(0,len(digi_sum_channel_list))])
    return digital_sum_result

def def_clusters_info( n_digitalsum_points=0, n_clusters=0, n_points=0,
                       x_mean=-999.0, y_mean=-999.0, t_mean=-999.0,
                       channelID=-999, timeID=-999):
    #
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

def def_L1_trigger_info( max_digi_sum=0,
                         x_mean=-999.0, y_mean=-999.0, t_mean=-999.0,
                         channelID=-999, timeID=-999):
    L1_trigger_info={'max_digi_sum':max_digi_sum,
                     'x_mean':x_mean,
                     'y_mean':y_mean,
                     't_mean':t_mean,
                     'channelID':channelID,
                     'timeID':timeID}
    #
    return L1_trigger_info

def get_L1_trigger_info( digitalsum, pixel_mapping, digi_sum_channel_list):
    max_digi_sum=np.max(digitalsum)
    row, col = np.unravel_index(np.argmax(digitalsum), digitalsum.shape)
    channelID=digi_sum_channel_list[row,0]
    timeID=col
    x_mean=pixel_mapping[channelID,0]
    y_mean=pixel_mapping[channelID,1]
    t_mean=col
    return def_L1_trigger_info( max_digi_sum=max_digi_sum,
                                x_mean=x_mean, y_mean=y_mean, t_mean=t_mean,
                                channelID=channelID, timeID=timeID)

def print_trigger_info(trigger_info):
    print(trigger_info.keys())
    for key in trigger_info.keys() :
        print(key," ",trigger_info[key])
    
def save_analyze_noise( data, pdf_file_out, n_samples, nsigma=12, nbinsfactor=2):
    #
    fig, ax = plt.subplots()
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_min = int(data_mean - nsigma*data_std)
    data_max = int(data_mean + nsigma*data_std)
    #data_min = 280
    #data_max = 330
    #nbins=int(2*nsigma*data_std*nbinsfactor)
    nbins=int(35*nbinsfactor)
    #nbins=11
    hist_data=ax.hist(data, bins=np.linspace(data_min, data_max, num=nbins), edgecolor='black')
    #ax.hist(data, bins=nbins)
    plt.xlabel('ADC value')
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
    #
    pdf_file_out_rates = str(str(pdf_file_out) + '_rates.pdf')
    save_and_analyze_rates_noise( hist_data, pdf_file_out_rates, n_samples)
    
def save_and_analyze_rates_noise( hist_data, pdf_file_out, n_samples):
    counts=hist_data[0]
    thresholds=hist_data[1][:-1]
    #
    print("np.sum(counts) = ",np.sum(counts))
    #
    rates = np.array([np.sum(counts[i:]) for i in np.arange(0,len(counts))])
    rates = rates/(_time_of_one_sample_s*n_samples)
    fig, ax = plt.subplots()
    ax.plot(thresholds, rates, 'bo-', linewidth=2, markersize=4)
    plt.xlabel('thresholds, ADC counts')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(True)
    # Show the plot
    #plt.show()
    ax.get_figure().savefig(pdf_file_out)
    ax.clear();
    ax.remove();
    plt.close('all');
    #
    df = pd.DataFrame({'thresholds': thresholds,
                       'counts': counts,
                       'rates': rates})
    csv_file_out=str(str(pdf_file_out)+str('.csv'))
    df.to_csv(csv_file_out, sep=' ', index=False)
    #
    return    
    
def analyze_noise( wf_noise, L1_digitalsum_noise, L3_digitalsum_noise, L3_digitalsum_noise_all):
    #
    print("analyze_noise")
    #
    print(len(wf_noise))
    print(len(L1_digitalsum_noise))
    print(len(L3_digitalsum_noise))
    print(len(L3_digitalsum_noise_all))
    #
    print(wf_noise[0].shape)
    print(L1_digitalsum_noise[0].shape)
    print(L3_digitalsum_noise[0].shape)
    print(L3_digitalsum_noise_all[0].shape)
    #
    print(_time_of_one_sample_s)
    #
    result = None
    for array in wf_noise:
        if result is None:
            result = array
        else:
            result = np.concatenate((result, array))
    #
    wf_noise_arr = result
    #
    result = None
    for array in L1_digitalsum_noise:
        if result is None:
            result = array
        else:
            result = np.concatenate((result, array))
    #
    L1_digitalsum_noise_arr = result
    #
    result = None
    for array in L1_digitalsum_noise:
        if result is None:
            result = array
        else:
            result = np.concatenate((result, array))
    #
    L3_digitalsum_noise_arr = result
    #
    result = None
    for array in L3_digitalsum_noise_all:
        if result is None:
            result = array
        else:
            result = np.concatenate((result, array))
    #
    L3_digitalsum_noise_all_arr = result
    #
    print(np.mean(wf_noise_arr)," ",np.std(wf_noise_arr))
    print(np.mean(L1_digitalsum_noise_arr)," ",np.std(L1_digitalsum_noise_arr))
    print(np.mean(L3_digitalsum_noise_arr)," ",np.std(L3_digitalsum_noise_arr))
    print(np.mean(L3_digitalsum_noise_all_arr)," ",np.std(L3_digitalsum_noise_all_arr))
    #
    save_analyze_noise( data=wf_noise_arr.flatten(), pdf_file_out="wf_noiseNSB268MHz_arr.pdf", n_samples=len(wf_noise))
    save_analyze_noise( data=L1_digitalsum_noise_arr.flatten(), pdf_file_out="L1_digitalsum_noiseNSB268MHz_arr.pdf", n_samples=len(L1_digitalsum_noise))
    save_analyze_noise( data=L3_digitalsum_noise_arr.flatten(), pdf_file_out="L3_digitalsum_noiseNSB268MHz_arr.pdf", n_samples=len(L3_digitalsum_noise))
    save_analyze_noise( data=L3_digitalsum_noise_all_arr.flatten(), pdf_file_out="L3_digitalsum_noiseNSB268MHz_arr_all.pdf", n_samples=len(L3_digitalsum_noise_all))
    #
    return
    
def evtloop_noise(datafilein, nevmax, pixel_mapping, L1_trigger_pixel_cluster_list, L3_trigger_DBSCAN_pixel_cluster_list, L3_trigger_DBSCAN_pixel_cluster_list_all):
    #
    print("evtloop_noise")
    #
    sf = SimTelFile(datafilein)
    wf=np.array([], dtype=np.uint16)
    ev_counter=0
    #
    tic = time.time()
    toc = time.time()
    it_cout = 0
    #
    wf_trigger_pixel_list=np.array([i for i in np.arange(0,pixel_mapping.shape[0])])
    wf_trigger_pixel_list=np.reshape(wf_trigger_pixel_list,(pixel_mapping.shape[0],1))
    #
    wf_noise_list=[]
    L1_digitalsum_noise_list=[]
    L3_digitalsum_noise_list=[]
    L3_digitalsum_noise_list_all=[]
    #
    for ev in sf:
        #
        LSTID=ev['telescope_events'].keys()
        #
        wf_list=[]
        n_pe_per_tel_list=[]
        LSTID_list=[]
        for i in LSTID :
            wf_list.append(ev['telescope_events'][i]['adc_samples'][0])
            n_pe_per_tel_list.append(int(ev['photoelectrons'][i-1]['n_pe']))
            LSTID_list.append(int(i-1))
        #
        for wf, npe, lst_id in zip( wf_list, n_pe_per_tel_list, LSTID_list) :
            try:
                if npe == _npe_noise:
                    if(len(wf_noise_list)<_n_max_noise_events):
                        wf_noise_list.append(wf)
                        L1_digitalsum_noise_list.append(digital_sum(wf=wf, digi_sum_channel_list=L1_trigger_pixel_cluster_list))
                        L3_digitalsum_noise_list.append(digital_sum(wf=wf, digi_sum_channel_list=L3_trigger_DBSCAN_pixel_cluster_list))
                        L3_digitalsum_noise_list_all.append(digital_sum(wf=wf, digi_sum_channel_list=L3_trigger_DBSCAN_pixel_cluster_list_all))
            except:
                pass
        #
        ev_counter=ev_counter+1
        #print_ev(ev)
        if (ev_counter >= nevmax and nevmax > 0):
            break
        if (it_cout%_event_modulo==0) :
            toc = time.time()
            print('{:10d} {:10.2f} s'.format(it_cout, toc - tic))
            tic = time.time()
        it_cout = it_cout + 1
        
    sf.close()
    #
    print("L1_digitalsum_noise_list          ",len(L1_digitalsum_noise_list))
    print("L1_digitalsum_noise_list[0].shape ",L1_digitalsum_noise_list[0].shape)
    #
    analyze_noise( wf_noise=wf_noise_list,
                   L1_digitalsum_noise=L1_digitalsum_noise_list,
                   L3_digitalsum_noise=L3_digitalsum_noise_list,
                   L3_digitalsum_noise_all=L3_digitalsum_noise_list_all)
    #        
    return

def evtloop(datafilein, nevmax, pixel_mapping, L1_trigger_pixel_cluster_list, L3_trigger_DBSCAN_pixel_cluster_list, L3_trigger_DBSCAN_pixel_cluster_list_all):
    #
    print("evtloop")
    #
    sf = SimTelFile(datafilein)
    wf=np.array([], dtype=np.uint16)
    ev_counter=0
    #
    tic = time.time()
    toc = time.time()
    it_cout = 0
    #
    event_info_list=[]
    #
    pixel_mapping_extended=extend_pixel_mapping( pixel_mapping=pixel_mapping, channel_list=L3_trigger_DBSCAN_pixel_cluster_list, number_of_wf_time_samples=_n_of_time_sample)
    pixel_mapping_extended_all=extend_pixel_mapping( pixel_mapping=pixel_mapping, channel_list=L3_trigger_DBSCAN_pixel_cluster_list_all, number_of_wf_time_samples=_n_of_time_sample)
    #
    for ev in sf:
        #
        # ev['telescope_events'].keys() | [1, 2, 3, 4]
        # ev['photoelectrons'].keys()   | [0, 1, 2, 3]
        #
        LSTID=ev['telescope_events'].keys()
        #print('ev')
        #print(LSTID)
        #
        wf_list=[]
        n_pe_per_tel_list=[]
        LSTID_list=[]
        #
        #L1_trigger_info_list=[]
        #DBSCAN_clusters_info_list=[]
        #DBSCAN_clusters_info_isolated_list=[]
        #
        L1_trigger_info_LST1=None
        L1_trigger_info_LST2=None
        L1_trigger_info_LST3=None
        L1_trigger_info_LST4=None
        DBSCAN_clusters_info_LST1=None
        DBSCAN_clusters_info_LST2=None
        DBSCAN_clusters_info_LST3=None
        DBSCAN_clusters_info_LST4=None
        DBSCAN_clusters_info_isolated_LST1=None
        DBSCAN_clusters_info_isolated_LST2=None
        DBSCAN_clusters_info_isolated_LST3=None
        DBSCAN_clusters_info_isolated_LST4=None
        #
        #
        ev_time=[0,0,0,0]
        nphotons=[0,0,0,0]
        n_pe=[0,0,0,0]
        n_pixels=[0,0,0,0]
        #
        #
        for i in LSTID :
            wf_list.append(ev['telescope_events'][i]['adc_samples'][0])
            n_pe_per_tel_list.append(int(ev['photoelectrons'][i-1]['n_pe']))
            LSTID_list.append(int(i-1))
            #
            #
            ev_time[(i-1)] = float(ev['telescope_events'][i]['header']['readout_time'])
            nphotons[(i-1)]=int(len(ev['photons'][(i-1)]))
            n_pe[(i-1)]=int(ev['photoelectrons'][(i-1)]['n_pe'])
            n_pixels[(i-1)]=int(ev['photoelectrons'][(i-1)]['n_pixels']-np.sum(ev['photoelectrons'][(i-1)]['photoelectrons']==0))
            #
            #
        #
        #
        for i in np.arange(0,len(n_pe)) :
            if (n_pe[i] == 0 and i == 0) :
                L1_trigger_info_LST1 = def_L1_trigger_info()
                DBSCAN_clusters_info_isolated_LST1 = def_clusters_info()
                DBSCAN_clusters_info_LST1 = def_clusters_info()
            elif (n_pe[i] == 0 and i == 1) :
                L1_trigger_info_LST2 = def_L1_trigger_info()
                DBSCAN_clusters_info_isolated_LST2 = def_clusters_info()
                DBSCAN_clusters_info_LST2 = def_clusters_info()
            elif (n_pe[i] == 0 and i == 2) :
                L1_trigger_info_LST3 = def_L1_trigger_info()
                DBSCAN_clusters_info_isolated_LST3 = def_clusters_info()
                DBSCAN_clusters_info_LST3 = def_clusters_info()
            elif (n_pe[i] == 0 and i == 3) :
                L1_trigger_info_LST4 = def_L1_trigger_info()
                DBSCAN_clusters_info_isolated_LST4 = def_clusters_info()
                DBSCAN_clusters_info_LST4 = def_clusters_info()                
        #
        #
        #print("event_id = ", int(ev['event_id']))
        #
        for wf, npe, lst_id in zip( wf_list, n_pe_per_tel_list, LSTID_list) :
            try:                
                #
                L1_digitalsum = digital_sum(wf=wf, digi_sum_channel_list=L1_trigger_pixel_cluster_list)
                L3_digitalsum = digital_sum(wf=wf, digi_sum_channel_list=L3_trigger_DBSCAN_pixel_cluster_list)
                L3_digitalsum_all = digital_sum(wf=wf, digi_sum_channel_list=L3_trigger_DBSCAN_pixel_cluster_list_all)
                #
                L1_trigger_info = get_L1_trigger_info(digitalsum=L1_digitalsum, pixel_mapping=pixel_mapping, digi_sum_channel_list=L1_trigger_pixel_cluster_list)
                #                
                DBSCAN_clusters_info_isolated = get_DBSCAN_clusters( digitalsum = L3_digitalsum,
                                                                     pixel_mapping = pixel_mapping,
                                                                     pixel_mapping_extended = pixel_mapping_extended,
                                                                     channel_list = L3_trigger_DBSCAN_pixel_cluster_list,
                                                                     time_norm = _time_norm_isolated,
                                                                     digitalsum_threshold = _DBSCAN_digitalsum_threshold_isolated,
                                                                     DBSCAN_eps = _DBSCAN_eps_isolated,
                                                                     DBSCAN_min_samples = _DBSCAN_min_samples_isolated)
                #
                DBSCAN_clusters_info = get_DBSCAN_clusters( digitalsum = L3_digitalsum_all,
                                                            pixel_mapping = pixel_mapping,
                                                            pixel_mapping_extended = pixel_mapping_extended_all,
                                                            channel_list = L3_trigger_DBSCAN_pixel_cluster_list_all,
                                                            time_norm = _time_norm,
                                                            digitalsum_threshold = _DBSCAN_digitalsum_threshold,
                                                            DBSCAN_eps = _DBSCAN_eps,
                                                            DBSCAN_min_samples = _DBSCAN_min_samples)
                #
                #
                if (lst_id == 0) :
                    L1_trigger_info_LST1 = L1_trigger_info
                    DBSCAN_clusters_info_isolated_LST1 = DBSCAN_clusters_info_isolated
                    DBSCAN_clusters_info_LST1 = DBSCAN_clusters_info
                elif (lst_id == 1) :
                    L1_trigger_info_LST2 = L1_trigger_info
                    DBSCAN_clusters_info_isolated_LST2 = DBSCAN_clusters_info_isolated
                    DBSCAN_clusters_info_LST2 = DBSCAN_clusters_info
                elif (lst_id == 2) :
                    L1_trigger_info_LST3 = L1_trigger_info
                    DBSCAN_clusters_info_isolated_LST3 = DBSCAN_clusters_info_isolated
                    DBSCAN_clusters_info_LST3 = DBSCAN_clusters_info                    
                elif (lst_id == 3) :
                    L1_trigger_info_LST4 = L1_trigger_info
                    DBSCAN_clusters_info_isolated_LST4 = DBSCAN_clusters_info_isolated
                    DBSCAN_clusters_info_LST4 = DBSCAN_clusters_info
                #
                #
            except:
                if (lst_id == 0) :
                    L1_trigger_info_LST1 = def_L1_trigger_info()
                    DBSCAN_clusters_info_isolated_LST1 = def_clusters_info()
                    DBSCAN_clusters_info_LST1 = def_clusters_info()
                elif (lst_id == 1) :
                    L1_trigger_info_LST2 = def_L1_trigger_info()
                    DBSCAN_clusters_info_isolated_LST2 = def_clusters_info()
                    DBSCAN_clusters_info_LST2 = def_clusters_info()
                elif (lst_id == 2) :
                    L1_trigger_info_LST3 = def_L1_trigger_info()
                    DBSCAN_clusters_info_isolated_LST3 = def_clusters_info()
                    DBSCAN_clusters_info_LST3 = def_clusters_info()
                elif (lst_id == 3) :
                    L1_trigger_info_LST4 = def_L1_trigger_info()
                    DBSCAN_clusters_info_isolated_LST4 = def_clusters_info()
                    DBSCAN_clusters_info_LST4 = def_clusters_info()
            #
            #
        #
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
                                ev_time[0],
                                ev_time[1],
                                ev_time[2],
                                ev_time[3],
                                nphotons[0],
                                nphotons[1],
                                nphotons[2],
                                nphotons[3],
                                n_pe[0],
                                n_pe[1],
                                n_pe[2],
                                n_pe[3],
                                n_pixels[0],
                                n_pixels[1],
                                n_pixels[2],
                                n_pixels[3],
                                L1_trigger_info_LST1['max_digi_sum'],
                                L1_trigger_info_LST1['x_mean'],
                                L1_trigger_info_LST1['y_mean'],
                                L1_trigger_info_LST1['t_mean'],
                                L1_trigger_info_LST1['channelID'],
                                L1_trigger_info_LST1['timeID'],
                                L1_trigger_info_LST2['max_digi_sum'],
                                L1_trigger_info_LST2['x_mean'],
                                L1_trigger_info_LST2['y_mean'],
                                L1_trigger_info_LST2['t_mean'],
                                L1_trigger_info_LST2['channelID'],
                                L1_trigger_info_LST2['timeID'],
                                L1_trigger_info_LST3['max_digi_sum'],
                                L1_trigger_info_LST3['x_mean'],
                                L1_trigger_info_LST3['y_mean'],
                                L1_trigger_info_LST3['t_mean'],
                                L1_trigger_info_LST3['channelID'],
                                L1_trigger_info_LST3['timeID'],
                                L1_trigger_info_LST4['max_digi_sum'],
                                L1_trigger_info_LST4['x_mean'],
                                L1_trigger_info_LST4['y_mean'],
                                L1_trigger_info_LST4['t_mean'],
                                L1_trigger_info_LST4['channelID'],
                                L1_trigger_info_LST4['timeID'],
                                DBSCAN_clusters_info_isolated_LST1['n_digitalsum_points'],
                                DBSCAN_clusters_info_isolated_LST1['n_clusters'],
                                DBSCAN_clusters_info_isolated_LST1['n_points'],
                                DBSCAN_clusters_info_isolated_LST1['x_mean'],
                                DBSCAN_clusters_info_isolated_LST1['y_mean'],
                                DBSCAN_clusters_info_isolated_LST1['t_mean'],
                                DBSCAN_clusters_info_isolated_LST1['channelID'],
                                DBSCAN_clusters_info_isolated_LST1['timeID'],
                                DBSCAN_clusters_info_isolated_LST2['n_digitalsum_points'],
                                DBSCAN_clusters_info_isolated_LST2['n_clusters'],
                                DBSCAN_clusters_info_isolated_LST2['n_points'],
                                DBSCAN_clusters_info_isolated_LST2['x_mean'],
                                DBSCAN_clusters_info_isolated_LST2['y_mean'],
                                DBSCAN_clusters_info_isolated_LST2['t_mean'],
                                DBSCAN_clusters_info_isolated_LST2['channelID'],
                                DBSCAN_clusters_info_isolated_LST2['timeID'],
                                DBSCAN_clusters_info_isolated_LST3['n_digitalsum_points'],
                                DBSCAN_clusters_info_isolated_LST3['n_clusters'],
                                DBSCAN_clusters_info_isolated_LST3['n_points'],
                                DBSCAN_clusters_info_isolated_LST3['x_mean'],
                                DBSCAN_clusters_info_isolated_LST3['y_mean'],
                                DBSCAN_clusters_info_isolated_LST3['t_mean'],
                                DBSCAN_clusters_info_isolated_LST3['channelID'],
                                DBSCAN_clusters_info_isolated_LST3['timeID'],
                                DBSCAN_clusters_info_isolated_LST4['n_digitalsum_points'],
                                DBSCAN_clusters_info_isolated_LST4['n_clusters'],
                                DBSCAN_clusters_info_isolated_LST4['n_points'],
                                DBSCAN_clusters_info_isolated_LST4['x_mean'],
                                DBSCAN_clusters_info_isolated_LST4['y_mean'],
                                DBSCAN_clusters_info_isolated_LST4['t_mean'],
                                DBSCAN_clusters_info_isolated_LST4['channelID'],
                                DBSCAN_clusters_info_LST4['timeID'],                      
                                DBSCAN_clusters_info_LST1['n_digitalsum_points'],
                                DBSCAN_clusters_info_LST1['n_clusters'],
                                DBSCAN_clusters_info_LST1['n_points'],
                                DBSCAN_clusters_info_LST1['x_mean'],
                                DBSCAN_clusters_info_LST1['y_mean'],
                                DBSCAN_clusters_info_LST1['t_mean'],
                                DBSCAN_clusters_info_LST1['channelID'],
                                DBSCAN_clusters_info_LST1['timeID'],
                                DBSCAN_clusters_info_LST2['n_digitalsum_points'],
                                DBSCAN_clusters_info_LST2['n_clusters'],
                                DBSCAN_clusters_info_LST2['n_points'],
                                DBSCAN_clusters_info_LST2['x_mean'],
                                DBSCAN_clusters_info_LST2['y_mean'],
                                DBSCAN_clusters_info_LST2['t_mean'],
                                DBSCAN_clusters_info_LST2['channelID'],
                                DBSCAN_clusters_info_LST2['timeID'],
                                DBSCAN_clusters_info_LST3['n_digitalsum_points'],
                                DBSCAN_clusters_info_LST3['n_clusters'],
                                DBSCAN_clusters_info_LST3['n_points'],
                                DBSCAN_clusters_info_LST3['x_mean'],
                                DBSCAN_clusters_info_LST3['y_mean'],
                                DBSCAN_clusters_info_LST3['t_mean'],
                                DBSCAN_clusters_info_LST3['channelID'],
                                DBSCAN_clusters_info_LST3['timeID'],
                                DBSCAN_clusters_info_LST4['n_digitalsum_points'],
                                DBSCAN_clusters_info_LST4['n_clusters'],
                                DBSCAN_clusters_info_LST4['n_points'],
                                DBSCAN_clusters_info_LST4['x_mean'],
                                DBSCAN_clusters_info_LST4['y_mean'],
                                DBSCAN_clusters_info_LST4['t_mean'],
                                DBSCAN_clusters_info_LST4['channelID'],
                                DBSCAN_clusters_info_LST4['timeID']])
        #
        #
        ev_counter=ev_counter+1
        #
        #
        if (ev_counter >= nevmax and nevmax > 0):
            break
        if (it_cout%_event_modulo==0) :
            toc = time.time()
            print('{:10d} {:10.2f} s'.format(it_cout, toc - tic))
            tic = time.time()
        it_cout = it_cout + 1
        
    sf.close()
    
    return event_info_list

def save_data(event_info_list, outpkl, outcsv):
    event_info_arr=np.array(event_info_list)
    pkl.dump(event_info_arr, open(outpkl, "wb"), protocol=pkl.HIGHEST_PROTOCOL)    
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
                       'ev_time_LST1': event_info_arr[:,11],
                       'ev_time_LST2': event_info_arr[:,12],
                       'ev_time_LST3': event_info_arr[:,13],
                       'ev_time_LST4': event_info_arr[:,14],
                       'nphotons_LST1': event_info_arr[:,15],
                       'nphotons_LST2': event_info_arr[:,16],
                       'nphotons_LST3': event_info_arr[:,17],
                       'nphotons_LST4': event_info_arr[:,18],
                       'n_pe_LST1': event_info_arr[:,19],
                       'n_pe_LST2': event_info_arr[:,20],
                       'n_pe_LST3': event_info_arr[:,21],
                       'n_pe_LST4': event_info_arr[:,22],
                       'n_pixels_LST1': event_info_arr[:,23],
                       'n_pixels_LST2': event_info_arr[:,24],
                       'n_pixels_LST3': event_info_arr[:,25],
                       'n_pixels_LST4': event_info_arr[:,26],                       
                       'L1_max_digi_sum_LST1': event_info_arr[:,27],
                       'L1_x_mean_LST1': event_info_arr[:,28],
                       'L1_y_mean_LST1': event_info_arr[:,29],
                       'L1_t_mean_LST1': event_info_arr[:,30],
                       'L1_channelID_LST1': event_info_arr[:,31],
                       'L1_timeID_LST1': event_info_arr[:,32],
                       'L1_max_digi_sum_LST2': event_info_arr[:,33],
                       'L1_x_mean_LST2': event_info_arr[:,34],
                       'L1_y_mean_LST2': event_info_arr[:,35],
                       'L1_t_mean_LST2': event_info_arr[:,36],
                       'L1_channelID_LST2': event_info_arr[:,37],
                       'L1_timeID_LST2': event_info_arr[:,38],
                       'L1_max_digi_sum_LST3': event_info_arr[:,39],
                       'L1_x_mean_LST3': event_info_arr[:,40],
                       'L1_y_mean_LST3': event_info_arr[:,41],
                       'L1_t_mean_LST3': event_info_arr[:,42],
                       'L1_channelID_LST3': event_info_arr[:,43],
                       'L1_timeID_LST3': event_info_arr[:,44],
                       'L1_max_digi_sum_LST4': event_info_arr[:,45],
                       'L1_x_mean_LST4': event_info_arr[:,46],
                       'L1_y_mean_LST4': event_info_arr[:,47],
                       'L1_t_mean_LST4': event_info_arr[:,48],
                       'L1_channelID_LST4': event_info_arr[:,49],
                       'L1_timeID_LST4': event_info_arr[:,50],                       
                       'L3_iso_n_digitalsum_points_LST1': event_info_arr[:,51],
                       'L3_iso_n_clusters_LST1': event_info_arr[:,52],
                       'L3_iso_n_points_LST1': event_info_arr[:,53],
                       'L3_iso_x_mean_LST1': event_info_arr[:,54],
                       'L3_iso_y_mean_LST1': event_info_arr[:,55],
                       'L3_iso_t_mean_LST1': event_info_arr[:,56],
                       'L3_iso_channelID_LST1': event_info_arr[:,57],
                       'L3_iso_timeID_LST1': event_info_arr[:,58],
                       'L3_iso_n_digitalsum_points_LST2': event_info_arr[:,59],
                       'L3_iso_n_clusters_LST2': event_info_arr[:,60],
                       'L3_iso_n_points_LST2': event_info_arr[:,61],
                       'L3_iso_x_mean_LST2': event_info_arr[:,62],
                       'L3_iso_y_mean_LST2': event_info_arr[:,63],
                       'L3_iso_t_mean_LST2': event_info_arr[:,64],
                       'L3_iso_channelID_LST2': event_info_arr[:,65],
                       'L3_iso_timeID_LST2': event_info_arr[:,66],
                       'L3_iso_n_digitalsum_points_LST3': event_info_arr[:,67],
                       'L3_iso_n_clusters_LST3': event_info_arr[:,68],
                       'L3_iso_n_points_LST3': event_info_arr[:,69],
                       'L3_iso_x_mean_LST3': event_info_arr[:,70],
                       'L3_iso_y_mean_LST3': event_info_arr[:,71],
                       'L3_iso_t_mean_LST3': event_info_arr[:,72],
                       'L3_iso_channelID_LST3': event_info_arr[:,73],
                       'L3_iso_timeID_LST3': event_info_arr[:,74],
                       'L3_iso_n_digitalsum_points_LST4': event_info_arr[:,75],
                       'L3_iso_n_clusters_LST4': event_info_arr[:,76],
                       'L3_iso_n_points_LST4': event_info_arr[:,77],
                       'L3_iso_x_mean_LST4': event_info_arr[:,78],
                       'L3_iso_y_mean_LST4': event_info_arr[:,79],
                       'L3_iso_t_mean_LST4': event_info_arr[:,80],
                       'L3_iso_channelID_LST4': event_info_arr[:,81],
                       'L3_iso_timeID_LST4': event_info_arr[:,82],                      
                       'L3_cl_n_digitalsum_points_LST1': event_info_arr[:,83],
                       'L3_cl_n_clusters_LST1': event_info_arr[:,84],
                       'L3_cl_n_points_LST1': event_info_arr[:,85],
                       'L3_cl_x_mean_LST1': event_info_arr[:,86],
                       'L3_cl_y_mean_LST1': event_info_arr[:,87],
                       'L3_cl_t_mean_LST1': event_info_arr[:,88],
                       'L3_cl_channelID_LST1': event_info_arr[:,89],
                       'L3_cl_timeID_LST1': event_info_arr[:,90],                       
                       'L3_cl_n_digitalsum_points_LST2': event_info_arr[:,91],
                       'L3_cl_n_clusters_LST2': event_info_arr[:,92],
                       'L3_cl_n_points_LST2': event_info_arr[:,93],
                       'L3_cl_x_mean_LST2': event_info_arr[:,94],
                       'L3_cl_y_mean_LST2': event_info_arr[:,95],
                       'L3_cl_t_mean_LST2': event_info_arr[:,96],
                       'L3_cl_channelID_LST2': event_info_arr[:,97],
                       'L3_cl_timeID_LST2': event_info_arr[:,98],
                       'L3_cl_n_digitalsum_points_LST3': event_info_arr[:,99],
                       'L3_cl_n_clusters_LST3': event_info_arr[:,100],
                       'L3_cl_n_points_LST3': event_info_arr[:,101],
                       'L3_cl_x_mean_LST3': event_info_arr[:,102],
                       'L3_cl_y_mean_LST3': event_info_arr[:,103],
                       'L3_cl_t_mean_LST3': event_info_arr[:,104],
                       'L3_cl_channelID_LST3': event_info_arr[:,105],
                       'L3_cl_timeID_LST3': event_info_arr[:,106],
                       'L3_cl_n_digitalsum_points_LST4': event_info_arr[:,107],
                       'L3_cl_n_clusters_LST4': event_info_arr[:,108],
                       'L3_cl_n_points_LST4': event_info_arr[:,109],
                       'L3_cl_x_mean_LST4': event_info_arr[:,110],
                       'L3_cl_y_mean_LST4': event_info_arr[:,111],
                       'L3_cl_t_mean_LST4': event_info_arr[:,112],
                       'L3_cl_channelID_LST4': event_info_arr[:,113],
                       'L3_cl_timeID_LST4': event_info_arr[:,114]})
    df.to_csv(outcsv)


def main():
    pass
    
if __name__ == "__main__":
    if (len(sys.argv)==9 and (str(sys.argv[1]) == "--noise")):
        #
        simtelIn = str(sys.argv[2])
        outpkl = str(sys.argv[3])
        outcsv = str(sys.argv[4])
        pixel_mapping_csv = str(sys.argv[5])
        isolated_flower_seed_super_flower_csv = str(sys.argv[6])
        isolated_flower_seed_flower_csv = str(sys.argv[7])
        all_seed_flower_csv = str(sys.argv[8])
        #
        print("sys.argv[1]                           = ", sys.argv[1])
        print("simtelIn                              = ", simtelIn)
        print("outpkl                                = ", outpkl)
        print("outcsv                                = ", outcsv)
        print("pixel_mapping_csv                     = ", pixel_mapping_csv)
        print("isolated_flower_seed_super_flower_csv = ", isolated_flower_seed_super_flower_csv)
        print("isolated_flower_seed_flower_csv       = ", isolated_flower_seed_flower_csv)
        print("all_seed_flower_csv                   = ", all_seed_flower_csv)
        #
        print_setup()
        #
        pixel_mapping = np.genfromtxt(pixel_mapping_csv)
        isolated_flower_seed_flower = np.genfromtxt(isolated_flower_seed_flower_csv,dtype=int) 
        isolated_flower_seed_super_flower = np.genfromtxt(isolated_flower_seed_super_flower_csv,dtype=int)
        all_seed_flower = np.genfromtxt(all_seed_flower_csv,dtype=int)
        #
        evtloop_noise( datafilein=simtelIn, nevmax=100,
                       pixel_mapping=pixel_mapping,
                       L1_trigger_pixel_cluster_list=isolated_flower_seed_super_flower,
                       L3_trigger_DBSCAN_pixel_cluster_list=isolated_flower_seed_flower,
                       L3_trigger_DBSCAN_pixel_cluster_list_all=all_seed_flower)
        #
    elif (len(sys.argv)==9 and (str(sys.argv[1]) == "--trg")):
        #
        simtelIn = str(sys.argv[2])
        outpkl = str(sys.argv[3])
        outcsv = str(sys.argv[4])
        pixel_mapping_csv = str(sys.argv[5])
        isolated_flower_seed_super_flower_csv = str(sys.argv[6])
        isolated_flower_seed_flower_csv = str(sys.argv[7])
        all_seed_flower_csv = str(sys.argv[8])
        #
        print("sys.argv[1]                           = ", sys.argv[1])
        print("simtelIn                              = ", simtelIn)
        print("outpkl                                = ", outpkl)
        print("outcsv                                = ", outcsv)
        print("pixel_mapping_csv                     = ", pixel_mapping_csv)
        print("isolated_flower_seed_super_flower_csv = ", isolated_flower_seed_super_flower_csv)
        print("isolated_flower_seed_flower_csv       = ", isolated_flower_seed_flower_csv)
        print("all_seed_flower_csv                   = ", all_seed_flower_csv)
        #
        print_setup()
        #
        pixel_mapping = np.genfromtxt(pixel_mapping_csv)
        isolated_flower_seed_flower = np.genfromtxt(isolated_flower_seed_flower_csv,dtype=int) 
        isolated_flower_seed_super_flower = np.genfromtxt(isolated_flower_seed_super_flower_csv,dtype=int)
        all_seed_flower = np.genfromtxt(all_seed_flower_csv,dtype=int)
        #
        event_info_list = evtloop( datafilein=simtelIn, nevmax=-1,
                                   pixel_mapping=pixel_mapping,
                                   L1_trigger_pixel_cluster_list=isolated_flower_seed_super_flower,
                                   L3_trigger_DBSCAN_pixel_cluster_list=isolated_flower_seed_flower,
                                   L3_trigger_DBSCAN_pixel_cluster_list_all=all_seed_flower)
        save_data(event_info_list, outpkl, outcsv)
        #
    else:
        print(" --> HELP info")
        print("len(sys.argv) = ",len(sys.argv))
        print("sys.argv      = ",sys.argv)
        print(" ---> for noise")
        print(" [1] --noise")
        print(" [2] simtelIn")
        print(" [3] outpkl")
        print(" [4] outcsv")
        print(" [5] pixel_mapping_csv")
        print(" [6] isolated_flower_seed_super_flower_csv")
        print(" [7] isolated_flower_seed_flower_csv")
        print(" [8] all_seed_flower_csv")
        print(" ---> for events")
        print(" [1] --trg")
        print(" [2] simtelIn")
        print(" [3] outpkl")
        print(" [4] outcsv")
        print(" [5] pixel_mapping_csv")
        print(" [6] isolated_flower_seed_super_flower_csv")
        print(" [7] isolated_flower_seed_flower_csv")
        print(" [8] all_seed_flower_csv")
