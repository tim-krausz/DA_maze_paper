"""Functions to analyze and vizualize photometry data.
Specifically analyses for DA maze paper main figure plots.
TODO: break into multiple PhotRats child classes"""

__author__ = "Tim Krausz"
__email__ = "krausz.tim@gmail.com"
__status__ = "development"
__license__ = "MIT"

from sklearn.linear_model import LogisticRegression as LogReg
from scipy import stats
import statsmodels.api as sm
from __main__ import *


def add_pairedHexStates2df(photrats):
    photrats.df.loc[:,"pairedHexStates"] = -1
    photrats.df.loc[photrats.df.hexlabels.diff()!=0,\
    "pairedHexStates"] = photrats.convert2pairedState(
           photrats.df.loc[photrats.df.hexlabels.diff()!=0,\
           "hexlabels"].values)
    photrats.df.loc[:,"pairedHexStates"] = photrats.df.loc[:,\
    "pairedHexStates"].replace(-1,method="ffill")

def plot_portAlignedDaInTime(photrats,secondHalfOnly=True,
    poolFactor="nom_rwd_chosen",useRatGroupLevel=True):
    photrats.set_pool_factor(poolFactor)
    photrats.set_plot_trace("green_z_scored")
    fig = plt.figure(figsize = (7,5))
    rwdhigh,omhigh,rwdmid,ommid,rwdlow,omlow = photrats.getSessionTercMeans(\
        secondHalf=secondHalfOnly,useRat=useRatGroupLevel)
    
    high_color = "red"#"indianred"
    mid_color = "firebrick"
    low_color = "maroon"
    high_colorOm = "dodgerblue"
    mid_colorOm = "blue"
    low_colorOm ="darkblue"
    
    xvals = np.arange(photrats.fs*photrats.plot_window[0],\
        photrats.fs*photrats.plot_window[1]+1)/photrats.fs
    ax1 = plt.gca()
    plot_avgWithSem(photrats,ax1,xvals,np.vstack((rwdhigh,omhigh)),'lightgrey','-',\
                    [None,-photrats.plot_window[0]*photrats.fs],"high p(Reward)")
    plot_avgWithSem(photrats,ax1,xvals,rwdhigh,high_color,'-',\
        [-photrats.plot_window[0]*photrats.fs,None])
    plot_avgWithSem(photrats,ax1,xvals,omhigh,high_colorOm,':',\
        [-photrats.plot_window[0]*photrats.fs,None])
    plot_avgWithSem(photrats,ax1,xvals,np.vstack((rwdmid,ommid)),'darkgrey','-',\
                    [None,-photrats.plot_window[0]*photrats.fs],"medium p(Reward)")
    plot_avgWithSem(photrats,ax1,xvals,rwdmid,mid_color,'-',\
        [-photrats.plot_window[0]*photrats.fs,None])
    plot_avgWithSem(photrats,ax1,xvals,ommid,mid_colorOm,':',\
        [-photrats.plot_window[0]*photrats.fs,None])
    ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
    ax1.set_xlabel('time (s) from port entry')
    plot_avgWithSem(photrats,ax1,xvals,np.vstack((rwdlow,omlow)),'dimgrey','-',\
                    [None,-photrats.plot_window[0]*photrats.fs],"low p(Reward)")
    plot_avgWithSem(photrats,ax1,xvals,rwdlow,low_color,'-',\
        [-photrats.plot_window[0]*photrats.fs,None])
    plot_avgWithSem(photrats,ax1,xvals,omlow,low_colorOm,':',\
        [-photrats.plot_window[0]*photrats.fs,None])
    ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
    ax1.set_xlabel('time (s) from port entry')
    ax1.legend()
    plt.xlabel("time from port entry (s)",fontsize='xx-large')
    plt.ylabel("DA (z-scored)",fontsize='xx-large')
    plt.tight_layout()
    return fig

def write_dict_to_file(my_dict, file_path):
    """
    Writes a dictionary to a text file at the specified file path.

    Parameters:
    my_dict (dict): The dictionary to write to the file.
    file_path (str): The path to the file to write to.
    """
    with open(file_path, 'w') as f:
        for key, value in my_dict.items():
            f.write(f'{key}: {value}\n')

def plot_portAlignedDaInTime_byQ(photrats):
    photrats.set_pool_factor("Q_chosen")
    photrats.set_plot_trace("green_z_scored")
    fig = plt.figure(figsize = (7,5))
    rwdhigh,omhigh,rwdmid,ommid,rwdlow,omlow = photrats.getSessionTercMeans(secondHalf=True)
    
    high_color = "red"#"indianred"
    mid_color = "firebrick"
    low_color = "maroon"
    high_colorOm = "dodgerblue"
    mid_colorOm = "blue"
    low_colorOm ="darkblue"
    
    xvals = np.arange(photrats.fs*photrats.plot_window[0],photrats.fs*photrats.plot_window[1]+1)/photrats.fs
    ax1 = plt.gca()
    plot_avgWithSem(photrats,ax1,xvals,np.vstack((rwdhigh,omhigh)),'lightgrey','-',\
                    [None,-photrats.plot_window[0]*photrats.fs],"high Q")
    plot_avgWithSem(photrats,ax1,xvals,rwdhigh,high_color,'-',[-photrats.plot_window[0]*photrats.fs,None])
    plot_avgWithSem(photrats,ax1,xvals,omhigh,high_colorOm,':',[-photrats.plot_window[0]*photrats.fs,None])
    plot_avgWithSem(photrats,ax1,xvals,np.vstack((rwdmid,ommid)),'darkgrey','-',\
                    [None,-photrats.plot_window[0]*photrats.fs],"medium Q")
    plot_avgWithSem(photrats,ax1,xvals,rwdmid,mid_color,'-',[-photrats.plot_window[0]*photrats.fs,None])
    plot_avgWithSem(photrats,ax1,xvals,ommid,mid_colorOm,':',[-photrats.plot_window[0]*photrats.fs,None])
    ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
    ax1.set_xlabel('time (s) from port entry')
    plot_avgWithSem(photrats,ax1,xvals,np.vstack((rwdlow,omlow)),'dimgrey','-',\
                    [None,-photrats.plot_window[0]*photrats.fs],"low Q")
    plot_avgWithSem(photrats,ax1,xvals,rwdlow,low_color,'-',[-photrats.plot_window[0]*photrats.fs,None])
    plot_avgWithSem(photrats,ax1,xvals,omlow,low_colorOm,':',[-photrats.plot_window[0]*photrats.fs,None])
    ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
    ax1.set_xlabel('time (s) from port entry')
    ax1.legend()
    plt.xlabel("time from port entry (s)",fontsize='xx-large')
    plt.ylabel("DA (z-scored)",fontsize='xx-large')
    plt.tight_layout()

def plot_avgWithSem(photratz,ax,xvals,plot_trace,colorString,linstyle='-',subset=[None,None],traceLabel=None):
    ax.plot(xvals[subset[0]:subset[1]],np.nanmean\
        (plot_trace,axis=0)[subset[0]:subset[1]],color=colorString,ls=linstyle,label=traceLabel)
    ax.fill_between(xvals[subset[0]:subset[1]],(np.nanmean\
        (plot_trace,axis=0)+sem(plot_trace,nan_policy="omit"))\
        [subset[0]:subset[1]],(np.nanmean\
        (plot_trace,axis=0)-sem(plot_trace,nan_policy="omit"))\
        [subset[0]:subset[1]],color=colorString,alpha=.5)


def plot_meanRatDAafterPortEntry(photrats,highInds,midInds,lowInds,pltCol="firebrick"):
    highMeans = calc_DaPeakTroughAfterInds(photrats,highInds)
    midMeans = calc_DaPeakTroughAfterInds(photrats,midInds)
    lowMeans = calc_DaPeakTroughAfterInds(photrats,lowInds)
    highMeans = [np.mean(rm) for rm in highMeans if len(rm)>0]
    midMeans = [np.mean(rm) for rm in midMeans if len(rm)>0]
    lowMeans = [np.mean(rm) for rm in lowMeans if len(rm)>0]
    plt.bar([0,1,2],[np.mean(highMeans),\
            np.mean(midMeans),np.mean(lowMeans)],color=pltCol,alpha=0.3)
    plt.ylabel("mean $\Delta$DA",fontsize='xx-large',fontweight='bold')
    plt.xlabel("p(reward)",fontsize='xx-large',fontweight='bold')
    for r in range(len(highMeans)):
        plt.scatter(x=np.add([0,1,2],np.random.randn(1)/10),\
            y=[np.mean(highMeans[r]),np.mean(midMeans[r]),np.mean(lowMeans[r])],\
                    c=pltCol,edgecolors='k',lw=2,s=45)
        plt.plot([0,1,2],[np.mean(highMeans[r]),\
            np.mean(midMeans[r]),np.mean(lowMeans[r])],color='k',alpha=0.5,lw=1)
    plt.xticks([0,1,2],["High","Med","Low"],fontsize="x-large")
    plt.tight_layout()

def plot_peakTroughDAafterPortEntry_barWithRats(photrats,highInds,midInds,lowInds,peak=True,pltCol="firebrick"):
    highMeans = calc_DaPeakTroughAfterInds(photrats,highInds,peak)
    midMeans = calc_DaPeakTroughAfterInds(photrats,midInds,peak)
    lowMeans = calc_DaPeakTroughAfterInds(photrats,lowInds,peak)
    highMeans = [np.mean(rm) for rm in highMeans if len(rm)>0]
    midMeans = [np.mean(rm) for rm in midMeans if len(rm)>0]
    lowMeans = [np.mean(rm) for rm in lowMeans if len(rm)>0]
    plt.bar([0,1,2],[np.mean(highMeans),\
            np.mean(midMeans),np.mean(lowMeans)],color=pltCol,alpha=0.3)
    plt.ylabel("mean $\Delta$DA",fontsize='xx-large',fontweight='bold')
    plt.xlabel("p(reward)",fontsize='xx-large',fontweight='bold')
    for r in range(len(highMeans)):
        plt.scatter(x=np.add([0,1,2],np.random.randn(1)/10),\
            y=[np.mean(highMeans[r]),np.mean(midMeans[r]),np.mean(lowMeans[r])],\
                    c=pltCol,edgecolors='k',lw=2,s=45)
        plt.plot([0,1,2],[np.mean(highMeans[r]),\
            np.mean(midMeans[r]),np.mean(lowMeans[r])],color='k',alpha=0.5,lw=1)
    plt.xticks([0,1,2],["High","Med","Low"],fontsize="x-large")
    plt.tight_layout()

def calc_DaChangeAtInds(photrats,indices):
    photrats.set_plot_window([0,1])
    tracesPost = get_TracesAroundIndex(photrats,indices)
    photrats.set_plot_window([-1,0])
    tracesPre = get_TracesAroundIndex(photrats,indices)
    traceChangeRats = photrats.df.loc[indices,"rat"].astype(str).values
    traceChangeRatMeans = []
    for rat in photrats.df.rat.unique():
        traceChangeRatMeans.append(np.mean(tracesPost[np.where(traceChangeRats==rat)[0]],axis=1)-\
                            np.mean(tracesPre[np.where(traceChangeRats==rat)[0]],axis=1))
    return traceChangeRatMeans

def calc_DaPeakTroughAfterInds(photrats,indices,peak=True):
    photrats.set_plot_window([0,1])
    tracesPost = get_TracesAroundIndex(photrats,indices)
    tracePeakRats = photrats.df.loc[indices,"rat"].astype(str).values
    tracePeakRatMeans = []
    for rat in photrats.df.rat.unique():
        if peak:
            tracePeakRatMeans.append(np.max(tracesPost[np.where(tracePeakRats==rat)[0]],axis=1))
        else:
            tracePeakRatMeans.append(np.min(tracesPost[np.where(tracePeakRats==rat)[0]],axis=1))
    return tracePeakRatMeans


def plot_peakTroughDaDifAfterPortEntry_barWithRats(photrats,highInds,
    midInds,lowInds,peak=True,pltCol="firebrick"):
    highMeans = calc_DaPeakTroughDiffAfterPortInds(photrats,highInds,peak)
    midMeans = calc_DaPeakTroughDiffAfterPortInds(photrats,midInds,peak)
    lowMeans = calc_DaPeakTroughDiffAfterPortInds(photrats,lowInds,peak)
    plt.bar([0,1,2],[np.mean(highMeans),\
            np.mean(midMeans),np.mean(lowMeans)],color=pltCol,alpha=0.3)
    plt.ylabel("mean $\Delta$DA",fontsize='xx-large',fontweight='bold')
    plt.xlabel("p(reward)",fontsize='xx-large',fontweight='bold')
    for r in range(len(highMeans)):
        plt.scatter(x=np.add([0,1,2],np.random.randn(1)/10),\
            y=[np.mean(highMeans[r]),np.mean(midMeans[r]),np.mean(lowMeans[r])],\
                    c=pltCol,edgecolors='k',lw=2,s=55)
        plt.plot([0,1,2],[np.mean(highMeans[r]),\
            np.mean(midMeans[r]),np.mean(lowMeans[r])],color='k',alpha=0.5,lw=1)
    plt.xticks([0,1,2],["High","Med","Low"],fontsize="x-large")
    plt.tight_layout()

def calc_DaPeakTroughDiffAfterPortInds(photrats,indices,peak=True):
    winMax = 0.5 if peak else 1.0
    photrats.set_plot_window([0,winMax])
    tracePeakRats = photrats.df.loc[indices,"rat"].astype(str).values
    tracePeakRatMeans = []
    for rat in photrats.df.rat.unique():
        tracesPost = get_TracesAroundIndex(photrats,
            indices[np.isin(indices,photrats.df.loc[photrats.df.rat==rat].index)])
        tracePost = np.mean(tracesPost,axis=0)
        if peak:
            tracePeakRatMeans.append(np.max(tracePost)-tracePost[0])
        else:
            tracePeakRatMeans.append(np.min(tracePost)-tracePost[0])
    return tracePeakRatMeans

def get_TracesAroundIndex(photrats,indices):
    traces = []
    for i in indices:
        if np.isnan(i) or i == -1:
            continue
        traces.append(photrats.df.loc[i+photrats.fs*photrats.plot_window[0]:\
                i+photrats.fs*photrats.plot_window[1],photrats.plot_trace].values)
    return np.array(traces)

def calc_DaChangeVprobCors(photrats):
    rwdCors = []
    omCors = []
    for rat in photrats.df.rat.unique():
        photrats.dat = photrats.df.loc[(photrats.df.rat==rat)\
                                       &(photrats.df.tri>25)&(photrats.df.rwd==1),]
        lowInds,midInds,highInds = photrats.getTriIndsByTerc()
        daChanges = np.concatenate([calc_DaChangeAtIndsOneRat(photrats,highInds,peak=True),
                     calc_DaChangeAtIndsOneRat(photrats,midInds,peak=True),
                     calc_DaChangeAtIndsOneRat(photrats,lowInds,peak=True)])
        probs = np.concatenate([photrats.dat.loc[highInds,"nom_rwd_chosen"].values/100,\
        photrats.dat.loc[midInds,"nom_rwd_chosen"].values/100,\
        photrats.dat.loc[lowInds,"nom_rwd_chosen"].values/100])
        rwdCors.append(pearsonr(probs,daChanges))
        photrats.dat = photrats.df.loc[(photrats.df.rat==rat)\
                                       &(photrats.df.tri>25)&(photrats.df.rwd==0),]
        photrats.dat_visinds = photrats.dat.loc[photrats.dat.port!=-100].index
        lowInds,midInds,highInds = photrats.getTriIndsByTerc()
        daChanges = np.concatenate([calc_DaChangeAtIndsOneRat(photrats,highInds,peak=False),
                     calc_DaChangeAtIndsOneRat(photrats,midInds,peak=False),
                     calc_DaChangeAtIndsOneRat(photrats,lowInds,peak=False)])
        probs = np.concatenate([photrats.dat.loc[highInds,"nom_rwd_chosen"].values/100,\
        photrats.dat.loc[midInds,"nom_rwd_chosen"].values/100,\
        photrats.dat.loc[lowInds,"nom_rwd_chosen"].values/100])
        omCors.append(pearsonr(probs,daChanges))
        pd.DataFrame(rwdCors,columns=["coef","p-val"]).to_csv(photrats.directory_prefix+"pearsonR_result_DaVsRpe_rwd.csv")
        pd.DataFrame(omCors,columns=["coef","p-val"]).to_csv(photrats.directory_prefix+"pearsonR_result_DaVsRpe_om.csv")
    return rwdCors,omCors

def calc_DaChangeAtIndsOneRat(photrats,indices,peak=True):
    winMax = 0.5 if peak else 1.0
    photrats.set_plot_window([0,winMax])
    tracesPost = get_TracesAroundIndex(photrats,indices)
    daChange = np.max(tracesPost,axis=1) - tracesPost[:,0]\
     if peak else np.min(tracesPost,axis=1) - tracesPost[:,0]
    return daChange

def calc_DaPeakTroughAfterIndsOneRat(photrats,indices,rat,peak=True):
    photrats.set_plot_window([0,1])
    tracesPost = get_TracesAroundIndex(photrats,indices)
    tracePeakRats = photrats.df.loc[indices,"rat"].astype(str).values
    tracePeakRatMeans = []
    if peak:
        tracePeakRatMeans = np.max(tracesPost[np.where(tracePeakRats==rat)[0]],axis=1)
    else:
        tracePeakRatMeans = np.min(tracesPost[np.where(tracePeakRats==rat)[0]],axis=1)
    return tracePeakRatMeans

def calcRpeRegByRatAndSesh(photrats,useQ=False):
    ratRwdRpes = {r:[] for r in photrats.df.loc[:,"rat"].unique()}
    ratOmRpes = {r:[] for r in photrats.df.loc[:,"rat"].unique()}
    ratRwdNs = {r:[] for r in photrats.df.loc[:,"rat"].unique()}
    ratOmNs = {r:[] for r in photrats.df.loc[:,"rat"].unique()}
    for r in tqdm(range(len(photrats.df.loc[:,"rat"].unique()))):
        rat = photrats.df.loc[:,"rat"].unique()[r]
        for s in photrats.df.loc[photrats.df.rat==rat,"session"].unique():
            if useQ:
                dat = photrats.df.loc[(photrats.df.session==s),:]
                qs = dat.loc[dat.port!=-100,"Q_chosen"].values
            else:
                dat = photrats.df.loc[(photrats.df.session==s)&(photrats.df.tri>25),:]
                pRwds = dat.loc[dat.port!=-100,"nom_rwd_chosen"].values
            rwds = dat.loc[dat.port!=-100,"rwd"].values
            rpes = rwds-qs if useQ else rwds-(pRwds/100)
            rweightsRwd,rweightsOm = calcRpeLagRegWeightsBinned(photrats,dat,rpes,rwds,100)
            ratRwdRpes[rat].append(rweightsRwd[0])
            ratOmRpes[rat].append(rweightsOm[0])
            ratRwdNs[rat].append(len(rwds[rwds==1]))
            ratOmNs[rat].append(len(rwds[rwds==0]))
    return ratRwdRpes,ratOmRpes,ratRwdNs,ratOmNs

def calcRpeLagRegWeights(photrats,dat,pRwdRpes,rwds):
    dat_vinds = dat.loc[(dat.port!=-100),:].index
    lags = np.arange(0,photrats.fs*2.5)
    rweightsRwd = np.zeros((1,len(lags)))
    rweightsOm = np.zeros((1,len(lags)))
    for n in range(len(lags)):
        preLagInds = dat_vinds[np.where(rwds==1)]+lags[n]
        lagInds = preLagInds[np.isin(preLagInds,dat.index)]
        yRwd = dat.loc[lagInds,"green_z_scored"]
        yRwd.reset_index(drop=True,inplace=True)
        XRwd = pd.DataFrame({"posRpe":pRwdRpes[np.where(rwds==1)][np.isin(preLagInds,dat.index)]})
        XRwd = XRwd.drop(yRwd.loc[yRwd.isnull()].index,axis=0)
        yRwd = yRwd.drop(yRwd.loc[yRwd.isnull()].index,axis=0)
        modRwd = LR(fit_intercept=True,normalize=False).fit(XRwd,yRwd)
        rweightsRwd[:,n] = modRwd.coef_
        preLagInds = dat_vinds[np.where(rwds==0)]+lags[n]
        lagInds = preLagInds[np.isin(preLagInds,dat.index)]
        yOm = dat.loc[lagInds,"green_z_scored"]
        yOm.reset_index(drop=True,inplace=True)
        XOm = pd.DataFrame({"posRpe":pRwdRpes[np.where(rwds==0)][np.isin(preLagInds,dat.index)]})
        XOm = XOm.drop(yOm.loc[yOm.isnull()].index,axis=0)
        yOm = yOm.drop(yOm.loc[yOm.isnull()].index,axis=0)
        modOm = LR(fit_intercept=True,normalize=False).fit(XOm,yOm)
        rweightsOm[:,n] = modOm.coef_
    return rweightsRwd,rweightsOm

def calcRpeLagRegWeightsBinned(photrats,dat,pRwdRpes,rwds,binsize=50):
    dat_vinds = dat.loc[(dat.port!=-100),:].index
    lags = np.arange(0,photrats.fs*2.5,int(photrats.fs*(binsize/1000)))
    rweightsRwd = np.zeros((1,len(lags)))
    rweightsOm = np.zeros((1,len(lags)))
    for n in range(0,len(lags)-1):
        preLagInds = dat_vinds[np.where(rwds==1)]+lags[n]
        lagInds = preLagInds[np.isin(preLagInds,dat.index)]
        das = []
        for l in lagInds:
            das.append(dat.loc[l:l+lags[n+1],"green_z_scored"].mean())
        yRwd = pd.Series(das)#dat.loc[lagInds,"green_z_scored"]
        XRwd = pd.DataFrame({"posRpe":pRwdRpes[np.where(rwds==1)][np.isin(preLagInds,dat.index)]})
        XRwd = XRwd.drop(yRwd.loc[yRwd.isnull()].index,axis=0)
        yRwd = yRwd.drop(yRwd.loc[yRwd.isnull()].index,axis=0)
        modRwd = LR(fit_intercept=True,normalize=False).fit(XRwd,yRwd)
        rweightsRwd[:,n] = modRwd.coef_
        preLagInds = dat_vinds[np.where(rwds==0)]+lags[n]
        lagInds = preLagInds[np.isin(preLagInds,dat.index)]
        das = []
        for l in lagInds:
            das.append(dat.loc[l:l+lags[n+1],"green_z_scored"].mean())
        yOm = pd.Series(das)
        XOm = pd.DataFrame({"posRpe":pRwdRpes[np.where(rwds==0)][np.isin(preLagInds,dat.index)]})
        XOm = XOm.drop(yOm.loc[yOm.isnull()].index,axis=0)
        yOm = yOm.drop(yOm.loc[yOm.isnull()].index,axis=0)
        modOm = LR(fit_intercept=True,normalize=False).fit(XOm,yOm)
        rweightsOm[:,n] = modOm.coef_
    return rweightsRwd,rweightsOm

def plot_ratMeans(xvals,ratDict,pltColor='darkred',pltLabel=""):
    ratmeans = []
    for rat in ratDict:
        ratmeans.append(np.nanmean(ratDict[rat],axis=0))
        #plt.plot(xvals,ratmeans[-1],color=pltColor,alpha=0.2,lw=.5)
    plt.plot(xvals,np.nanmean(ratmeans,axis=0),color=pltColor,lw=3,label=pltLabel)
    plt.fill_between(xvals,np.nanmean(ratmeans,axis=0)-sem(ratmeans),\
          np.mean(ratmeans,axis=0)+sem(ratmeans),color=pltColor,lw=3,alpha=0.3)

def plot_sigPoints(xvals,ratDict,pltColor='darkred',plot99=False):
    rat95Errors = []
    rat99Errors = []
    for rat in ratDict:
        er95 = sem(ratDict[rat],axis=0)*1.96
        er99 = sem(ratDict[rat],axis=0)*2.58
        meanTrace = np.nanmean(ratDict[rat],axis=0)
        rat95Errors.append(((meanTrace-er95)>0).astype(int))
        rat99Errors.append(((meanTrace-er99)>0).astype(int))
    if plot99:
        plt.plot(xvals,np.sum(rat99Errors,axis=0)/10,color=pltColor)
    else:
        plt.plot(xvals,np.sum(rat95Errors,axis=0)/10,color=pltColor)
        
def save_RpeNumbers(photrats,ratRwdNs,ratOmNs):
    numbers = 'rewarded:\n'
    for rat in ratRwdNs:
        numbers += rat +" had " + str(len(ratRwdNs[rat])) +\
        " sessions with " + str(ratRwdNs[rat]) + " = " + str(np.sum(ratRwdNs[rat])) + " RPEs.\n"
    numbers += "omissions:\n"
    for rat in ratOmNs:
        numbers += rat +" had " + str(len(ratOmNs[rat])) +\
        " sessions  with " + str(ratOmNs[rat]) + " = " + str(np.sum(ratOmNs[rat])) + " RPEs.\n"
    
    with open(photrats.directory_prefix+"rpeLagRegNumbers.txt", 'w') as f:
        f.write(numbers)

def removeErrantBlock1Assignments(photrats):
    '''Identify and remove data appended to end of certain sessions
    marked as "block 1" incorrectly'''
    start_extra_block1_inds = photrats.df.loc[\
    (photrats.df.block.diff()<0)&(photrats.df.session.diff()==0),:].index
    seshStartInds = photrats.df.loc[(photrats.df.session.diff()!=0)].index
    end_extraBlock1_inds = []
    for i in start_extra_block1_inds[:-1]:
        end_extraBlock1_inds.append(seshStartInds[np.where(seshStartInds>i)[0][0]])
    end_extraBlock1_inds.append(photrats.df.index.max())
    for i in range(len(start_extra_block1_inds)):
        photrats.df.drop(index=np.arange(\
            start_extra_block1_inds[i],end_extraBlock1_inds[i]),axis=0,inplace=True)
    photrats.df.drop(index=photrats.df.index.max(),axis=0,inplace=True)
    photrats.df.reset_index(inplace=True)
    photrats.get_visinds()

def find_newHexAdjInds(photrats):
    newHexAdjInds = []
    newHexAdjStates = []
    enteredHex = []
    enteredHexSoon = []
    allEntryInds = photrats.df.loc[photrats.df.hexlabels.diff()!=0,:].index
    for s in photrats.sesh_newlyAvailHexes:
        for b in range(len(photrats.sesh_newlyAvailHexes[s])):
            #get indices of first entry into an adjacent-to-newly-available hex
            newHexAdjInds.append(photrats.df.loc[(photrats.df.session==s)\
                &(photrats.df.block==b+2)&(photrats.df.adj2newlyAvail.diff()==1)].index.min())
            if np.isnan(newHexAdjInds[-1]):
                print("session ",str(s)," block ",b+1)
                newHexAdjStates.append(-1)
                enteredHex.append(-1)
                enteredHexSoon.append(-1)
            else:
                nextHexInd = allEntryInds[np.where(allEntryInds==newHexAdjInds[-1])[0][0]+1]
                nextHex = photrats.df.loc[nextHexInd,"hexlabels"]#newly avail are in hexID format
                nextHexes = photrats.df.loc[allEntryInds[np.where(allEntryInds==newHexAdjInds[-1])[0][0]+1:\
                         np.where(allEntryInds==newHexAdjInds[-1])[0][0]+4],"hexlabels"].values
                enteredHex.append((nextHex==photrats.sesh_newlyAvailHexes[s][b]).astype(int)[0])
                enteredHexSoon.append(int(photrats.sesh_newlyAvailHexes[s][b][0] in nextHexes))
                newHexAdjStates.append(photrats.df.loc[newHexAdjInds[-1],"pairedHexStates"])
    return newHexAdjInds,newHexAdjStates,enteredHex,enteredHexSoon

def find_enteredVignoredNewlyAvailInds(newHexAdjInds,enteredHex):
    enteredInds = [n for n,z in zip(newHexAdjInds,enteredHex) if z==1]
    ignoredInds = [n for n,z in zip(newHexAdjInds,enteredHex) if z==0]
    return enteredInds,ignoredInds

def find_blockedHexAdjInds(photrats):
    blockedHexAdjInds = []
    blockedHexAdjStates = []
    for s in photrats.sesh_newlyBlockedHexes:
        for b in range(len(photrats.sesh_newlyBlockedHexes[s])):
            blockedHexAdjInds.append(photrats.df.loc[(photrats.df.session==s)\
                &(photrats.df.block==b+2)&(photrats.df.adj2newlyBlocked.diff()==1)].index.min())
            if np.isnan(blockedHexAdjInds[-1]):
                print("session ",str(s)," block ",b+1)
                blockedHexAdjStates.append(-1)
            else:
                blockedHexAdjStates.append(photrats.df.loc[blockedHexAdjInds[-1],"pairedHexStates"])
    return blockedHexAdjInds,blockedHexAdjStates

# can plot aligned to newly avail entry, or
def find_newHexEntryAndPriorHexInds(photrats):
    newHexInds = []
    newHexStates = []
    adjHexStates = [] #previously entered hex
    adjHexInds = []
    allEntryInds = photrats.df.loc[photrats.df.pairedHexStates.diff()!=0,:].index
    for s in photrats.sesh_newlyAvailHexes:
        for b in range(len(photrats.sesh_newlyAvailHexes[s])):
            newHexInds.append(photrats.df.loc[(photrats.df.session==s)\
                &(photrats.df.block==b+2)&(photrats.df.newlyAvailHex.diff()==1)].index.min())
            if np.isnan(newHexInds[-1]):
                print("session ",str(s)," block ",b+1)
                newHexStates.append(-1)
                adjHexInds.append(-1)
                adjHexStates.append(-1)
            else:
                newHexStates.append(photrats.df.loc[newHexInds[-1],"pairedHexStates"])
                try:
                    adjHexInds.append(allEntryInds[np.where(allEntryInds==newHexInds[-1])[0][0]-1])
                    adjHexStates.append(photrats.df.loc[adjHexInds[-1],"pairedHexStates"])
                except:
                    adjHexInds.append(-1)
                    adjHexStates.append(-1)
                    print("No previous adjacent hex entry detected for session ",str(s)," block ",b+1)
    return newHexInds,newHexStates,adjHexStates,adjHexInds

def plotFirstEntryHexChangeMeanOverRats(photrats,availratmeans,blockedratmeans,legend_on=False,pltCol1='deeppink',pltCol2='k',ls2='-'):
    photrats.set_plot_trace("green_z_scored")
    photrats.set_plot_window([-5,5])
    fig = plt.figure(figsize=(7,5))#(4.8,5))
    xvals = np.arange(photrats.plot_window[0]*photrats.fs,photrats.plot_window[1]*photrats.fs+1)/photrats.fs
    smoothWin = int(photrats.fs/10)
    toplt = pd.Series(np.mean(availratmeans,axis=0)).rolling(smoothWin).mean().values
    topltSem = pd.Series(sem(availratmeans,axis=0)).rolling(smoothWin).mean().values
    plt.plot(xvals,toplt,label="Newly available",color=pltCol1,lw=3)
    plt.fill_between(xvals,toplt-topltSem,toplt+topltSem,color=pltCol1,alpha=.3)
    toplt = pd.Series(np.mean(blockedratmeans,axis=0)).rolling(smoothWin).mean().values
    topltSem = pd.Series(sem(blockedratmeans,axis=0)).rolling(smoothWin).mean().values
    plt.plot(xvals,toplt,label="Newly blocked",color=pltCol2,ls=ls2,lw=3)
    plt.fill_between(xvals,toplt-topltSem,toplt+topltSem,color=pltCol2,alpha=.3)
    plt.xlabel("time from hex entry (s)",fontsize="xx-large")
    plt.ylabel("Mean z-scored DA",fontsize="xx-large")
    plt.axvline(x=0,ls='--',color='k',alpha=.8,lw=1)
    plt.xticks(np.arange(-5,6))
    plt.tight_layout()
    plt.ylim(-.9,2.8)
    if legend_on:
        plt.legend()
    plt.tight_layout()
    return fig

def calc_DaPeakDiffAfterNewPathInds(photrats,indices):
    photrats.set_plot_window([-1,0.25])
    tracePeakRats = photrats.df.loc[indices,"rat"].astype(str).values
    tracePeakRatMeans = []
    missingInds = []
    for rat in photrats.df.rat.unique():
        if rat not in tracePeakRats:
            missingInds.append(np.where(photrats.df.rat.unique()==rat)[0][0])
            continue
        tracesPost = get_TracesAroundIndex(photrats,indices[tracePeakRats==rat])
        bline = tracesPost[:,0]#np.mean(tracesPost,axis=0)[0]
        #tracePost = np.mean(tracesPost,axis=0)[photrats.fs*1:]
        daChanges = np.max(tracesPost[:,photrats.fs*1:],axis=1)-bline
        tracePeakRatMeans.append(np.mean(daChanges))#max(tracePost)-bline)
    return tracePeakRatMeans,missingInds

def plot_ratTracesAtHexChangeDiscovery(photrats,inds,pltCol='k'):
    '''Plot individual rat averages at discovery of newly available and newly blocked paths.
    Return average of rat average traces.'''
    xvals = np.arange(photrats.plot_window[0]*photrats.fs,photrats.plot_window[1]*photrats.fs+1)/photrats.fs
    fig = plt.figure()
    tracePeakRats = photrats.df.loc[inds,"rat"].astype(str).values
    ratmeans,n_PerRat = get_ratTracesAtHexChangeDiscovery(photrats,xvals,inds)
    plt.plot(xvals,np.mean(ratmeans,axis=0),color=pltCol)
    plt.ylim(-1.5,4.9)
    plt.ylabel("mean DA")
    plt.xlabel("time from port entry (s)")
    plt.tight_layout()
    return fig,ratmeans,n_PerRat

def get_ratTracesAtHexChangeDiscovery(photrats,xvals,indices,plotTraces=True):
    tracePeakRats = photrats.df.loc[indices,"rat"].astype(str).values
    ratmeans = []
    n_PerRat = {r:[] for r in photrats.df.rat.unique()}
    for rat in photrats.df.rat.unique():
        tracesPost = get_TracesAroundIndex(photrats,indices[tracePeakRats==rat])
        tracePost = np.mean(tracesPost,axis=0)
        n_PerRat[rat] = len(tracesPost)
        if len(tracesPost)>0:
            ratmeans.append(tracePost)
            if plotTraces:
                plt.plot(xvals,tracePost,color='k',alpha=0.3,lw=1)
    return ratmeans,n_PerRat

def plotFirstEntryHexChange(photrats,adjHexInds,blockedHexAdjInds,legend_on=False):
    #photrats.set_plot_trace("green_z_scored")
    photrats.set_plot_window([-5,5])
    smoothWin = int(photrats.fs/4)
    fig = plt.figure(figsize=(7,5))#(4.8,5))
    xvals = np.arange(photrats.plot_window[0]*photrats.fs,photrats.plot_window[1]*photrats.fs+1)/photrats.fs
    
    avail_traces,blocked_traces = get_availAndBlockedTraces(photrats,adjHexInds,blockedHexAdjInds)
    
    #toplt = pd.Series(np.median(avail_traces,axis=0)).rolling(smoothWin).mean().values
    toplt = pd.Series(np.mean(avail_traces,axis=0)).rolling(smoothWin).mean().values
    topltSem = pd.Series(sem(avail_traces,axis=0)).rolling(smoothWin).mean().values
    plt.plot(xvals,toplt,label="Newly available",color='deeppink',lw=3)
    plt.fill_between(xvals,toplt-topltSem,toplt+topltSem,color='deeppink',alpha=.3)
    toplt = pd.Series(np.mean(blocked_traces,axis=0)).rolling(smoothWin).mean().values
    topltSem = pd.Series(sem(blocked_traces,axis=0)).rolling(smoothWin).mean().values
    plt.plot(xvals,toplt,label="Newly blocked",color='k',ls=':',lw=3)
    plt.fill_between(xvals,toplt-topltSem,toplt+topltSem,color='k',ls=':',alpha=.3)
    plt.xlabel("time from hex entry (s)",fontsize="xx-large")
    #plt.ylabel("median z-scored DA",fontsize="xx-large"")
    plt.ylabel("Mean z-scored DA",fontsize="xx-large")
    plt.axvline(x=0,ls='--',color='k',alpha=.8,lw=1)
    plt.xticks(np.arange(-5,6))
    plt.ylim([-.4,1.3])
    if legend_on:
        plt.legend()
    plt.tight_layout()
    return fig

def plotFirstAdjEntryByEnteredVsIgnored(photrats,enteredInds,ignoredInds,legend_on=False,pltCol1='deeppink',pltCol2='k',ls2='-'):
    #photrats.set_plot_trace("green_z_scored")
    photrats.set_plot_window([-5,5])
    smoothWin = int(photrats.fs/10)
    fig = plt.figure(figsize=(7,5))
    xvals = np.arange(photrats.plot_window[0]*photrats.fs,photrats.plot_window[1]*photrats.fs+1)/photrats.fs
    
    avail_traces,blocked_traces = get_availAndBlockedTraces(photrats,enteredInds,ignoredInds)
    
    #toplt = pd.Series(np.median(avail_traces,axis=0)).rolling(smoothWin).mean().values
    toplt = pd.Series(np.mean(avail_traces,axis=0)).rolling(smoothWin).mean().values
    topltSem = pd.Series(sem(avail_traces,axis=0)).rolling(smoothWin).mean().values
    plt.plot(xvals,toplt,label="entered",color=pltCol1,lw=3)
    plt.fill_between(xvals,toplt-topltSem,toplt+topltSem,color=pltCol1,alpha=.2)
    toplt = pd.Series(np.mean(blocked_traces,axis=0)).rolling(smoothWin).mean().values
    topltSem = pd.Series(sem(blocked_traces,axis=0)).rolling(smoothWin).mean().values
    plt.plot(xvals,toplt,label="ignored",ls=ls2,color=pltCol2,lw=3)
    plt.fill_between(xvals,toplt-topltSem,toplt+topltSem,color=pltCol2,alpha=.2)
    plt.xlabel("Time from changed-hex discovery (s)",fontsize="xx-large")
    plt.ylabel("Mean z-scored DA",fontsize="xx-large")
    plt.axvline(x=0,ls='--',color='k',alpha=.9,lw=1)
    plt.xticks(np.arange(-5,6))
    if legend_on:
        plt.legend()
    plt.tight_layout()
    return fig

def get_availAndBlockedTraces(photrats,adjHexInds,blockedHexAdjInds):
    avail_traces = []
    blocked_traces = []
    for i in adjHexInds:
        if np.isnan(i) or i == -1:
            continue
        avail_traces.append(photrats.df.loc[i+photrats.fs*photrats.plot_window[0]:\
                i+photrats.fs*photrats.plot_window[1],photrats.plot_trace].values)
    for i in blockedHexAdjInds:
        if np.isnan(i):
            continue
        blocked_traces.append(photrats.df.loc[i+photrats.fs*photrats.plot_window[0]:\
                i+photrats.fs*photrats.plot_window[1],photrats.plot_trace].values)
    return np.array(avail_traces),np.array(blocked_traces)

def plot_meanRatDAafterHexEntry(photrats,adjHexInds,blockedHexAdjInds,pltCol1="#27aeef",pltCol2= "#b33dc6"):
    availRatMeans,blockedRatMeans = calc_DaChangeAtHexEntry(photrats,adjHexInds,blockedHexAdjInds)
    blockedMeans = [np.mean(rm) for rm in blockedRatMeans if len(rm)>0]
    availMeans = [np.mean(rm) for rm in availRatMeans if len(rm)>0]
    fig = plt.figure(figsize=(4,5.5))
    plt.bar([0,1],[np.mean(availMeans),\
            np.mean(blockedMeans)],color='k',alpha=0.3)
    plt.ylabel("mean $\Delta$DA",fontsize='xx-large',fontweight='bold')
    plt.xlabel("hex type",fontsize='xx-large',fontweight='bold')
    for r in range(len(availMeans)):
        plt.scatter(x=0,y=np.mean(availMeans[r]),color=pltCol1,marker='o')
        try:
            plt.scatter(x=1,y=np.mean(blockedMeans[r]),color=pltCol2,marker='o')
            plt.plot([0,1],[np.mean(availMeans[r]),np.mean(blockedMeans[r])],color='k',alpha=0.5,lw=1)
        except:
            continue
    print("blocked hex: ")
    sigBlocked = get_sigRats_fromMeanList(blockedMeans)
    print("avail hex: ")
    sigAvail = get_sigRats_fromMeanList(availMeans)
    missingInd = [i for i in range(len(blockedRatMeans)) if len(blockedRatMeans[i])==0]
    if len(missingInd)>0:
        [availMeans.pop(i) for i in missingInd]
    sigPaired = get_sigRatsPaired_from2samples(availMeans,blockedMeans,"greater")
    plot_sigMarkers(sigPaired,0.5,2.4)
    plot_sigMarkers(sigAvail,-.05,2.1)
    plot_sigMarkers(sigBlocked,1,2.1)
    plt.tight_layout()
    return fig

def plot_meanRatDaChangeAfterHexEntry(photrats,adjHexInds,blockedHexAdjInds,pltCol1="#27aeef",pltCol2= "#b33dc6"):
    #availRatMeans,blockedRatMeans = calc_DaChangeAtHexEntry(photrats,adjHexInds,blockedHexAdjInds)
    availMeans,_ = calc_DaPeakDiffAfterNewPathInds(photrats,adjHexInds)
    blockedMeans,missingInd = calc_DaPeakDiffAfterNewPathInds(photrats,blockedHexAdjInds)
    fig = plt.figure(figsize=(4,5.5))
    plt.bar([0,1],[np.mean(availMeans),\
            np.mean(blockedMeans)],color='k',alpha=0.3)
    plt.ylabel("mean $\Delta$DA",fontsize='xx-large',fontweight='bold')
    plt.xlabel("hex type",fontsize='xx-large',fontweight='bold')
    for r in range(len(availMeans)):
        plt.scatter(x=0,y=np.mean(availMeans[r]),color=pltCol1,marker='o')
        try:
            plt.scatter(x=1,y=np.mean(blockedMeans[r]),color=pltCol2,marker='o')
            plt.plot([0,1],[np.mean(availMeans[r]),np.mean(blockedMeans[r])],color='k',alpha=0.5,lw=1)
        except:
            continue
    print("blocked hex: ")
    sigBlocked = get_sigRats_fromMeanList(blockedMeans)
    print("avail hex: ")
    sigAvail = get_sigRats_fromMeanList(availMeans)
    #missingInd = [i for i in range(len(blockedRatMeans)) if len(blockedRatMeans[i])==0]
    if len(missingInd)>0:
        [availMeans.pop(i) for i in missingInd]
    print("paired test")
    sigPaired = get_sigRatsPaired_from2samples(availMeans,blockedMeans,"greater")
    plot_sigMarkers(sigPaired,0.5,2.4)
    plot_sigMarkers(sigAvail,-.05,2.1)
    plot_sigMarkers(sigBlocked,1,2.1)
    plt.tight_layout()
    return fig

def calc_DaChangeAtHexEntry(photrats,adjHexInds,blockedHexAdjInds):
    photrats.set_plot_window([0,1])
    avail_tracesPost,blocked_tracesPost = get_availAndBlockedTraces(photrats,adjHexInds,blockedHexAdjInds)
    photrats.set_plot_window([-1,0])
    avail_tracesPre,blocked_tracesPre = get_availAndBlockedTraces(photrats,adjHexInds,blockedHexAdjInds)
    availRats = photrats.df.loc[adjHexInds,"rat"].astype(str).values
    blockedRats = photrats.df.loc[blockedHexAdjInds,"rat"].astype(str).values
    availRatMeans = []
    blockedRatMeans = []
    for rat in photrats.df.rat.unique():
        availRatMeans.append(np.mean(avail_tracesPost[np.where(availRats==rat)[0]],axis=1)-\
                            np.mean(avail_tracesPre[np.where(availRats==rat)[0]],axis=1))
        blockedRatMeans.append(np.mean(blocked_tracesPost[np.where(blockedRats==rat)[0]],axis=1)-\
                              np.mean(blocked_tracesPre[np.where(blockedRats==rat)[0]],axis=1))
    return availRatMeans,blockedRatMeans

def get_log_pchoos_v_costNben(photrats):
    df = photrats.triframe.copy()
    df['ratcodes'] = df.rat.astype('category').cat.codes
    seshs=df.session.unique()
    for s in range(len(seshs)):
        sdf = df.loc[(df.session==seshs[s])].copy()
        rdf = pd.DataFrame({'pRwdDif':photrats.get_lr_dif_val(\
            sdf,'nom_rwd'),'ldif':photrats.get_lr_dif_val(sdf,'dist')})
        rdf['choose_L'] = sdf.lrchoice.values
        rdf['session']=s
        rdf['rat'] = sdf.ratcodes.values
        rdf['tri'] = sdf.tri.values
        rdf['block'] = sdf.block.values
        rdf.loc[:,"rt-1"] = np.nan
        sdf.reset_index(inplace=True)
        for p in range(3):
            lagInds = sdf.loc[sdf.port==p,:].index
            rdf.loc[lagInds,"rt-1"] = sdf.loc[lagInds,'rwd'].shift(1).values
        if s == 0:
            photrats.regdf = rdf
        else:
            photrats.regdf = photrats.regdf.append(rdf,ignore_index=True)
    photrats.regdf.loc[photrats.regdf.choose_L==2,'choose_L']=np.nan

def plot_choiceRegWeightsByRat(photrats):
    regWeightsByRat = calc_choiceRegWeightsByRat(photrats)
    xvals = ["relative\np(reward)","relative\ndistance"]
    fig = plt.figure(figsize=(3.55,4))
    plt.bar([0,1],np.mean(regWeightsByRat,axis=0)[1:3],color='grey',alpha=.5)
    sns.stripplot(data=regWeightsByRat.loc[:,["relative p(R)","relative distance"]],
        color='k',size=5,marker='D',alpha=.9)
    plt.axhline(y=0,ls='--',color='k',lw=1)
    plt.xticks([0,1],xvals,fontsize="large",fontstyle="italic")
    #plt.ylim(-10,2.2)
    plt.ylabel("Port Choice "+r"$\beta $",fontsize='xx-large',fontweight='bold')
    plt.xlabel("Choice Feature",fontsize='xx-large',fontweight='bold')
    plt.tight_layout()
    fig.savefig(photrats.directory_prefix+"pChooseLregCoefsIndividualRats.pdf")

def calc_choiceRegWeightsByRat_overSessions(photrats):
    try:
        photrats.regdf
    except:
        print("choose left regDf not yet saved. Getting now...")
        get_log_pchoos_v_costNben(photrats)
        add_scaledVarsToRegDf(photrats)
    regWeights = np.zeros((len(photrats.triframe.rat.unique()),3))
    for r in range(len(photrats.regdf.rat.unique())):
        regDf = photrats.regdf.loc[\
            (photrats.regdf.rat==photrats.regdf.rat.unique()[r])&(photrats.regdf.tri>25),\
            ["session","choose_L","rwdDifScaled","lengthDifScaled"]]
        regDf = regDf.loc[regDf.notnull().all(axis=1),:]
        ratRegCoefs = []
        for s in regDf.session.unique():
            y = regDf.loc[regDf.session==s,"choose_L"]
            X = regDf.loc[regDf.session==s,["rwdDifScaled","lengthDifScaled"]]
            mod = LogReg(fit_intercept=True).fit(X,y)
            ratRegCoefs.append(np.concatenate([mod.intercept_,mod.coef_[0]]))
        regWeights[r,:] = np.mean(ratRegCoefs,axis=0)
    return pd.DataFrame(regWeights,columns=["intercept","relative p(R)","relative distance"])

def run_sklearnLogRegWithPval(X,y):
    lm = LogReg(fit_intercept=True).fit(X,y)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)
    new_X = np.append(np.ones((len(X),1)), X, axis=1)
    M_S_E = (sum((y-predictions)**2))/(len(new_X)-len(new_X[0]))
    v_b = M_S_E*(np.linalg.inv(np.dot(new_X.T,new_X)).diagonal())
    s_b = np.sqrt(v_b)
    t_b = params/ s_b
    p_val =[2*(1-stats.t.cdf(np.abs(i),(len(new_X)-len(new_X[0])))) for i in t_b]
    return np.concatenate(params),p_val

def run_smLogRegWithPval(X,y):
    X = np.hstack([np.ones(len(X)).reshape(-1,1),X])
    mod = sm.Logit(y, X).fit()
    coefs = mod.summary2().tables[1]['Coef.'].values
    pvals = mod.summary2().tables[1]["P>|z|"].values
    return coefs,pvals
    
def calc_choiceRegWeightsByRat(photrats):
    try:
        photrats.regdf
    except:
        print("choose left regDf not yet saved. Getting now...")
        get_log_pchoos_v_costNben(photrats)
        add_scaledVarsToRegDf(photrats)
    regWeights = np.zeros((len(photrats.triframe.rat.unique()),3))
    ratSigLevels = np.zeros((len(photrats.triframe.rat.unique()),3))
    for r in range(len(photrats.regdf.rat.unique())):
        regDf = photrats.regdf.loc[\
            (photrats.regdf.rat==photrats.regdf.rat.unique()[r])&(photrats.regdf.tri>25),\
            ["session","choose_L","rwdDifScaled","lengthDifScaled"]]
        regDf = regDf.loc[regDf.notnull().all(axis=1),:]
        y = regDf.loc[:,"choose_L"]
        X = regDf.loc[:,["rwdDifScaled","lengthDifScaled"]]
        betas,pvals = run_smLogRegWithPval(X,y)
        regWeights[r,:] = betas
        ratSigLevels[r,:] = pvals
        ratRegDf = pd.DataFrame(regWeights,columns=["intercept","relative p(R)","relative distance"])
        ratRegDf.loc[:,["intercept p value","relative p(R) p value","relative distance p value"]] = ratSigLevels
    return ratRegDf
    
def get_winStayDf(data):
    data.loc[:,"fromport2"] = data.loc[:,"port"].shift(2).copy()
    data.loc[:,"same2"] = 0
    data.loc[:,"cw"] = 0
    data.loc[data.port!=-100,"cw"] = make_lr(data.loc[data.port!=-100,"port"].values)
    data.loc[data.port==data.fromport2,"same2"] = 1
    vinds0 = data.loc[data.port==0,:].index
    vinds1 = data.loc[data.port==1,:].index
    vinds2 = data.loc[data.port==2,:].index
    stays = []
    prevrwds_taken = []
    prevrwds_alt = []
    same2s = []
    cws = []
    indset = 0
    for inds in [vinds0,vinds1,vinds2]:
        for i in range(1,len(inds)):
            portz = [0,1,2]
            portz.remove(indset)
            prevgoal = data.port.shift(-1)[inds[i-1]]
            currentgoal = data.port.shift(-1)[inds[i]]
            try:
                portz.remove(currentgoal)
            except:
                continue
            try:
                prevrwds_alt.append(data.loc[(data.port.shift(1)==portz[0])&
                    (data.port==currentgoal)&(data.index<inds[i]),'rwd'].values[-1])
            except:
                prevrwds_alt.append(np.nan)
            try:prevrwds_taken.append(data.loc[(data.port.shift(1)==indset)&
                    (data.port==currentgoal)&(data.index<inds[i]),'rwd'].values[-1])
            except:
                prevrwds_taken.append(np.nan)
            same2s.append(data.loc[inds[i]+1,"same2"])
            cws.append(data.loc[inds[i]+1,"cw"])
            stay = 1 if prevgoal == currentgoal else 0
            stays.append(stay)
        indset += 1
    df = pd.DataFrame({'stay':stays,'prev_rwd_taken':prevrwds_taken,
                  "prev_rwd_alt":prevrwds_alt,"same2":same2s,"cw":cws})
    return df

def create_triframe(photrats):
    photrats.get_visinds()
    photrats.triframe = photrats.df.loc[photrats.visinds]
    photrats.triframe['lrchoice'] = make_lr(photrats.triframe.port.values)
    photrats.triframe['lrchoice'] = photrats.triframe.lrchoice.astype("int8")
    photrats.triframe.drop(['x','y','vel',"green_z_scored","beamA",'beamB',
                            'beamC'],axis=1,inplace=True)
    photrats.triframe.reset_index(inplace=True)

def createAndSaveWsdf4R(photrats):
    for s in photrats.triframe.loc[:,"session"].unique():
        wsdf = get_winStayDf(photrats.triframe.loc[photrats.triframe.session==s].copy())
        wsdf.loc[:,'session']=s
        wsdf.loc[:,'rat']=photrats.triframe.loc[photrats.triframe.session==s,'rat'].unique()[0]
        if s == photrats.triframe.session.unique()[0]:
            wsdf_all = wsdf
        else:
            wsdf_all = wsdf_all.append(wsdf,ignore_index=True)
    wsdf_all.to_csv(photrats.directory_prefix+"wslshift.csv")
    return wsdf_all

def create_sameValtChoiceDict(photrats,wsdf_all):
    ratSameValtChoiceProbs = {r:{'pDiffRwdVsOmSame':[],'pDiffRwdVsOmAlt':[]}\
     for r in photrats.triframe.loc[:,"rat"].unique()}
    ratNs = {r:{'pDiffRwdVsOmSameRwd':[],'pDiffRwdVsOmAltRwd':[],\
                'pDiffRwdVsOmSameOm':[],'pDiffRwdVsOmAltOm':[]}\
                 for r in photrats.triframe.loc[:,"rat"].unique()}
    for rat in photrats.triframe.rat.unique():
        for s in photrats.triframe.loc[photrats.triframe.rat==rat,"session"].unique():
            dat = photrats.triframe.loc[(photrats.triframe.rat==rat)&\
                (photrats.triframe.session==s)&(photrats.triframe.same2==1),:]
            ratSameValtChoiceProbs[rat]['pDiffRwdVsOmSame'].append(\
               dat.loc[(dat["samePath_t-1_left"]==1)&(dat["rt-1_left"]==1),"lrchoice"].mean()\
               -dat.loc[(dat["samePath_t-1_left"]==1)&(dat["rt-1_left"]==0),"lrchoice"].mean())
            ratSameValtChoiceProbs[rat]['pDiffRwdVsOmAlt'].append(\
               dat.loc[(dat["samePath_t-1_left"]==0)&(dat["rt-1_left"]==1),"lrchoice"].mean()\
               -dat.loc[(dat["samePath_t-1_left"]==0)&(dat["rt-1_left"]==0),"lrchoice"].mean())
            ratNs[rat]['pDiffRwdVsOmSameRwd'].append(dat.loc[\
                (dat["samePath_t-1_left"]==1)&(dat["rt-1_left"]==1),].shape[0])
            ratNs[rat]['pDiffRwdVsOmSameOm'].append(dat.loc[\
                (dat["samePath_t-1_left"]==1)&(dat["rt-1_left"]==0),].shape[0])
            ratNs[rat]['pDiffRwdVsOmAltRwd'].append(dat.loc[\
                (dat["samePath_t-1_left"]==0)&(dat["rt-1_left"]==1),].shape[0])
            ratNs[rat]['pDiffRwdVsOmAltOm'].append(dat.loc[\
                (dat["samePath_t-1_left"]==0)&(dat["rt-1_left"]==0),].shape[0])
    return ratSameValtChoiceProbs,ratNs

def plotChoosePortSameVAlt(photrats,ratSameValtChoiceProbs):
    fig = plt.figure(figsize=(4,5))
    plt.axhline(y=0,ls='--',color='k',lw=1)
    ratmeans_samePath = []
    ratmeans_altPath = []
    for rat in ratSameValtChoiceProbs:
        ratmeans_samePath.append(np.nanmean(ratSameValtChoiceProbs[rat]["pDiffRwdVsOmSame"]))
        ratmeans_altPath.append(np.nanmean(ratSameValtChoiceProbs[rat]["pDiffRwdVsOmAlt"]))
        plt.scatter(x=["Same\nPath", "Alternative\nPath"],y=[ratmeans_samePath[-1],
                  ratmeans_altPath[-1]],color='k',s=50,marker='D')
    sns.barplot(x=["Same\nPath", "Alternative\nPath"],y=[np.mean(ratmeans_samePath),
                     np.mean(ratmeans_altPath)],palette=["#ff7f0e","#2ca02c"],alpha=.5)
    plt.ylabel("$\Delta$ P(choose port)",fontsize='xx-large')
    plt.suptitle("Port choice increase\nfollowing reward - omission",fontsize='xx-large')
    plt.tight_layout()
    plt.xticks()
    sameSigRats = get_sigRats_fromMeanList(ratmeans_samePath,altString="greater")
    plot_sigMarkers(sameSigRats,-.1,yval=0.31)
    altSigRats = get_sigRats_fromMeanList(ratmeans_altPath,altString="greater")
    plot_sigMarkers(altSigRats,.95,yval=0.31)
    return fig
    

def plot_choiceSigmoidsByRelativeCostsAndBen(photrats,saveFigs=True):
    fig = plot_pChooseVrelativeVar_byRat(photrats,use_scaled=False,use_prob=True)
    fig1 = plot_pChooseVrelativeVar_byRat(photrats,use_scaled=True,use_prob=True)
    fig2 = plot_pChooseVrelativeVar_byRat(photrats,use_scaled=False,use_prob=False)
    fig3 = plot_pChooseVrelativeVar_byRat(photrats,use_scaled=True,use_prob=False)
    ax = fig.axes[0]
    ax.set_xticks([-80,-40,0,40,80])
    ax.set_ylim(0.15,0.76)
    ax.set_yticks(np.linspace(.2,.7,6))
    ax = fig1.axes[0]
    ax.set_xticks([0,.5,1])
    ax.set_ylim(0.15,0.76)
    ax.set_yticks(np.linspace(.2,.7,6))
    ax = fig3.axes[0]
    ax.set_xticks([0,0.5,1])
    ax.set_yticks(np.linspace(0,1,5))
    ax = fig2.axes[0]
    ax.set_xticks([-10,-5,0,5,10])
    ax.set_yticks(np.linspace(0,1,5))
    if saveFigs:
        fig.savefig(photrats.directory_prefix+"rat_logPchooseVpRwd.pdf")
        fig1.savefig(photrats.directory_prefix+"rat_logPchooseVscaledPrwdDif.pdf")
        fig2.savefig(photrats.directory_prefix+"rat_logPchooseVlengthDif.pdf")
        fig3.savefig(photrats.directory_prefix+"rat_logPchooseVscaledLengthDif.pdf")

def create_PortChoiceSameValtDf(wsdf_all):
    omSamePath = []
    rwdSamePath = []
    omAltPath = []
    rwdAltPath = []
    omSameSem = []
    rwdSameSem = []
    omAltSem = []
    rwdAltSem = []
    for r in wsdf_all.rat.unique():
        dat = wsdf_all.loc[(wsdf_all.rat==r)&(wsdf_all.same2==1),:]
        omSamePath.append(dat.loc[(dat["prev_rwd_taken"]==0),"stay"].mean())
        rwdSamePath.append(dat.loc[(dat["prev_rwd_taken"]==1),"stay"].mean())
        omAltPath.append(dat.loc[(dat["prev_rwd_alt"]==0),"stay"].mean())
        rwdAltPath.append(dat.loc[(dat["prev_rwd_alt"]==1),"stay"].mean())
        omSameSem.append(dat.loc[(dat["prev_rwd_taken"]==1),"stay"].sem())
        rwdSameSem.append(dat.loc[(dat["prev_rwd_taken"]==1),"stay"].sem())
        omAltSem.append(dat.loc[(dat["prev_rwd_alt"]==0),"stay"].sem())
        rwdAltSem.append(dat.loc[(dat["prev_rwd_alt"]==1),"stay"].sem())
    choiceSameValtDf = pd.DataFrame({"omSame":omSamePath,"rwdSame":rwdSamePath,"omAlt":omAltPath,"rwdAlt":rwdAltPath,\
                             "omSameSem":omSameSem,"rwdSameSem":rwdSameSem,"omAltSem":omAltSem,"rwdAltSem":rwdAltSem})
    return choiceSameValtDf

def get_sessionProbAndDistChangeInds(photrats):
    probseshs = photrats.triframe.loc[photrats.triframe.session_type=='prob','session'].unique()
    portz = ['a','b','c']
    high2lowProb = {}
    low2highProb = {}
    
    for s in probseshs:
        tdat = photrats.triframe.loc[photrats.triframe.session==s]
        if len(tdat.block.unique())<2:
            continue
        high2lowProb[s] = {b:np.array([0,0,0]) for b in tdat.block.unique()}
        low2highProb[s] = {b:np.array([0,0,0]) for b in tdat.block.unique()}    
        probChanges = np.diff(tdat.loc[tdat.block.diff()!=0,\
            ["nom_rwd_a","nom_rwd_b","nom_rwd_c"]].values,axis=0)
        probIncreasesByBlock = np.array(probChanges>0).astype(int)
        probDecreasesByBlock = np.array(probChanges<0).astype(int)
        for i in range(len(probDecreasesByBlock)):
            low2highProb[s][i+1] = probIncreasesByBlock[i]
            high2lowProb[s][i+1] = probDecreasesByBlock[i]
    
    barseshs = photrats.triframe.loc[photrats.triframe.session_type=='barrier','session'].unique()
    
    pathz = ["AB","AC","BC"]
    high2lowDist = {}
    low2highDist = {}
    for s in barseshs:
        tdat = photrats.triframe.loc[photrats.triframe.session==s]
        if len(tdat.block.unique())<2:
            continue
        high2lowDist[s] = {b:np.array([0,0,0]) for b in tdat.block.unique()}
        low2highDist[s] = {b:np.array([0,0,0]) for b in tdat.block.unique()}    
        distChanges = np.diff(tdat.loc[tdat.block.diff()!=0,\
            ["lenAC","lenBC","lenAB"]].values,axis=0)
        distIncreasesByBlock = np.array(distChanges>0).astype(int)
        distDecreasesByBlock = np.array(distChanges<0).astype(int)
        for i in range(len(distDecreasesByBlock)):
            low2highDist[s][i+1] = distIncreasesByBlock[i]
            high2lowDist[s][i+1] = distDecreasesByBlock[i]
    return high2lowProb,high2lowDist,low2highProb,low2highDist

def get_ProbDecreasePVisit(photrats,rat,high2lowProb):
    high2lowProbPVisit = []
    for s in high2lowProb:
        if photrats.triframe.loc[photrats.triframe.session==s,'rat'].values[0] != rat:
            continue
        tdat = photrats.triframe.loc[photrats.triframe.session==s,:]
        for b in high2lowProb[s]:
            if 1 not in high2lowProb[s][b]:
                continue        
            bports = np.where(high2lowProb[s][b]==1)[0]
            for bport in bports:
                bind = tdat.loc[tdat.block.diff()!=0,:].index[b]
                dat = photrats.triframe.loc[bind-49:bind+50,:].copy()
                dat.loc[:,'choosePort'] = 0
                dat.loc[dat.port==bport,'choosePort']=1
                #mark trials where port of interest is available for choice
                dat.loc[:,"portAvail"] = np.array(dat.port.shift(1)!=bport).astype(int)
                availdat = dat.loc[dat.portAvail==1,:]
                try:
                    blkTransInd = availdat.loc[availdat.block.diff()==1,:].index[0]
                except:
                    continue
                preTransSum = availdat.loc[:blkTransInd,"choosePort"].values
                postTransSum = availdat.loc[blkTransInd:,"choosePort"].values
                pvisitPort = pd.DataFrame({"chosePort":np.full(100,np.nan)})
                try:
                    pvisitPort.loc[blkTransInd-dat.index.min()-len(preTransSum):\
                               blkTransInd-dat.index.min()-1,"chosePort"] = preTransSum
                except:
                    print("didn't work for session "+str(s)+" block "+str(b))
                    continue
                pvisitPort.loc[blkTransInd-dat.index.min():blkTransInd-\
                               dat.index.min()+len(postTransSum)-1,"chosePort"] = postTransSum
                high2lowProbPVisit.append(pvisitPort.values.T[0])
    return np.mean(high2lowProbPVisit,axis=0),len(high2lowProbPVisit)

def get_ProbIncreasePVisit(photrats,rat,low2highProb):
    low2highProbPVisit = []
    for s in low2highProb:
        if photrats.triframe.loc[photrats.triframe.session==s,'rat'].values[0] != rat:
            continue
        tdat = photrats.triframe.loc[photrats.triframe.session==s,:]
        for b in low2highProb[s]:
            if 1 not in low2highProb[s][b]:
                continue        
            bports = np.where(low2highProb[s][b]==1)[0]
            for bport in bports:
                bind = tdat.loc[tdat.block.diff()!=0,:].index[b]
                dat = photrats.triframe.loc[bind-49:bind+50,:].copy()
                dat.loc[:,'choosePort'] = 0
                dat.loc[dat.port==bport,'choosePort']=1
                dat.loc[:,"portAvail"] = np.array(dat.port.shift(1)!=bport).astype(int)
                availdat = dat.loc[dat.portAvail==1,:]
                try:
                    blkTransInd = availdat.loc[availdat.block.diff()==1,:].index[0]
                except:
                    continue
                preTransSum = availdat.loc[:blkTransInd,"choosePort"].values
                postTransSum = availdat.loc[blkTransInd:,"choosePort"].values
                pvisitPort = pd.DataFrame({"chosePort":np.full(100,np.nan)})
                try:
                    pvisitPort.loc[blkTransInd-dat.index.min()-len(preTransSum):\
                               blkTransInd-dat.index.min()-1,"chosePort"] = preTransSum
                except:
                    print("didn't work for session "+str(s)+" block "+str(b))
                    continue
                pvisitPort.loc[blkTransInd-dat.index.min():blkTransInd-\
                               dat.index.min()+len(postTransSum)-1,"chosePort"] = postTransSum
                low2highProbPVisit.append(pvisitPort.values.T[0])
    return np.mean(low2highProbPVisit,axis=0),len(low2highProbPVisit)


pathPortPairs = [["01","10"],["02","20"],["21","12"]]
def get_DistIncreasePVisit(photrats,rat,high2lowDist):
    high2lowDistPVisit = []
    for s in high2lowDist:
        if s not in photrats.triframe.loc[photrats.triframe.rat==rat,"session"].unique():
            continue
        tdat = photrats.triframe.loc[photrats.triframe.session==s,:]
        tdat.loc[:,"path"]= tdat.port.shift(1).fillna(-1).astype("uint8").astype(str).values\
        + tdat.port.astype("uint8").astype(str).values
        for b in high2lowDist[s]:
            if 1 not in high2lowDist[s][b]:
                continue
            bpaths = pathPortPairs[np.where(np.array(high2lowDist[s][b])==1)[0][0]]
            for i in range(len(bpaths)):
                pathports = [int(i) for i in bpaths[i]]
                bind = tdat.loc[tdat.block.diff()!=0].index[b]
                dat = tdat.loc[bind-49:bind+50].copy()
                if len(dat)<100:
                    continue
                dat.loc[:,'choosePath'] = 0
                dat.loc[np.isin(dat.path,bpaths),'choosePath']=1
                dat.loc[:,"pathAvail"] = np.isin(dat.port.shift(1),pathports).astype(int)
                availdat = dat.loc[dat.pathAvail==1,:]
                blkTransInd = availdat.loc[availdat.block.diff()==1,:].index[0]
                preTransSum = availdat.loc[:blkTransInd,"choosePath"].values
                postTransSum = availdat.loc[blkTransInd:,"choosePath"].values
                pTakePath = pd.DataFrame({"takeSum":np.full(100,np.nan)})
                pTakePath.loc[blkTransInd-dat.index.min()-len(preTransSum)+1:\
                               blkTransInd-dat.index.min(),"takeSum"] = preTransSum
                pTakePath.loc[blkTransInd-dat.index.min():blkTransInd-\
                               dat.index.min()+len(postTransSum)-1,"takeSum"] = postTransSum
                high2lowDistPVisit.append(pTakePath.values.T[0])
    return np.mean(high2lowDistPVisit,axis=0),len(high2lowDistPVisit)

def get_DistDecreasePVisit(photrats,rat,low2highDist):
    low2highDistPVisit = []
    for s in low2highDist:
        if s not in photrats.triframe.loc[photrats.triframe.rat==rat,"session"].unique():
            continue
        tdat = photrats.triframe.loc[photrats.triframe.session==s,:]
        tdat.loc[:,"path"]= tdat.port.shift(1).fillna(-1).astype("uint8").astype(str).values\
        + tdat.port.astype("uint8").astype(str).values
        for b in low2highDist[s]:
            if 1 not in low2highDist[s][b]:
                continue
            bpaths = pathPortPairs[np.where(np.array(low2highDist[s][b])==1)[0][0]]
            for i in range(len(bpaths)):
                pathports = [int(i) for i in bpaths[i]]
                bind = tdat.loc[tdat.block.diff()!=0].index[b]
                dat = tdat.loc[bind-49:bind+50].copy()
                if len(dat)<100:
                    continue
                dat.loc[:,'choosePath'] = 0
                dat.loc[np.isin(dat.path,bpaths),'choosePath']=1
                dat.loc[:,"pathAvail"] = np.isin(dat.port.shift(1),pathports).astype(int)
                availdat = dat.loc[dat.pathAvail==1,:]
                blkTransInd = availdat.loc[availdat.block.diff()==1,:].index[0]
                preTransSum = availdat.loc[:blkTransInd,"choosePath"].values
                postTransSum = availdat.loc[blkTransInd:,"choosePath"].values
                pTakePath = pd.DataFrame({"takeSum":np.full(100,np.nan)})
                pTakePath.loc[blkTransInd-dat.index.min()-len(preTransSum)+1:\
                               blkTransInd-dat.index.min(),"takeSum"] = preTransSum
                pTakePath.loc[blkTransInd-dat.index.min():blkTransInd-\
                               dat.index.min()+len(postTransSum)-1,"takeSum"] = postTransSum
                low2highDistPVisit.append(pTakePath.values.T[0])
    return np.mean(low2highDistPVisit,axis=0),len(low2highDistPVisit)

def calc_pChoosePortAtProbBlkTrans(photrats,high2lowProb,low2highProb):
    lows = []
    highs = []
    lowcount = 0
    highcount = 0
    for rat in photrats.triframe.rat.unique().astype(list):
        if rat =="IM-1322":
            continue
        lowVisInfo = get_ProbDecreasePVisit(photrats,rat,high2lowProb)
        highVisInfo = get_ProbIncreasePVisit(photrats,rat,low2highProb)
        lows.append(lowVisInfo[0])
        lowcount += lowVisInfo[1]
        highs.append(highVisInfo[0])
        highcount += highVisInfo[1]
    return lows,highs,lowcount,highcount

def calc_pChoosePathAtBarBlkTrans(photrats,high2lowDist,low2highDist):
    lows = []
    highs = []
    lowcount = 0
    highcount = 0
    for rat in photrats.triframe.rat.unique().astype(list):
        lowVisInfo = get_DistDecreasePVisit(photrats,rat,high2lowDist)
        highVisInfo = get_DistIncreasePVisit(photrats,rat,low2highDist)
        lows.append(lowVisInfo[0])
        lowcount += lowVisInfo[1]
        highs.append(highVisInfo[0])
        highcount += highVisInfo[1]
    return lows,highs,lowcount,highcount

def plot_portChoiceAtProbBlkTrans(photrats,lows,highs,
    smoothWin=5,blineStart=4,legend_on=True):
    xvals = np.arange(-50,50)
    fig = plt.figure(figsize=(5,3.5))
    toplt = np.mean(highs,axis=0)
    toplt = toplt-np.mean(toplt[int(len(toplt)/2)-blineStart:int(len(toplt)/2)+1])
    toplt = pd.Series(toplt).rolling(window=smoothWin).mean()
    plt.plot(xvals,toplt,label = 'p(rwd) Increase',lw=3,color='k')
    plt.fill_between(xvals,toplt+pd.Series(sem(highs)).rolling(smoothWin).mean(),
                     toplt-pd.Series(sem(highs)).rolling(smoothWin).mean(),color='k',alpha=.2)
    toplt = np.mean(lows,axis=0)
    toplt = toplt-np.mean(toplt[int(len(toplt)/2)-blineStart:int(len(toplt)/2)+1])
    toplt = pd.Series(toplt).rolling(window=smoothWin).mean()
    plt.plot(xvals,toplt,label = 'p(rwd) Decrease',lw=3,color='k',ls='--')
    plt.fill_between(xvals,toplt+pd.Series(sem(lows)).rolling(smoothWin).mean(),
                     toplt-pd.Series(sem(lows)).rolling(smoothWin).mean(),color='k',alpha=.2)
    plt.axvline(x=0,color='k')
    plt.ylabel("\u0394 p(visit port)",fontsize='xx-large')
    plt.xlabel("trials from p(rwd) change",fontsize='xx-large')
    ax = plt.gca()
    ax.tick_params(axis="x", direction="inout")
    ax.tick_params(axis="y", direction="inout")
    plt.xlim(0,20)
    if legend_on:
        plt.legend()
    plt.tight_layout()
    return fig


def plot_portChoiceAtBarBlkTrans(photrats,lows,highs,
    smoothWin=5,blineStart=10,legend_on=True):
    xvals = np.arange(-48,52)
    fig = plt.figure(figsize=(4.8,3.1))
    toplt = np.mean(lows,axis=0)
    toplt = toplt-np.mean(toplt[int(len(toplt)/2)-blineStart:int(len(toplt)/2)])
    toplt = pd.Series(toplt).rolling(window=smoothWin).mean()
    plt.plot(xvals,toplt,label = 'Path length decrease',lw=3,color='k')
    plt.fill_between(xvals,toplt+pd.Series(sem(highs)).rolling(smoothWin).mean(),
                     toplt-pd.Series(sem(highs)).rolling(smoothWin).mean(),color='k',alpha=.2)
    toplt = np.mean(highs,axis=0)
    toplt = toplt-np.mean(toplt[int(len(toplt)/2)-blineStart:int(len(toplt)/2)])
    toplt = pd.Series(toplt).rolling(window=smoothWin).mean()
    plt.plot(xvals,toplt,label = 'Path length increase',lw=3,color='k',ls='--')
    plt.fill_between(xvals,toplt+pd.Series(sem(lows)).rolling(smoothWin).mean(),
                     toplt-pd.Series(sem(lows)).rolling(smoothWin).mean(),color='k',alpha=.2)
    plt.axvline(x=0,color='k')
    plt.ylabel("\u0394 p(take path)",fontsize='xx-large')
    plt.xlabel("trials from barrier change",fontsize='xx-large')
    ax = plt.gca()
    ax.tick_params(axis="x", direction="inout")
    ax.tick_params(axis="y", direction="inout")
    plt.xlim(0,20)
    if legend_on:
        plt.legend()
    plt.tight_layout()
    return fig

def plot_logChoosPortVsPrwdDif(photrats):
    plt.figure(figsize=(3.2,3.5))
    for r in photrats.regdf.rat.unique():
        sns.regplot(x="pRwdDif",y="choose_L",data=photrats.regdf.loc[
            (photrats.regdf.rat==r)&(photrats.regdf.tri>25)],logistic=True,
                   scatter_kws={"color":'lightblue','alpha':0.00,"marker":''},line_kws={"color":'k',"lw":2,"alpha":0.4},ci=None)
    sns.regplot(x="pRwdDif",y="choose_L",data=photrats.regdf.loc[
            (photrats.regdf.tri>25)],logistic=True,
                   scatter_kws={"color":'lightblue','alpha':0.00,"marker":''},line_kws={"color":'k',"lw":3,"alpha":1})
    plt.ylabel("P(choose port)",fontsize="xx-large")
    plt.xlabel("relative P(rwd)",fontsize="xx-large")
    plt.ylim(.1,.83)#0.18,0.72)
    plt.xlim(-100,100)
    plt.xticks([-80,-40,0,40,80])
    plt.tight_layout()

def plot_logChoosPortVsDistDif(photrats):
    plt.figure(figsize=(3.2,3.5))
    for r in photrats.regdf.rat.unique():
        sns.regplot(x="ldif",y="choose_L",data=photrats.regdf.loc[photrats.regdf.rat==r],logistic=True,
                   scatter_kws={"color":'lightblue','alpha':0.00,"marker":''},line_kws={"color":'k',"lw":2,"alpha":0.4},ci=None)
    sns.regplot(x="ldif",y="choose_L",data=photrats.regdf,logistic=True,
                   scatter_kws={"color":'lightblue','alpha':0.00,"marker":''},line_kws={"color":'k',"lw":3,"alpha":1.0})
    plt.ylabel("P(choose port)",fontsize="xx-large")
    plt.xlabel("relative path length\n(hexes)",fontsize="xx-large")
    plt.ylim(0,1)
    plt.xticks([-10,-5,0,5,10])
    plt.tight_layout()


def add_scaledVarsToRegDf(photrats):
    photrats.regdf.loc[:,"rwdDifScaled"] = valscale.fit_transform(\
        photrats.regdf.loc[:,"pRwdDif"].values.reshape(-1,1))
    photrats.regdf.loc[:,"lengthDifScaled"] = valscale.fit_transform(\
        photrats.regdf.loc[:,"ldif"].values.reshape(-1,1))

def plot_pChooseVrelativeVar_byRat(photrats,use_scaled=False,use_prob=True):
    if use_prob:
        varString = "rwdDifScaled" if use_scaled else "pRwdDif"
        xstring = "relative value\n(scaled p(reward))" if use_scaled\
        else "relative value\n(reward probability)"
    else:
        varString = "lengthDifScaled" if use_scaled else "ldif"
        xstring = "relative path length\n(transformed hexes)" if use_scaled\
        else "relative path length\n(hexes)"
    fig = plt.figure(figsize=(3.2,3.5))
    for r in photrats.regdf.rat.unique():
        sns.regplot(x=varString,y="choose_L",data=photrats.regdf.loc[photrats.regdf.rat==r],logistic=True,
                   scatter_kws={"color":'lightblue','alpha':0.00,"marker":''},line_kws={"color":'k',"lw":2,"alpha":0.4},ci=None)
    sns.regplot(x=varString,y="choose_L",data=photrats.regdf,logistic=True,
                   scatter_kws={"color":'lightblue','alpha':0.00,"marker":''},line_kws={"color":'k',"lw":3,"alpha":1.0})
    plt.ylabel("P(choose port)",fontsize="xx-large")
    plt.xlabel(xstring,fontsize="xx-large")
    plt.ylim(0,1)
    #plt.xticks([-10,-5,0,5,10])
    plt.tight_layout()
    return fig

def plotRegSigLevel(regSum,x,y,regInd):
    if "Pr(>|z|)" in regSum.columns:
        if regSum.loc[regInd,"Pr(>|z|)"]<0.001:
            plt.text(x=x-.1,y=y,s="***",fontweight='bold',fontsize='xx-large')
        elif regSum.loc[regInd,"Pr(>|z|)"]<0.01:
            plt.text(x=x-.05,y=y,s="**",fontweight='bold',fontsize='xx-large')
        elif regSum.loc[regInd,"Pr(>|z|)"]<0.05:
            plt.text(x=x,y=y,s="*",fontweight='bold',fontsize='xx-large')
        else:
            plt.text(x=x,y=y,s="ns",fontweight='bold',fontsize='xx-large')
    else:
        if regSum.loc[regInd,"t value"]>3.291:
            plt.text(x=x-.1,y=y,s="***",fontweight='bold',fontsize='xx-large')
        elif regSum.loc[regInd,"t value"]>2.58:
            plt.text(x=x-.05,y=y,s="**",fontweight='bold',fontsize='xx-large')
        elif regSum.loc[regInd,"t value"]>1.96:
            plt.text(x=x,y=y,s="*",fontweight='bold',fontsize=25)
        else:
            plt.text(x=x,y=y,s="ns",fontweight='bold',fontsize='xx-large')

def plot_sigMarkers(sigrats,r,yval):
    if sigrats[0]==1:
        plt.text(x=r, y=yval, s='***',fontweight='bold',fontsize='xx-large')
    elif sigrats[1]==1:
        plt.text(x=r, y=yval, s='**',fontweight='bold',fontsize='xx-large')
    elif sigrats[2]==1:
        plt.text(x=r, y=yval, s='*',fontweight='bold',fontsize='xx-large')
    else:
        plt.text(x=r-.25, y=yval, s='ns',fontweight='bold',fontsize='x-large')
        

def add_prevRwdTakenAndAlt(photrats):
    '''add columns indicating whether rat was rewarded
    when previously taking the same (taken) or alternative path
    to the currently chosen port.'''
    photrats.triframe.loc[:,["prev_rwd_taken","prev_rwd_alt"]] = np.nan
    photrats.triframe.loc[:,["prev_rwd_taken","prev_rwd_alt"]] = -100
    photrats.triframe.loc[(photrats.triframe["samePath_t-1"]==0),"prev_rwd_alt"] = \
        photrats.triframe.loc[(photrats.triframe["samePath_t-1"]==0),"rwd"].values
    photrats.triframe.loc[(photrats.triframe["samePath_t-1"]==1),"prev_rwd_taken"] = \
        photrats.triframe.loc[(photrats.triframe["samePath_t-1"]==1),"rwd"].values

def calc_percentDroppedFramesBySesh(photrats,phot_fs=250,frame_rate=15):
    sesh_droppedFramePercents = {s:None for s in photrats.df.session.unique()}
    expected_frame_percentage = (frame_rate/phot_fs)*100
    for s in photrats.df.session.unique():
        dat = photrats.df.loc[photrats.df.session==s,].copy()
        actual_frame_percentage = (dat.loc[dat.x.notnull(),'x'].shape[0]/dat.shape[0])*100
        sesh_droppedFramePercents[s] = ((expected_frame_percentage-actual_frame_percentage)\
                                        /expected_frame_percentage)*100
    return sesh_droppedFramePercents
