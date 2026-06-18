from __future__ import division
import scipy
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import t

def effect_size_cohensD_one(a):
    cohens_d = (np.mean(a)) / np.std(np.array(a))
    print ('Effect Size for One-Sample T-Test')
    return cohens_d


def effect_size_cohensD_dep(a,b):
    dof = (len(a) + len(b) - 2)
    cohens_d = (np.mean(a) - np.mean(b)) / np.std(np.array(a)-np.array(b))
    print ('Effect Size for Dependent T-Test')
    return cohens_d

#Here the pooled standard deviation accounts for unequal sample sizes
def effect_size_cohensD(a,b):
    mean1=np.mean(a)
    std1=np.std(a)
    count1=len(a)
    mean2=np.mean(b)
    std2=np.std(b)
    count2=len(b)
        
    dof = (count1 + count2 - 2)
    cohens_d = (mean1 - mean2) / np.sqrt(((count1 - 1) * std1 ** 2 + (count2 - 1) * std2 ** 2) / dof)
    print ('Effect Size for Independent T-Test')
    return cohens_d

def conf_int(a):
    print ('Confidence Intervall')
    print (DescrStatsW(a).tconfint_mean())
   
def one_sample_tests(df, var):
    print (scipy.stats.shapiro(df[var])) 
    print (scipy.stats.ttest_1samp(df[var],0))
    print (scipy.stats.wilcoxon(df[var]))
    print ('number of samples' + str(len(df[var])))
    print ('mean' + str(np.mean(df[var])))
    print  ('effect')
    print (effect_size_cohensD_one(data[data.asc==0].rel_AU))
    print ('CI')
    print (conf_int(data[data.asc==0].rel_AU))



def two_ind_sample_tests(df1, df2, var): 
    
    ks, pdis = scipy.stats.ks_2samp(df1[var], df2[var])
    if (pdis<0.05):
        print ('distributions of samples is not equal')
        print ('K=: ' + str(ks))
        print ('p=' + str(pdis))
        
    
    _, p1 =scipy.stats.shapiro(df1[var])     
    _, p2 =scipy.stats.shapiro(df2[var]) 


    if ((p1<0.05)|(p2<0.05)):
        print ('not normally distributed: p1=' + str(p1) + ' ' + 'p2=' + str(p2))
        U,p= scipy.stats.mannwhitneyu(df1[var], df2[var])
        print ('Mann-Whitney-U-Test: U=' + str(U) + 'p=' + str(p))
        #Effektsize 
        n1=len(df1[var])
        n2=len(df2[var])
        effectsize=1-(2*U/(n1*n2))
        print ('Effectsize:' + str(effectsize))

        print ("Dataframe one" )
        print ('median:' +str(np.median(df1[var])) )  
        print ("Dataframe two" )
        print ('median:' + str(np.median(df2[var])))

    else:
        print ('normally distributed - now variance homogenity is checked')
        lev, pvar = scipy.stats.levene(df1[var], df2[var])
        if (pvar<0.05):
            print ('variances of samples are not equal!')
            print ('stats-lev: ' + str (lev))
            print ('p:' + str (pvar))
            print (np.var(df1[var]))
            print (np.var(df2[var]))
                        
            print (scipy.stats.ttest_ind(df1[var], df2[var], equal_var=False))
            
        else:    
            print ('variances of samples is equal')
            print (scipy.stats.ttest_ind(df1[var], df2[var]))
            
        print (effect_size_cohensD(df1[var], df2[var]))
        print ("Dataframe one" )
        print ('mean:' +str(np.mean(df1[var])))
        conf_int(df1[var])
        print ("Dataframe two" )
        print ('mean:' + str(np.mean(df2[var])))
        conf_int(df2[var])

# In[ ]:

def two_dep_sample_tests(df1, df2, var): 
    _, pdis = scipy.stats.ks_2samp(df1[var], df2[var])
    if (pdis<0.05):
        print ('distributions of samples is not equal')
    else: 

        _, p1 =scipy.stats.shapiro(df1[var])        
        _, p2 =scipy.stats.shapiro(df2[var]) 

        if ((p1<0.05)|(p2<0.05)):
            print ('not normally distributed: p1=' + str(p1) + ' ' + 'p2=' + str(p2))
            print (scipy.stats.wilcoxon(df1[var], df2[var]))
            #EFFECTSIZE???        
            print ("Dataframe one") 
            print ('median:' +str(np.median(df1[var]))) 
            print ("Dataframe two")
            print ('median:' + str(np.median(df2[var])))

        else:
            print ('normally distributed - now variance homogenity is checked:')
            lev, pvar = scipy.stats.levene(df1[var], df2[var]) # alternative: scipy.stats.bartlett()
            if (pvar<0.05):
                print ('stats:' + lev)
                print ('p-val:' + pvar)
                print ('variances of samples are not equal!')
            else:     
                print (scipy.stats.ttest_rel(df1[var], df2[var], axis=0))
                print (effect_size_cohensD(df1[var], df2[var]))
                print ("Dataframe one" )
                print ('mean:' +str(np.mean(df1[var])))
                conf_int(df1[var])
                print ("Dataframe two" )
                print ('mean:' + str(np.mean(df2[var])))
                conf_int(df2[var])


def correlation(df1, df2):

    _, p1 =scipy.stats.shapiro(df1)        
    _, p2 =scipy.stats.shapiro(df2) 

    if ((p1<0.05)|(p2<0.05)):
        print ('not normally distributed: p1=' + str(p1) + ' ' + 'p2=' + str(p2))
        print (scipy.stats.spearmanr(df1, df2))
    else:
        print ('normally distributed: p1=' + str(p1) + ' ' + 'p2=' + str(p2))
        print (scipy.stats.pearsonr(df1, df2))
        
def rep_anova(df, var):
    #Repeated ANOVA
    df_pos_vpn=df[df.val=='pos'].reset_index().groupby(["vpn"]).mean()
    df_neg_vpn=df[df.val=='neg'].reset_index().groupby(["vpn"]).mean()
    time=len(df_pos_vpn[zyg_s].T)
    vpns=len(df_pos_vpn[zyg_s])

    emotion=2
    muscle=2

    print ('Effect of Muscle and Emotion over time')
    data=np.array(pd.concat([df_pos_vpn[cor_s], df_pos_vpn[zyg_s], 
                             df_neg_vpn[cor_s], df_neg_vpn[zyg_s]], axis=1))
    factorlevels=[emotion, muscle, time]


    print (mne.stats.f_mway_rm(data, factorlevels, effects='A*B'))
    print (mne.stats.f_mway_rm(data, factorlevels, effects='A*B*C'))


    data=pd.concat([df_pos_cor, df_pos_zyg, df_neg_cor, df_neg_zyg], axis=0)

    df_pt = pt.DataFrame(data[['muscle_activity', 'vpn', 'muscle', 'val']])
    (aov)=df_pt.anova('muscle_activity', sub='vpn', wfactors=['muscle', 'val'])
    print(aov)
    
    print ('not ready')
    
def mixedml(data):
    data=df.groupby(['vpn',  'instruction']).mean().reset_index()
    model = sm.MixedLM.from_formula('correct' + " ~ asc+instruction", data, groups=data['vpn']) #, re_formula="~instruction")
    mdf = model.fit()
    print(mdf.summary())
    print ('not ready')
    
def regression(data):
    data=df[df.instruction==1].groupby(['vpn',  'pic']).mean().reset_index()
    model = ols('rel_AU ~  asc', data).fit()
    print (model.summary())
    print ('not ready')

def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated 
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val