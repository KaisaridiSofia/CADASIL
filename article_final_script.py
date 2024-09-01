import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from scipy import stats

from leaspy import Leaspy, Data, AlgorithmSettings, IndividualParameters, __watermark__
# Watermark trace with all packages versions
__watermark__

#load the initial database
all_data = pd.read_excel('/Users/sofia.kaisaridi/Desktop/Données 2022/bases finales (leaspy)/all_corrigee.xlsx')

#count the visits and print
all_data = all_data.drop('visits',axis=1)
all_data['visits'] = all_data.groupby(['ID']).cumcount()+1 #add a counter for the visits of the deceased (lines only with rankin=6)

max_visits = all_data.drop_duplicates(subset=['ID'], keep='last')
print('Patients only with the inclusion visit: ',max_visits.loc[max_visits['visits']==1].shape) 
print('Patients only with more than two visits:', max_visits.loc[max_visits['visits']>=2].shape) 

#split the datasets 
only_INC = max_visits.loc[max_visits['visits']==1]
suivi = max_visits.loc[max_visits['visits']>=2]

print(suivi[['visits']].describe())

#keep only the patients who have more than two visits
all_data = all_data.drop('visits',axis=1)
suivi_ = max_visits[['ID','visits']]
all_data = pd.merge(all_data, suivi_, how='outer', on='ID')
suivi_ = all_data.loc[all_data['visits']>=2]

#inclusion age
suivi_.loc[suivi_['visite']=='INC','TIME'].describe()

#duration of follow-up
INC = suivi_.loc[suivi_['visite']=='INC',['ID','TIME']]
INC = INC.rename(columns = {'TIME' : 'TIME_INC'})

last = suivi_.drop_duplicates(subset=['ID'], keep='last')
last = last[['ID','TIME']]
last = last.rename(columns = {'TIME' : 'TIME_LAST'})

duree_suivi = pd.merge(INC, last, how='outer', on='ID')
duree_suivi['duree_suivi'] = duree_suivi['TIME_LAST'] - duree_suivi['TIME_INC']
duree_suivi

duree_suivi.describe()

#get the longitudinal dataset for leaspy
corriger = suivi_[['ID','TIME','visite','indexscbarthel','indexscrankin','indexnihss','initiation','scoretot','tmtat','tmtbt',
                            'tmtbe','echelv_cor','gblibre_total','ind_react','rdtotal','empan_bis','barragecor_bis','coderep_bis',
                            'vadas_cog']]
corriger.describe()

#transformation
corriger[['indexscbarthel_t']] = 100 - corriger[['indexscbarthel']]
corriger[['initiation_t']] = 37 - corriger[['initiation']]
corriger[['scoretot_t']] = 144 - corriger[['scoretot']]
corriger[['echelv_cor_t']] = 100 - corriger[['echelv_cor']]
corriger[['gblibre_total_t']] = 48 - corriger[['gblibre_total']]
corriger[['ind_react_t']] = 100 - corriger[['ind_react']]
corriger[['rdtotal_t']] = 16 - corriger[['rdtotal']]

#normalisarion
corriger[['indexscbarthel_st']] = corriger[['indexscbarthel_t']]/100
corriger[['indexscrankin_st']] = corriger[['indexscrankin']]/6
corriger[['indexnihss_st']] = corriger[['indexnihss']]/42
corriger[['initiation_st']] = corriger[['initiation_t']]/37
corriger[['scoretot_st']] = corriger[['scoretot_t']]/144
corriger[['tmtat_st']] = corriger[['tmtat']]/180
corriger[['tmtbt_st']] = corriger[['tmtbt']]/300
corriger[['tmtbe_st']] = corriger[['tmtbe']]/24
corriger[['echelv_cor_st']] = corriger[['echelv_cor_t']]/100
corriger[['gblibre_total_st']] = corriger[['gblibre_total_t']]/48
corriger[['ind_react_st']] = (corriger[['ind_react_t']]-1)/(100-1)
corriger[['rdtotal_st']] = corriger[['rdtotal_t']]/16
corriger[['empan_bis_st']] = (corriger[['empan_bis']]-1)/(5-1)
corriger[['barragecor_bis_st']] = (corriger[['barragecor_bis']]-1)/(10-1)
corriger[['coderep_bis_st']] = (corriger[['coderep_bis']]-1)/(10-1)
corriger[['vadas_cog_st']] = corriger[['vadas_cog']]/125

#leaspy 
leaspy_clinical = corriger[['ID','TIME','indexscbarthel_st','indexscrankin_st','indexnihss_st','initiation_st','tmtat_st','tmtbt_st',
                            'tmtbe_st','echelv_cor_st','gblibre_total_st','ind_react_st','rdtotal_st','empan_bis_st','barragecor_bis_st','coderep_bis_st']]
leaspy_clinical = leaspy_clinical.loc[leaspy_clinical['TIME'].notnull()]

leaspy_clinical_data = Data.from_dataframe(leaspy_clinical)
leaspy_clinical_data

algo_settings = AlgorithmSettings('mcmc_saem', 
                                  n_iter=25000,           # n_iter defines the number of iterations
                                  progress_bar=True)     # To display a nice progression bar during calibration

algo_settings.set_logs(
    path='/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/4sources/logs', # Creates a logs file ; if existing, ask if rewrite it
    save_periodicity=50, # Saves the values in csv files every N iterations
    console_print_periodicity=1000, # Displays logs in the console/terminal every N iterations, or None
    plot_periodicity=1000, # Generates the convergence plots every N iterations
    overwrite_logs_folder=True # if True and the logs folder already exists, it entirely overwrites it
)

leaspy = Leaspy('logistic', 
                source_dimension=3, # we have 14 outcomes
                noise_model='gaussian_diagonal', # Optional: To get a noise estimate per feature keep it this way (default)
                )

#run the model
leaspy.fit(leaspy_clinical_data, settings=algo_settings)

#save the model
leaspy.save('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/4sources/model_parameters.json')

#load the model we last ran
leaspy = Leaspy.load('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/model_parameters.json')

# —— Get the average individual parameters
mean_xi = leaspy.model.parameters['xi_mean'].numpy()
mean_tau = leaspy.model.parameters['tau_mean'].numpy()
mean_source = leaspy.model.parameters['sources_mean'].numpy().tolist()
number_of_sources = leaspy.model.source_dimension
mean_sources = [mean_source]*number_of_sources

# —— Store the average individual parameters in a dedicated object
average_parameters = {
    'xi': mean_xi,
    'tau': mean_tau,
    'sources': mean_sources
}

ip_average = IndividualParameters()
ip_average.add_individual_parameters('average', average_parameters)

# Personalization and age reparametrization
settings_personalization = AlgorithmSettings('scipy_minimize', use_jacobian=True)
ip = leaspy.personalize(leaspy_clinical_data, settings=settings_personalization)

_, param_ind = ip.to_pytorch()

import torch
torch_tensor = torch.tensor(leaspy_clinical['TIME'].values).float()

xi = param_ind['xi'].numpy()
tau = param_ind['tau'].numpy()
timepoints = torch_tensor.numpy()

xi_df = pd.DataFrame(xi)
xi_df.rename(columns = {0:'xi'}, inplace = True)
tau_df = pd.DataFrame(tau)
tau_df.rename(columns = {0:'tau'}, inplace = True)
timepoints_df = leaspy_clinical[['ID','TIME']]
timepoints_df.columns.values[0:2] =['ID', 'timepoints'] #rename the columns

xi_df['id'] = range(1, len(xi_df) + 1)
tau_df['id'] = range(1, len(tau_df) + 1)
timepoints_df['id'] = timepoints_df.groupby(['ID']).ngroup()+1

#create a dataset with ALL the patients but only the data at the inclusion visit
clinical_INC = all_data.loc[all_data['visite']=='INC',['id','age']]
clinical_INC.columns.values[0:3] =['ID','t0']

#create a dataset with all the observations with the individual parameters and the timepoints of each observation
df = pd.merge(timepoints_df, xi_df, how='left', on='id')
df = pd.merge(df, tau_df, how='left', on='id')
df_ip = ip.to_dataframe()

#merge to add the t0 and calculate the time difference and parametrisation
df = pd.merge(df, clinical_INC, how='left', on=['ID'])
df['diff'] = df['timepoints'] - df['t0']
df['time_re'] = df['tau'] + df['diff']

#get the correct order for the graph
leaspy.model.features = ['indexscbarthel', #0
 'indexscrankin', #1
 'indexnihss', #2
 'initiation', #3
 'tmtat', #4
 'tmtbt', #5
 'tmtbe', #6
 'echelv', #7
 'gblibre_total', #8
 'ind_react', #9
 'rdtotal', #10
 'empan', #11
 'barragecor', #12
 'coderep'] #13


order = [13,11,12,5,8,1,4,7,3,9,10,6,2,0]

#get the graph
timepoints = np.linspace(20, 105, 85)
values = leaspy.estimate({'average': timepoints}, ip_average)

def plot_trajectory(timepoints, reconstruction, observations=None, *, 
                    xlabel='Years', ylabel='Normalized feature value'):

    if observations is not None:
        ages = observations.index.values
    
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 1)
    colors = ['red', 'lime', 'blue', 'black', 'fuchsia', 'yellow', 'aqua',
              'silver', 'maroon', 'olive', 'green', 'teal', 'navy', 'purple']
    
    for c, name, val in zip(colors, leaspy.model.features, reconstruction.T):
        plt.plot(timepoints, val, label=name, c=c, linewidth=3)
        if observations is not None:
            plt.plot(ages, observations[name], c=c, marker='o', markersize=12, 
                     linewidth=1, linestyle=':')
    
    plt.xlim(min(timepoints), max(timepoints))
    plt.xlabel('Reparametrized age')
    plt.ylabel(ylabel)
    plt.vlines(x=leaspy.model.parameters['tau_mean'].numpy(), ymin=0, ymax=1, colors='black', linestyles='dotted')
    plt.text(59,0.9, s='tau_mean')
    plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
               bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.title('Population progression')

    plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/leaspy_multi_sub_scores.png',
                bbox_inches='tight',dpi=1200)
    
plot_trajectory(timepoints, values['average'])
plt.show()

#create groups according to disease time
param_pop = pd.DataFrame(leaspy.model.features)
param_pop['g'] = pd.DataFrame(leaspy.model.parameters['g'].numpy())
param_pop['v0'] = pd.DataFrame(leaspy.model.parameters['v0'].numpy())
param_pop['p0'] = pd.DataFrame(1/(1+leaspy.model.parameters['g'].numpy()))

param_pop['g_re'] = np.exp(param_pop['g'])
param_pop['p0_re'] = pd.DataFrame(1/(1+param_pop['g_re']))

param_pop.columns.values[0:1] =['feature']
print('g<0 : ','\n', param_pop.sort_values(by='p0_re', ascending=False).loc[param_pop['g']<0], '\n')
print('0<g<1 : ','\n', param_pop.sort_values(by='p0_re', ascending=False).loc[(param_pop['g']>0) & (param_pop['g']<1)], '\n')
print('1<g : ','\n', param_pop.sort_values(by='p0_re', ascending=False).loc[(param_pop['g']>1)], '\n')

#get the sources
sources_df = pd.DataFrame(np.array(param_ind['sources']))
sources_df['id'] = range(1, len(sources_df) + 1)
sources_df.columns.values[0:3] = ['sources_0','sources_1','sources_2']
sources_df

#merge with all the individual parmeters
df = pd.merge(df, sources_df, how='left', on='id')
ind_param = df[['ID','xi','tau','sources_0','sources_1','sources_2']]
ind_param = ind_param.drop_duplicates('ID')
ind_param = ind_param.dropna()
ind_param

#function to transform the sources to omegas
import numpy as np

from leaspy import IndividualParameters



def append_spaceshifts_to_individual_parameters_dataframe(df_individual_parameters, leaspy, *, time_homogeneous=False):
    r"""
    Returns a new dataframe with space shift columns.

    Parameters
    ----------
    df_individual_parameters: pandas.DataFrame
        Dataframe of the individual parameters, idexed by subjects' identifiers. Each row corresponds to an individual.
    leaspy: leaspy.Leaspy
        Leaspy object with initialized model.
    time_homogeneous: bool
        If False, returns the standard space-shift (A * s_i). If True, normalizes the w_i by a coefficient in order to
        get a space_shift homogeneous to a time (in years). Default False.

    Returns
    -------
    df_ip: pandas.DataFrame
        Copy of the initial dataframe with additional columns being the space shifts of the individuals.
    """
    df_ip = df_individual_parameters.copy()

    sources = df_ip[['sources_' + str(i) for i in range(leaspy.model.source_dimension)]].values.T
    spaceshifts = np.dot(leaspy.model.attributes.mixing_matrix, sources)

    # Time normalization (if applicable)
    if time_homogeneous:
        if leaspy.type == 'logistic':
            coeff = 1/leaspy.model.attributes.velocities.numpy()
            spaceshifts = np.multiply(spaceshifts.T, coeff).T

        else:
            raise NotImplementedError(
                f'The time normalization has not been implemented for the {leaspy.model.type()} model')

    # Appending the space-shifts to the dataframe
    for i, spaceshift_coord in enumerate(spaceshifts):
        name_ss = f'w_{i}'
        if time_homogeneous:
            name_ss += 'c'
        df_ip[name_ss] = spaceshift_coord


    return df_ip

#a dataframe with all the individual parameters
params_all = append_spaceshifts_to_individual_parameters_dataframe(ind_param, leaspy, time_homogeneous=False)
params_all

#load the covariables but lose the previously calculated xi and tau
cov = pd.read_excel('/Users/sofia.kaisaridi/Desktop/Résultats/covariables_par_patient.xlsx')
cov = cov.drop(['tau','xi'], axis=1)

#transform the categories when needed
cov['smoking_new'] = 'no' #ancient and non-smokers
cov.loc[cov['smoking_t']=='current', 'smoking_new'] = 'yes' #only current smokers 
cov['HTA_new'] = 'no' #other or none
cov.loc[cov['FDRV_HTA']=='HTA', 'HTA_new'] = 'HTA' #HTA

cov[['ID','sexe_t','neduc_t','smoking_new','HTA_new']]

#merge them all
cov_params = params_all
cov_params = pd.merge(params_all, cov, how='left', on='ID')
cov_params

print(cov_params[['sexe_t']].value_counts())
print(cov_params[['sexe_t']].value_counts(normalize=True).round(2))

print(cov_params[['neduc_t']].value_counts())
print(cov_params[['neduc_t']].value_counts(normalize=True).round(2))

print(cov_params[['smoking_t']].value_counts())
print(cov_params[['smoking_t']].value_counts(normalize=True).round(2))

print(cov_params[['calcool_t']].value_counts())
print(cov_params[['calcool_t']].value_counts(normalize=True).round(2))

print(cov_params[['Domaine']].value_counts())
print(cov_params[['Domaine']].value_counts(normalize=True).round(2))

print(cov_params[['hcholes']].value_counts())
print(cov_params[['hcholes']].value_counts(normalize=True).round(2))

print(cov_params[['diab']].value_counts())
print(cov_params[['diab']].value_counts(normalize=True).round(2))

print(cov_params[['FDRV']].value_counts())
print(cov_params[['FDRV']].value_counts(normalize=True).round(2))

print(cov_params[['FDRV_presence']].value_counts())
print(cov_params[['FDRV_presence']].value_counts(normalize=True).round(2))

print(cov_params[['FDRV_howmany']].value_counts())
print(cov_params[['FDRV_howmany']].value_counts(normalize=True).round(2))

print(cov_params[['smoking_new']].value_counts())
print(cov_params[['smoking_new']].value_counts(normalize=True).round(2))

print(cov_params[['HTA_new']].value_counts())
print(cov_params[['HTA_new']].value_counts(normalize=True).round(2))

baseline = corriger.loc[corriger['visite']=='INC']
baseline.iloc[0:395,1:20].describe().round(2)

#missing data
(leaspy_clinical.isna().sum()/2007).round(2)

# Mann-Whitney U test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal

# compare samples
print('Significant results for TAU from Mann-Whitney and Kruskal-Wallis tests \n')
print('Gender')
stat, p = mannwhitneyu(cov_params.loc[cov_params['sexe_t']=='men','tau'], 
                       cov_params.loc[cov_params['sexe_t']=='women','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Education level')
stat, p = mannwhitneyu(cov_params.loc[cov_params['neduc_t']=='>=13 years','tau'], 
                       cov_params.loc[cov_params['neduc_t']=='<13 years','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Smoking')
stat, p = kruskal(cov_params.loc[cov_params['smoking_t']=='current','tau'], 
                  cov_params.loc[cov_params['smoking_t']=='former','tau'], 
                  cov_params.loc[cov_params['smoking_t']=='never','tau'])
if p<0.05: 
    print('3 categories : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_t']=='current','tau'], 
                       cov_params.loc[cov_params['smoking_t']=='former','tau'])
if p<0.05: 
    print('Current vs Former : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_t']=='current','tau'], 
                       cov_params.loc[cov_params['smoking_t']=='never','tau'])
if p<0.05: 
    print('Current vs Never : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_t']=='former','tau'], 
                       cov_params.loc[cov_params['smoking_t']=='never','tau'])
if p<0.05: 
    print('Former vs Never : Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Alcohol')
stat, p = kruskal(cov_params.loc[cov_params['calcool_t']=='<2 glasses','tau'], 
                  cov_params.loc[cov_params['calcool_t']=='>2 glasses','tau'], 
                  cov_params.loc[cov_params['calcool_t']=='never','tau'])
if p<0.05: 
    print('3 categories : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='<2 glasses','tau'], 
                       cov_params.loc[cov_params['calcool_t']=='>2 glasses','tau'])
if p<0.05: 
    print('<2 glasses vs >2 glasses : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='<2 glasses','tau'], 
                       cov_params.loc[cov_params['calcool_t']=='never','tau'])
if p<0.05: 
    print('<2 glasses vs never : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='>2 glasses','tau'], 
                       cov_params.loc[cov_params['calcool_t']=='never','tau'])
if p<0.05: 
    print('>2 glasses vs never Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('FDRV_presence')
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_presence']=='at least 1','tau'], 
                       cov_params.loc[cov_params['FDRV_presence']=='None','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('FDRV_HTA')
stat, p = kruskal(cov_params.loc[cov_params['FDRV_HTA']=='HTA','tau'], 
                  cov_params.loc[cov_params['FDRV_HTA']=='other','tau'], 
                  cov_params.loc[cov_params['FDRV_HTA']=='None','tau'])
if p<0.05: 
    print('3 categories : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_HTA']=='HTA','tau'], 
                       cov_params.loc[cov_params['FDRV_HTA']=='other','tau'])
if p<0.05: 
    print('HTA vs other : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_HTA']=='HTA','tau'], 
                       cov_params.loc[cov_params['FDRV_HTA']=='None','tau'])
if p<0.05: 
    print('HTA vs None : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_HTA']=='other','tau'], 
                       cov_params.loc[cov_params['FDRV_HTA']=='None','tau'])
if p<0.05: 
    print('Other vs None : Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('FDRV_howmany')
stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','tau'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='2','tau'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='1','tau'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='None','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='2','tau'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='1','tau'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='None','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','tau'],
                  cov_params.loc[cov_params['FDRV_howmany']=='1','tau'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='None','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','tau'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='2','tau'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='None','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','tau'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='2','tau'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='1','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','tau'], 
                       cov_params.loc[cov_params['FDRV_howmany']=='2','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','tau'], 
                       cov_params.loc[cov_params['FDRV_howmany']=='1','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','tau'],
                       cov_params.loc[cov_params['FDRV_howmany']=='None','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='2','tau'],
                       cov_params.loc[cov_params['FDRV_howmany']=='1','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='2','tau'], 
                       cov_params.loc[cov_params['FDRV_howmany']=='None','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='1','tau'],
                       cov_params.loc[cov_params['FDRV_howmany']=='None','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Domaine')
stat, p = mannwhitneyu(cov_params.loc[cov_params['Domaine']=='1-6','tau'], 
                       cov_params.loc[cov_params['Domaine']=='7-34','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Significant results for TAU from Mann-Whitney and Kruskal-Wallis tests with new grouping \n')
print('Smoking new (grouping current as yes, former and never as no)')

stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_new']=='yes','tau'],
                       cov_params.loc[cov_params['smoking_new']=='no','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('HTA new (grouping HTA as yes, other and none as no)')

stat, p = mannwhitneyu(cov_params.loc[cov_params['HTA_new']=='HTA','tau'], 
                       cov_params.loc[cov_params['HTA_new']=='no','tau'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Significant results for XI from Mann-Whitney and Kruskal-Wallis tests \n')
print('Gender')
stat, p = mannwhitneyu(cov_params.loc[cov_params['sexe_t']=='men','xi'], 
                       cov_params.loc[cov_params['sexe_t']=='women','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Education level')
stat, p = mannwhitneyu(cov_params.loc[cov_params['neduc_t']=='>=13 years','xi'], 
                       cov_params.loc[cov_params['neduc_t']=='<13 years','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Smoking')
stat, p = kruskal(cov_params.loc[cov_params['smoking_t']=='current','xi'], 
                  cov_params.loc[cov_params['smoking_t']=='former','xi'], 
                  cov_params.loc[cov_params['smoking_t']=='never','xi'])
if p<0.05: 
    print('3 categories : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_t']=='current','xi'], 
                       cov_params.loc[cov_params['smoking_t']=='former','xi'])
if p<0.05: 
    print('Current vs Former : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_t']=='current','xi'], 
                       cov_params.loc[cov_params['smoking_t']=='never','xi'])
if p<0.05: 
    print('Current vs Never : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_t']=='former','xi'], 
                       cov_params.loc[cov_params['smoking_t']=='never','xi'])
if p<0.05: 
    print('Former vs Never : Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Alcohol')
stat, p = kruskal(cov_params.loc[cov_params['calcool_t']=='<2 glasses','xi'], 
                  cov_params.loc[cov_params['calcool_t']=='>2 glasses','xi'], 
                  cov_params.loc[cov_params['calcool_t']=='never','xi'])
if p<0.05: 
    print('3 categories : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='<2 glasses','xi'], 
                       cov_params.loc[cov_params['calcool_t']=='>2 glasses','xi'])
if p<0.05: 
    print('<2 glasses vs >2 glasses : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='<2 glasses','xi'], 
                       cov_params.loc[cov_params['calcool_t']=='never','xi'])
if p<0.05: 
    print('<2 glasses vs never : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='>2 glasses','xi'], 
                       cov_params.loc[cov_params['calcool_t']=='never','xi'])
if p<0.05: 
    print('>2 glasses vs never Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('FDRV_presence')
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_presence']=='at least 1','xi'], 
                       cov_params.loc[cov_params['FDRV_presence']=='None','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('FDRV_HTA')
stat, p = kruskal(cov_params.loc[cov_params['FDRV_HTA']=='HTA','xi'], 
                  cov_params.loc[cov_params['FDRV_HTA']=='other','xi'], 
                  cov_params.loc[cov_params['FDRV_HTA']=='None','xi'])
if p<0.05: 
    print('3 categories : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_HTA']=='HTA','xi'], 
                       cov_params.loc[cov_params['FDRV_HTA']=='other','xi'])
if p<0.05: 
    print('HTA vs other : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_HTA']=='HTA','xi'], 
                       cov_params.loc[cov_params['FDRV_HTA']=='None','xi'])
if p<0.05: 
    print('HTA vs None : Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_HTA']=='other','xi'], 
                       cov_params.loc[cov_params['FDRV_HTA']=='None','xi'])
if p<0.05: 
    print('Other vs None : Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('FDRV_howmany')
stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','xi'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='2','xi'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='1','xi'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='None','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='2','xi'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='1','xi'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='None','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','xi'],
                  cov_params.loc[cov_params['FDRV_howmany']=='1','xi'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='None','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','xi'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='2','xi'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='None','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','xi'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='2','xi'], 
                  cov_params.loc[cov_params['FDRV_howmany']=='1','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','xi'], 
                       cov_params.loc[cov_params['FDRV_howmany']=='2','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','xi'], 
                       cov_params.loc[cov_params['FDRV_howmany']=='1','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','xi'],
                       cov_params.loc[cov_params['FDRV_howmany']=='None','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='2','xi'],
                       cov_params.loc[cov_params['FDRV_howmany']=='1','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='2','xi'], 
                       cov_params.loc[cov_params['FDRV_howmany']=='None','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='1','xi'],
                       cov_params.loc[cov_params['FDRV_howmany']=='None','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Domaine')
stat, p = mannwhitneyu(cov_params.loc[cov_params['Domaine']=='1-6','xi'], 
                       cov_params.loc[cov_params['Domaine']=='7-34','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('Significant results for XI from Mann-Whitney and Kruskal-Wallis tests with new grouping \n')
print('Smoking new (grouping current as yes, former and never as no)')

stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_new']=='yes','xi'],
                       cov_params.loc[cov_params['smoking_new']=='no','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

print('HTA new (grouping HTA as yes, other and none as no)')

stat, p = mannwhitneyu(cov_params.loc[cov_params['HTA_new']=='HTA','xi'], 
                       cov_params.loc[cov_params['HTA_new']=='no','xi'])
if p<0.05: 
    print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

cov_params.loc[cov_params['HTA_new']=='HTA','HTA_new'] = 'yes'

for i in range(3):
    print('Source', i , '\n' )
    #print(cov_params[['sources_'+str(i)]].head())
    print('Significant results from Mann-Whitney and Kruskal-Wallis tests \n')
    print('Gender')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['sexe_t']=='men','sources_'+str(i)], 
                           cov_params.loc[cov_params['sexe_t']=='women','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('Education level')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['neduc_t']=='>=13 years','sources_'+str(i)], 
                           cov_params.loc[cov_params['neduc_t']=='<13 years','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('Smoking')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_new']=='yes','sources_'+str(i)],
                       cov_params.loc[cov_params['smoking_new']=='no','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('Alcohol')
    stat, p = kruskal(cov_params.loc[cov_params['calcool_t']=='<2 glasses','sources_'+str(i)], 
                      cov_params.loc[cov_params['calcool_t']=='>2 glasses','sources_'+str(i)], 
                      cov_params.loc[cov_params['calcool_t']=='never','sources_'+str(i)])
    if p<0.05: 
        print('3 categories : Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='<2 glasses','sources_'+str(i)], 
                           cov_params.loc[cov_params['calcool_t']=='>2 glasses','sources_'+str(i)])
    if p<0.05: 
        print('<2 glasses vs >2 glasses : Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='<2 glasses','sources_'+str(i)], 
                           cov_params.loc[cov_params['calcool_t']=='never','sources_'+str(i)])
    if p<0.05: 
        print('<2 glasses vs never : Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='>2 glasses','sources_'+str(i)], 
                           cov_params.loc[cov_params['calcool_t']=='never','sources_'+str(i)])
    if p<0.05: 
        print('>2 glasses vs never Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('FDRV_presence')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_presence']=='at least 1','sources_'+str(i)], 
                           cov_params.loc[cov_params['FDRV_presence']=='None','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('FDRV_HTA')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['HTA_new']=='yes','sources_'+str(i)], 
                       cov_params.loc[cov_params['HTA_new']=='no','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('FDRV_howmany')
    stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','sources_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='2','sources_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='1','sources_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='None','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='2','sources_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='1','sources_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='None','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','sources_'+str(i)],
                      cov_params.loc[cov_params['FDRV_howmany']=='1','sources_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='None','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','sources_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='2','sources_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='None','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','sources_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='2','sources_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='1','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','sources_'+str(i)], 
                           cov_params.loc[cov_params['FDRV_howmany']=='2','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','sources_'+str(i)], 
                           cov_params.loc[cov_params['FDRV_howmany']=='1','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','sources_'+str(i)],
                           cov_params.loc[cov_params['FDRV_howmany']=='None','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='2','sources_'+str(i)],
                           cov_params.loc[cov_params['FDRV_howmany']=='1','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='2','sources_'+str(i)], 
                           cov_params.loc[cov_params['FDRV_howmany']=='None','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='1','sources_'+str(i)],
                           cov_params.loc[cov_params['FDRV_howmany']=='None','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('Domaine')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['Domaine']=='1-6','sources_'+str(i)], 
                           cov_params.loc[cov_params['Domaine']=='7-34','sources_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')
    
for i in range(14):
    print('Omegas', i , '\n' )
    #print(cov_params[['sources_'+str(i)]].head())
    print('Significant results from Mann-Whitney and Kruskal-Wallis tests \n')
    print('Gender')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['sexe_t']=='men','w_'+str(i)], 
                           cov_params.loc[cov_params['sexe_t']=='women','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('Education level')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['neduc_t']=='>=13 years','w_'+str(i)], 
                           cov_params.loc[cov_params['neduc_t']=='<13 years','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('Smoking')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_new']=='yes','w_'+str(i)],
                       cov_params.loc[cov_params['smoking_new']=='no','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('Alcohol')
    stat, p = kruskal(cov_params.loc[cov_params['calcool_t']=='<2 glasses','w_'+str(i)], 
                      cov_params.loc[cov_params['calcool_t']=='>2 glasses','w_'+str(i)], 
                      cov_params.loc[cov_params['calcool_t']=='never','w_'+str(i)])
    if p<0.05: 
        print('3 categories : Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='<2 glasses','w_'+str(i)], 
                           cov_params.loc[cov_params['calcool_t']=='>2 glasses','w_'+str(i)])
    if p<0.05: 
        print('<2 glasses vs >2 glasses : Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='<2 glasses','w_'+str(i)], 
                           cov_params.loc[cov_params['calcool_t']=='never','w_'+str(i)])
    if p<0.05: 
        print('<2 glasses vs never : Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['calcool_t']=='>2 glasses','w_'+str(i)], 
                           cov_params.loc[cov_params['calcool_t']=='never','w_'+str(i)])
    if p<0.05: 
        print('>2 glasses vs never Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('FDRV_presence')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_presence']=='at least 1','w_'+str(i)], 
                           cov_params.loc[cov_params['FDRV_presence']=='None','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('FDRV_HTA')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['HTA_new']=='yes','w_'+str(i)], 
                       cov_params.loc[cov_params['HTA_new']=='no','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('FDRV_howmany')
    stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','w_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='2','w_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='1','w_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='None','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='2','w_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='1','w_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='None','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','w_'+str(i)],
                      cov_params.loc[cov_params['FDRV_howmany']=='1','w_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='None','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','w_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='2','w_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='None','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = kruskal(cov_params.loc[cov_params['FDRV_howmany']=='>2','w_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='2','w_'+str(i)], 
                      cov_params.loc[cov_params['FDRV_howmany']=='1','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','w_'+str(i)], 
                           cov_params.loc[cov_params['FDRV_howmany']=='2','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','w_'+str(i)], 
                           cov_params.loc[cov_params['FDRV_howmany']=='1','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='>2','w_'+str(i)],
                           cov_params.loc[cov_params['FDRV_howmany']=='None','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='2','w_'+str(i)],
                           cov_params.loc[cov_params['FDRV_howmany']=='1','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='2','w_'+str(i)], 
                           cov_params.loc[cov_params['FDRV_howmany']=='None','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(cov_params.loc[cov_params['FDRV_howmany']=='1','w_'+str(i)],
                           cov_params.loc[cov_params['FDRV_howmany']=='None','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    print('Domaine')
    stat, p = mannwhitneyu(cov_params.loc[cov_params['Domaine']=='1-6','w_'+str(i)], 
                           cov_params.loc[cov_params['Domaine']=='7-34','w_'+str(i)])
    if p<0.05: 
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')
    
##GMM univariate
test = cov_params[['tau','xi','sources_0','sources_1','sources_2']]

## training gaussian mixture model 
from sklearn.mixture import GaussianMixture

n_components = np.arange(1, 11)
models = [GaussianMixture(n_components=n, covariance_type='full', random_state=0).fit(test)
          for n in n_components]

plt.plot(n_components, [m.bic(test) for m in models], label='BIC')
plt.plot(n_components, [m.aic(test) for m in models], label='AIC')
plt.xticks(ticks=np.arange(1, 10, step=1))
plt.legend(loc='best')
plt.xlabel('number of subgroups')
plt.ylabel('criterion value')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM_criteria.png',dpi=1200)
plt.show()

print([m.bic(test) for m in models])
print([m.aic(test) for m in models])

## 1000 repetitions to stabilize the results
test = cov_params[['tau','xi','sources_0','sources_1','sources_2']]
frame = pd.DataFrame(test)
gmm = GaussianMixture(n_components=2, random_state=121223)

for i in range(1001):
    
    gmm.fit(test)

    ##predictions from gmm
    labels = gmm.predict(test)
    frame['labels'] = labels
    frame['tau'] = cov_params['tau']
    frame['xi'] = cov_params['xi']
    cluster_centers = frame.groupby(['labels'])['tau'].agg(['mean', 'count', 'std'])

    cluster_centers['clusters'] = [0,1]
    cluster_centers['name'] = ['inter','inter']
    cluster_centers['min'] = cluster_centers['mean'].min()
    cluster_centers['max'] = cluster_centers['mean'].max()
    cluster_centers.loc[cluster_centers['mean']==cluster_centers['min'], 'name'] = 'precoce'
    cluster_centers.loc[cluster_centers['mean']==cluster_centers['max'], 'name'] = 'tardif'
    cluster_centers['iteration_'+str(i)] = cluster_centers['name']
    
    gmm_labels = pd.DataFrame(labels)
    gmm_labels.rename(columns={0:'clusters'},inplace=True)
    cluster_centers_keep = cluster_centers[['clusters','iteration_'+str(i)]]
    gmm_labels['id'] = gmm_labels.index +1
    gmm_labels = pd.merge(gmm_labels, cluster_centers_keep, how='left', on='clusters')
    gmm_labels = gmm_labels.drop('clusters', axis=1)
    if i==1:
        gmm_labels_all = gmm_labels
    if i>1:
        gmm_labels_all = pd.merge(gmm_labels_all, gmm_labels, how='left', on='id')

    
## cluster_centers
gmm_labels_all['mode_value'] = gmm_labels_all.mode(axis=1)

frame.drop('labels', axis=1)
frame['labels'] = gmm_labels_all['mode_value']

frame.groupby(['labels'])['tau','xi'].agg(['mean', 'count', 'std'])

frame = append_spaceshifts_to_individual_parameters_dataframe(frame, leaspy, time_homogeneous=False)
frame

print(frame.groupby(['labels'])['w_0','w_1','w_2','w_3','w_4'].agg(['mean', 'count', 'std']))
print(frame.groupby(['labels'])['w_5','w_6','w_7','w_8','w_9'].agg(['mean', 'count', 'std']))
print(frame.groupby(['labels'])['w_10','w_11','w_12','w_13'].agg(['mean', 'count', 'std']))

print(frame.groupby(['labels'])['w_0','w_1','w_2','w_3','w_4'].agg(['mean']))
print(frame.groupby(['labels'])['w_5','w_6','w_7','w_8','w_9'].agg(['mean']))
print(frame.groupby(['labels'])['w_10','w_11','w_12','w_13'].agg(['mean']))

import seaborn as sns
#distribution plots
frame['labels_eng'] = 'inter'
frame.loc[frame['labels']=='precoce','labels_eng'] = 'early'
frame.loc[frame['labels']=='tardif','labels_eng'] = 'late'

fig = plt.figure(figsize=(8,8))
sns.jointplot(data=frame, x='xi', y='tau', hue='labels_eng',palette=['darkgreen','firebrick'])
plt.legend(title='subgroup')
plt.xlabel('Progression rate - ξ values')
plt.ylabel('Time shift - τ values (years)')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/final_figures/Fig4_GMM2_joint_distribution_scatter.png',
                bbox_inches='tight',dpi=1200)
plt.show()


sns.displot(frame, x='tau', kind='kde', hue='labels_eng',palette=['darkgreen','firebrick'])
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_tau.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='xi', kind='kde', hue='labels_eng',palette=['darkgreen','firebrick'])
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_xi.png',
                bbox_inches='tight',dpi=1200)
plt.show()

frame[['subgroup']] = frame[['labels_eng']]

sns.displot(frame, x='w_0', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega Barthel')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_Barthel.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_1', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega Rankin')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_Rankin.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_2', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega NIHSS')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_NIHSS.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_3', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega MDRSinitiation')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2__distribution_w_Initiation.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_4', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega TMTAT')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_TMTAT.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_5', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega TMTBT')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_TMTBT.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_6', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega TMTBE')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_TMTBE.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_7', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega EQVAS')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_Echelv.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_8', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega GBfree')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_Gblibre_total.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_9', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega GBcueing')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_Ind_react.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_10', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega GBdelayed')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_Rdtotal.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_11', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.ylabel('density')
plt.xlabel('omega BackwardDigit')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_Empan.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_12', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega DigitCancel')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_Barragecor.png',
                bbox_inches='tight',dpi=1200)
plt.show()

sns.displot(frame, x='w_13', kind='kde', hue='subgroup',palette=['darkgreen','firebrick'])
plt.xlabel('omega SymbolDigit')
plt.ylabel('density')
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/outputs_leaspy/25000/sub_scores/3sources/GMM2_distribution_w_Coderep.png',
                bbox_inches='tight',dpi=1200)
plt.show()

## Mann-Whitney


for i in frame.columns.values[0:2]:
    x=frame.loc[frame['labels_eng']=='early', i]
    y=frame.loc[frame['labels_eng']=='late', i]
    
    #if ks_2samp(x, y).pvalue < 0.05 :
    print('Mann-Whitney U tests for the differences in the distributions of', i, '\n')
    print('p-value :', mannwhitneyu(x, y).pvalue)
    print('\n')
        
for i in frame.columns.values[6:20]:
    x=frame.loc[frame['labels_eng']=='early', i]
    y=frame.loc[frame['labels_eng']=='late', i]
    
    #if ks_2samp(x, y).pvalue < 0.05 :
    print('Mann-Whitney U tests for the differences in the distributions of', i, '\n')
    print('p-value :', mannwhitneyu(x, y).pvalue)
    print('\n')
    
x=frame.loc[frame['labels_eng']=='early', 'tau']
y=frame.loc[frame['labels_eng']=='late', 'tau']
print(x.median())
print(x.quantile([0, 0.25, 0.751,1]))

for i in frame.columns.values[0:2]:
    x=frame.loc[frame['labels_eng']=='early', i]
    y=frame.loc[frame['labels_eng']=='late', i]
    
    print(i, '\n')
    print('early : \n', 'median : ', np.round(x.median(),2), '\n', 'IQR :', np.round(x.quantile([0, 0.25, 0.75, 1]),2))
    print('\n')
    
    print('late : \n', 'median : ', np.round(y.median(),2), '\n', 'IQR :', np.round(y.quantile([0, 0.25, 0.75, 1]),2))
    print('\n')

for i in frame.columns.values[6:20]:
    x=frame.loc[frame['labels_eng']=='early', i]
    y=frame.loc[frame['labels_eng']=='late', i]
    
    print(i, '\n')
    print('early : \n', 'median : ', np.round(x.median(),2), '\n', 'IQR :', np.round(x.quantile([0, 0.25, 0.75, 1]),2))
    print('\n')
    
    print('late : \n', 'median : ', np.round(y.median(),2), '\n', 'IQR :', np.round(y.quantile([0, 0.25, 0.75, 1]),2))
    print('\n')

#3D figgure to combine omega differences
from mpl_toolkits.mplot3d import Axes3D

frame.loc[frame['labels_eng']=='early','c'] = 'firebrick'
frame.loc[frame['labels_eng']=='late','c'] = 'forestgreen'

fig, ax = plt.subplots(1,3,figsize=(18,6),subplot_kw=dict(projection='3d'))
fig.suptitle('Intermarker spacing parameters - ω values',y=0.95, fontsize=16)

x = frame['w_1']
y = frame['w_0']
z = frame['w_2']
c = frame['c']
ax[0].set_xlabel('Modified Rankin Scale')
ax[0].set_ylabel('Barthel Index')
ax[0].set_zlabel('NIHSS Index')
ax[0].scatter3D(x,y,z, c = c)
ax[0].set_title('A. Motor disability, Dependency \n& Focal Neurological Deficits')
red_patch = mpatches.Patch(color='firebrick', label='early')
green_patch = mpatches.Patch(color='forestgreen', label='late')
ax[0].legend(handles=[red_patch,green_patch], loc='best', title='subgroup')

x = frame['w_11']
y = frame['w_12']
z = frame['w_13']
c = frame['c']
ax[1].set_xlabel('Backward Digit Span')
ax[1].set_ylabel('Digit Cancellation Test')
ax[1].set_zlabel('Symbol Digit Task')
ax[1].scatter3D(x,y,z, c = c)
ax[1].set_title('B. Vadas-Cog subscores')

red_patch = mpatches.Patch(color='firebrick', label='early')
green_patch = mpatches.Patch(color='forestgreen', label='late')
ax[1].legend(handles=[red_patch,green_patch], loc='best', title='subgroup')

x = frame['w_7']
y = frame['w_9']
z = frame['w_10']
c = frame['c']
ax[2].set_xlabel('EQVAS Quality of Life')
ax[2].set_ylabel('GB Index of Sensitivity to Cueing')
ax[2].set_zlabel('GB Delayed Total recall')
ax[2].scatter3D(x,y,z ,c = c)
ax[2].set_title('C. Life quality & Memory')
red_patch = mpatches.Patch(color='firebrick', label='early')
green_patch = mpatches.Patch(color='forestgreen', label='late')
ax[2].legend(handles=[red_patch,green_patch], loc='best', title='subgroup')

plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/final_figures/Fig5_GMM2_example_omegas.png',
                bbox_inches='tight',dpi=1200)
plt.show()

#chi-2 entre covariables
from scipy.stats import chi2_contingency

frame[['sexe_t']] = cov[['sexe_t']]
frame[['neduc_t']] = cov[['neduc_t']]
frame[['smoking_new']] = cov[['smoking_new']]
frame[['HTA_new']] = cov[['HTA_new']]
frame[['Risk_new']] = cov[['Risk_new']]
frame[['Domaine']] = cov[['Domaine']]

data_cont=pd.crosstab(frame['labels'],frame['sexe_t'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['neduc_t'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['smoking_new'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['HTA_new'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['Risk_new'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['Domaine'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

#get the leaspy curves by group

early = frame.loc[frame['labels_eng']=='early',['labels_eng','xi','tau','sources_0','sources_1','sources_2']]
late = frame.loc[frame['labels_eng']=='late',['labels_eng','xi','tau','sources_0','sources_1','sources_2']]

print('We have', early.shape[0] ,'individuals in the early group (', round((early.shape[0]/395)*100),'%) with means:')
print(early.mean())


print('We have', late.shape[0],'individuals in the late group (', round((late.shape[0]/395)*100),'%) with means:')
print(late.mean())

ip_average_early = {'average': {'xi': 0.729912,
  'tau': 58.274002,
  'sources': [ -0.307865, 0.048520, -0.708221]}}

ip_average_late = {'average': {'xi': -0.178480,
  'tau': 69.326569,
  'sources': [ 0.254624, -0.021589, 0.392392]}}

print(ip_average_early)
print(ip_average_late)
print(ip_average._individual_parameters)

#get the two graphs 
timepoints = np.linspace(20, 105, 85)
values_early = leaspy.estimate({'average': timepoints}, ip_average_early)
values_late = leaspy.estimate({'average': timepoints}, ip_average_late)

def plot_trajectory(timepoints, reconstruction, observations=None, *, ax=None,
                    xlabel='Years', ylabel='Normalized feature value', tau_mean, title):
    
    if observations is not None:
        ages = observations.index.values
    
    plt.figure(figsize=(10, 5))

    plt.ylim(0, 1)
    colors = ['red', 'lime', 'blue', 'black', 'fuchsia', 'yellow', 'aqua',
              'silver', 'maroon', 'olive', 'green', 'teal', 'navy', 'purple']
    
    for c, name, val in zip(colors, leaspy.model.features, reconstruction.T):
        plt.plot(timepoints, val, label=name, c=c, linewidth=3)
        if observations is not None:
            plt.plot(ages, observations[name], c=c, marker='o', markersize=12, 
                     linewidth=1, linestyle=':')
    
    plt.xlim(min(timepoints), max(timepoints))
    plt.xlabel('Reparametrized age')
    plt.vlines(x=tau_mean, ymin=0, ymax=1, colors='black', linestyles='dotted')
    plt.ylabel(ylabel)
    plt.grid()
    plt.title(title)

plot_trajectory(timepoints, values_early['average'],tau_mean = 58.274002, title='Population progression for patients in the early group')
plot_trajectory(timepoints, values_late['average'],tau_mean = 69.326569, title='Population progression for patients in the late group')

#function to plot only a subgroup
def plot_trajectory_cluster(timepoints, reconstruction, observations=None, *, ax=None,
                    xlabel='Years', ylabel='normalized feature value', tau_mean, title):
    
    tau_text = tau_mean-8
    if observations is not None:
        ages = observations.index.values

    plt.ylim(0, 1)
    colors = ['red', 'lime', 'blue', 'black', 'fuchsia', 'yellow', 'aqua',
              'silver', 'maroon', 'olive', 'green', 'teal', 'navy', 'purple']
    
    for c, name, val in zip(colors, features_eng, reconstruction.T):
        plt.plot(timepoints, val, label=name, c=c, linewidth=3)
        if observations is not None:
            plt.plot(ages, observations[name], c=c, marker='o', markersize=12, 
                     linewidth=1, linestyle=':')
    
    plt.xlim(min(timepoints), max(timepoints))
    plt.xlabel('disease age')
    plt.vlines(x=tau_mean, ymin=0, ymax=1, colors='black', linestyles='dotted')
    plt.text(tau_text,0.95, s='tau_mean')
    plt.ylabel(ylabel)
    plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
               bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.title(title)

fig, ax = plt.subplots(figsize=(10,10), nrows=2, ncols=1)
plt.subplot(2, 1, 1)
plot_trajectory_cluster(timepoints, values_early['average'], tau_mean = 58.274002, title='Population progression for patients in the early group')
plt.subplot(2, 1, 2)
plot_trajectory_cluster(timepoints, values_late['average'],tau_mean = 69.326569, title='Population progression for patients in the late group')
plt.tight_layout()
plt.show() 

features_eng = ['Barthel Index', 'Modified Rankin Scale','NIHSS Index','MDRS Initiation','TMT A Time','TMT B Time','TMT B Errors',
                'EQVAS Quality of life ','GB Total Free Recall','GB Index of Sensitivity to Cueing','GB Delayed Total Recall',
                'VADAS-Cog Backward Digit Span', 'VADAS-Cog Digit Cancellation Task', 'VADAS-Cog Symbol Digit Test']
features_eng

#minor modifications to the function to plot the population trajectory
def plot_trajectory_all(timepoints, reconstruction, observations=None, *, 
                    xlabel='Years', ylabel='normalized feature value'):

    if observations is not None:
        ages = observations.index.values
    
    plt.ylim(0, 1)
    colors = ['red', 'lime', 'blue', 'black', 'fuchsia', 'yellow', 'aqua',
              'silver', 'maroon', 'olive', 'green', 'teal', 'navy', 'purple']
    
    for c, name, val in zip(colors, features_eng, reconstruction.T):
        plt.plot(timepoints, val, label=name, c=c, linewidth=3)
        if observations is not None:
            plt.plot(ages, observations[name], c=c, marker='o', markersize=12, 
                     linewidth=1, linestyle=':')
    
    plt.xlim(min(timepoints), max(timepoints))
    plt.xlabel('disease age')
    plt.ylabel(ylabel)
    plt.vlines(x=leaspy.model.parameters['tau_mean'].numpy(), ymin=0, ymax=1, colors='black', linestyles='dotted')
    plt.text(60,0.9, s='tau_mean')
    plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
               bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.title('A')
    
#final figure
fig, ax = plt.subplots(figsize=(12,12), nrows=3, ncols=1)

plt.subplot(3, 1, 1)
plot_trajectory_all(timepoints, values['average'])
plt.subplot(3, 1, 2)
plot_trajectory_cluster(timepoints, values_early['average'], tau_mean = 58.274002, title='B')
plt.subplot(3, 1, 3)
plot_trajectory_cluster(timepoints, values_late['average'],tau_mean = 69.326569, title='C')
plt.tight_layout()
plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/final_figures/leaspy_multi_sub_scores_population+clusters.png',
            bbox_inches='tight',
            dpi=1200)
plt.show()

### BOXPLOTS

cov_params_long = cov_params.iloc[:,0:20]

sexe_men = cov_params.loc[cov_params['sexe_t'] == 'men',['ID','sexe_t']]
sexe_men[['covariable']] = 'gender'
sexe_men = sexe_men.rename(columns={'sexe_t': 'value'})

sexe_women = cov_params.loc[cov_params['sexe_t'] == 'women',['ID','sexe_t']]
sexe_women[['covariable']] = 'gender'
sexe_women = sexe_women.rename(columns={'sexe_t': 'value'})

educ_low = cov_params.loc[cov_params['neduc_t'] == '<13 years',['ID','neduc_t']]
educ_low[['covariable']] = 'education level'
educ_low = educ_low.rename(columns={'neduc_t': 'value'})

educ_high = cov_params.loc[cov_params['neduc_t'] == '>=13 years',['ID','neduc_t']]
educ_high[['covariable']] = 'education level'
educ_high = educ_high.rename(columns={'neduc_t': 'value'})

smoking_yes = cov_params.loc[cov_params['smoking_new'] == 'yes',['ID','smoking_new']]
smoking_yes[['covariable']] = 'smoking'
smoking_yes = smoking_yes.rename(columns={'smoking_new': 'value'})

smoking_no = cov_params.loc[cov_params['smoking_new'] == 'no',['ID','smoking_new']]
smoking_no[['covariable']] = 'smoking'
smoking_no = smoking_no.rename(columns={'smoking_new': 'value'})

HTA_yes = cov_params.loc[cov_params['HTA_new'] == 'yes',['ID','HTA_new']]
HTA_yes[['covariable']] = 'HTA'
HTA_yes = HTA_yes.rename(columns={'HTA_new': 'value'})

HTA_no = cov_params.loc[cov_params['HTA_new'] == 'no',['ID','HTA_new']]
HTA_no[['covariable']] = 'HTA'
HTA_no = HTA_no.rename(columns={'HTA_new': 'value'})

domaine_high = cov_params.loc[cov_params['Domaine'] == '1-6',['ID','Domaine']]
domaine_high[['covariable']] = 'Domaine'
domaine_high = domaine_high.rename(columns={'Domaine': 'value'})

domaine_low = cov_params.loc[cov_params['Domaine'] == '7-34',['ID','Domaine']]
domaine_low[['covariable']] = 'Domaine'
domaine_low = domaine_low.rename(columns={'Domaine': 'value'})

cov_params_long_men = pd.merge(cov_params_long, sexe_men, on='ID', how='right')
cov_params_long_women = pd.merge(cov_params_long, sexe_women, on='ID', how='right')

cov_params_long_low = pd.merge(cov_params_long, educ_low, on='ID', how='right')
cov_params_long_high = pd.merge(cov_params_long, educ_high, on='ID', how='right')

cov_params_long_smoking = pd.merge(cov_params_long, smoking_yes, on='ID', how='right')
cov_params_long_nosmoking = pd.merge(cov_params_long, smoking_no, on='ID', how='right')

cov_params_long_HTA = pd.merge(cov_params_long, HTA_yes, on='ID', how='right')
cov_params_long_noHTA = pd.merge(cov_params_long, HTA_no, on='ID', how='right')

cov_params_long_domaine_high = pd.merge(cov_params_long, domaine_high, on='ID', how='right')
cov_params_long_domaine_low = pd.merge(cov_params_long, domaine_low, on='ID', how='right')

sexe = pd.concat([cov_params_long_men, cov_params_long_women], axis=0)

educ = pd.concat([cov_params_long_low, cov_params_long_high], axis=0)

smoking = pd.concat([cov_params_long_smoking, cov_params_long_nosmoking], axis=0)

HTA = pd.concat([cov_params_long_HTA, cov_params_long_noHTA], axis=0)

domaine = pd.concat([cov_params_long_domaine_high, cov_params_long_domaine_low], axis=0)

total = pd.concat([sexe, educ, smoking, HTA, domaine], axis=0)
total.sort_values(by=['ID','covariable']).iloc[0:10,:]

##boxplots of tau and xi

my_colors = {'men': 'firebrick', 'women': 'forestgreen',
            '>=13 years': 'forestgreen', '<13 years': 'firebrick',
            'yes': 'firebrick', 'no' : 'forestgreen',
            '1-6': 'firebrick', '7-34': 'forestgreen'}

x1 = total.loc[total['value']=='men','tau']
x2 = total.loc[total['value']=='women','tau']
x3 = total.loc[total['value']=='<13 years','tau']
x4 = total.loc[total['value']=='>=13 years','tau']
x5 = total.loc[(total['value']=='yes') & (total['covariable']=='smoking'),'tau']
x6 = total.loc[(total['value']=='no') & (total['covariable']=='smoking'),'tau']
x7 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'tau']
x8 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'tau']
x9 = total.loc[total['value']=='1-6','tau']
x10 = total.loc[total['value']=='7-34','tau']

labels = ['men$^*$', 'women$^*$', 'low education$^*$', 'high education$^*$', 'smoking$^*$', 'no smoking$^*$', 'hypertension$^*$', 'no hypertension$^*$', 'mutation in 1-6$^*$', 'mutation in 7-34$^*$']


index_as_texts = labels
custom_index = [1,2,4,5,7,8,10,11,13,14]

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8))
medianprops = dict(color='black')
bplot1 = ax1.boxplot([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10],
                     vert=False, patch_artist=True,  # fill with color
                     medianprops=medianprops, widths=0.9,
                    positions = [1,2,4,5,7,8,10,11,13,14]) 
ax1.set_xlabel('τ values')
ax1.set_yticks(custom_index, index_as_texts, fontsize=11)

colors = ['firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen','firebrick', 'forestgreen']

for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)

x1 = total.loc[total['value']=='men','xi']
x2 = total.loc[total['value']=='women','xi']
x3 = total.loc[total['value']=='<13 years','xi']
x4 = total.loc[total['value']=='>=13 years','xi']
x5 = total.loc[(total['value']=='yes') & (total['covariable']=='smoking'),'xi']
x6 = total.loc[(total['value']=='no') & (total['covariable']=='smoking'),'xi']
x7 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'xi']
x8 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'xi']
x9 = total.loc[total['value']=='1-6','xi']
x10 = total.loc[total['value']=='7-34','xi']
labels = ['men$^*$', 'women$^*$', 'low education', 'high education', 'smoking', 'no smoking', 'hypertension', 'no hypertension', 'mutation in 1-6', 'mutation in 7-34']

index_as_texts = labels

bplot1 = ax2.boxplot([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10],
                     vert=False, patch_artist=True,  # fill with color
                     medianprops=medianprops, widths=0.9,
                    positions = [1,2,4,5,7,8,10,11,13,14]) 
ax2.set_xlabel('ξ values')
ax2.set_yticks(custom_index, index_as_texts, fontsize=11)
ax2.set_ylim(0.5,14.5)

colors = ['firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen','firebrick', 'forestgreen']
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)

plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/final_figures/Fig2_boxplots_temporal_covariables_reviewed.png',
            dpi=1200)
plt.show()

##omegas boxplot

import matplotlib.patches as mpatches
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
medianprops = dict(color='black')

x1 = total.loc[total['value']=='<13 years','w_0']
x2 = total.loc[total['value']=='>=13 years','w_0']
x3 = total.loc[total['value']=='<13 years','w_1']
x4 = total.loc[total['value']=='>=13 years','w_1']
x5 = total.loc[total['value']=='<13 years','w_2']
x6 = total.loc[total['value']=='>=13 years','w_2']
x7 = total.loc[total['value']=='<13 years','w_3']
x8 = total.loc[total['value']=='>=13 years','w_3']
x9 = total.loc[total['value']=='<13 years','w_4']
x10 = total.loc[total['value']=='>=13 years','w_4']
x11 = total.loc[total['value']=='<13 years','w_5']
x12 = total.loc[total['value']=='>=13 years','w_5']
x13 = total.loc[total['value']=='<13 years','w_6']
x14 = total.loc[total['value']=='>=13 years','w_6']
x15 = total.loc[total['value']=='<13 years','w_7']
x16 = total.loc[total['value']=='>=13 years','w_7']
x17 = total.loc[total['value']=='<13 years','w_8']
x18 = total.loc[total['value']=='>=13 years','w_8']
x19 = total.loc[total['value']=='<13 years','w_9']
x20 = total.loc[total['value']=='>=13 years','w_9']
x21 = total.loc[total['value']=='<13 years','w_10']
x22 = total.loc[total['value']=='>=13 years','w_10']
x23 = total.loc[total['value']=='<13 years','w_11']
x24 = total.loc[total['value']=='>=13 years','w_11']
x25 = total.loc[total['value']=='<13 years','w_12']
x26 = total.loc[total['value']=='>=13 years','w_12']
x27 = total.loc[total['value']=='<13 years','w_13']
x28 = total.loc[total['value']=='>=13 years','w_13']

labels = ['Barthel$^*$','Rankin$^*$','NIHSS$^*$','MDRS Initiation$^*$',
          'TMT A Time$^*$','TMT B Time$^*$','TMT B Errors$^*$','Quality of life',
          'GB Free$^*$','GB Cueing$^*$','GB Delayed$^*$',
          'Backward Digit$^*$','Digit Cancellation$^*$','Symbol Digit$^*$']
index_as_texts = labels
custom_index = [1.5,4.5,7.5,10.5,13.5,16.5,19.5,
                22.5, 25.5, 28.5, 31.5, 34.5, 37.5, 40.5]

bplot1 = ax1.boxplot([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14,
                     x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28],
                     vert=False, patch_artist=True,  # fill with color
                     medianprops=medianprops, widths=0.9,
                     positions = [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 
                                  25,26, 28,29, 31,32, 34,35, 37,38, 40,41])
ax1.set_xlabel('ω values')
ax1.set_yticks(custom_index, index_as_texts, fontsize=9, rotation=45)

red_patch = mpatches.Patch(color='firebrick', label='low education')
green_patch = mpatches.Patch(color='forestgreen', label='high education')
ax1.legend(handles=[red_patch,green_patch], loc='lower right')
ax1.set_ylim(0.5,41.5)

# fill with colors
colors = ['firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen','firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen', 'firebrick', 'forestgreen']

for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)


x1 = total.loc[total['value']=='men','w_0']
x2 = total.loc[total['value']=='women','w_0']
x3 = total.loc[total['value']=='men','w_1']
x4 = total.loc[total['value']=='women','w_1']
x5 = total.loc[total['value']=='men','w_2']
x6 = total.loc[total['value']=='women','w_2']
x7 = total.loc[total['value']=='men','w_3']
x8 = total.loc[total['value']=='women','w_3']
x9 = total.loc[total['value']=='men','w_4']
x10 = total.loc[total['value']=='women','w_4']
x11 = total.loc[total['value']=='men','w_5']
x12 = total.loc[total['value']=='women','w_5']
x13 = total.loc[total['value']=='men','w_6']
x14 = total.loc[total['value']=='women','w_6']
x15 = total.loc[total['value']=='men','w_7']
x16 = total.loc[total['value']=='women','w_7']
x17 = total.loc[total['value']=='men','w_8']
x18 = total.loc[total['value']=='women','w_8']
x19 = total.loc[total['value']=='men','w_9']
x20 = total.loc[total['value']=='women','w_9']
x21 = total.loc[total['value']=='men','w_10']
x22 = total.loc[total['value']=='women','w_10']
x23 = total.loc[total['value']=='men','w_11']
x24 = total.loc[total['value']=='women','w_11']
x25 = total.loc[total['value']=='men','w_12']
x26 = total.loc[total['value']=='women','w_12']
x27 = total.loc[total['value']=='men','w_13']
x28 = total.loc[total['value']=='women','w_13']

labels = ['Barthel','Rankin','NIHSS$^*$','MDRS Initiation',
          'TMT A Time','TMT B Time','TMT B Errors','Quality of life$^*$',
          'GB Free$^*$','GB Cueing$^*$','GB Delayed$^*$',
          'Backward Digit','Digit Cancellation','Symbol Digit']

index_as_texts = labels
custom_index = [1.5,4.5,7.5,10.5,13.5,16.5,19.5,
                22.5, 25.5, 28.5, 31.5, 34.5, 37.5, 40.5]

bplot1 = ax2.boxplot([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14,
                     x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28],
                     vert=False, patch_artist=True,  # fill with color
                     medianprops=medianprops, widths=0.9,
                     positions = [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 
                                  25,26, 28,29, 31,32, 34,35, 37,38, 40,41])

ax2.set_xlabel('ω values')
ax2.set_ylabel('')
ax2.set_yticks(custom_index, index_as_texts, fontsize=9, rotation=45)

red_patch = mpatches.Patch(color='firebrick', label='men')
green_patch = mpatches.Patch(color='forestgreen', label='women')
ax2.legend(handles=[red_patch,green_patch], loc='lower right')
ax2.set_ylim(0.5,41.5)

colors = ['firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen','firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen', 'firebrick', 'forestgreen']
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)


x1 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_0']
x2 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_0']
x3 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_1']
x4 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_1']
x5 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_2']
x6 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_2']
x7 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_3']
x8 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_3']
x9 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_4']
x10 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_4']
x11 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_5']
x12 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_5']
x13 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_6']
x14 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_6']
x15 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_7']
x16 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_7']
x17 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_8']
x18 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_8']
x19 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_9']
x20 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_9']
x21 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_10']
x22 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_10']
x23 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_11']
x24 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_11']
x25 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_12']
x26 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_12']
x27 = total.loc[(total['value']=='yes') & (total['covariable']=='HTA'),'w_13']
x28 = total.loc[(total['value']=='no') & (total['covariable']=='HTA'),'w_13']

labels = ['Barthel','Rankin$^*$','NIHSS$^*$','MDRS Initiation',
          'TMT A Time','TMT B Time','TMT B Errors','Quality of life',
          'GB Free$^*$','GB Cueing','GB Delayed',
          'Backward Digit','Digit Cancellation$^*$','Symbol Digit']

index_as_texts = labels
custom_index = [1.5,4.5,7.5,10.5,13.5,16.5,19.5,
                22.5, 25.5, 28.5, 31.5, 34.5, 37.5, 40.5]

bplot1 = ax3.boxplot([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14,
                     x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28],
                     vert=False, patch_artist=True,  # fill with color
                     medianprops=medianprops, widths=0.9,
                     positions = [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 
                                  25,26, 28,29, 31,32, 34,35, 37,38, 40,41])

ax3.set_xlabel('ω values')
ax3.set_ylabel('')
ax3.set_yticks(custom_index, index_as_texts, fontsize=9, rotation=45)

red_patch = mpatches.Patch(color='firebrick', label='hypertension')
green_patch = mpatches.Patch(color='forestgreen', label='no hypertension')
ax3.legend(handles=[red_patch,green_patch], loc='lower right')
ax3.set_ylim(0.5,41.5)

colors = colors = ['firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen','firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen', 'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
         'firebrick', 'forestgreen', 'firebrick', 'forestgreen']
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)

plt.savefig('/Users/sofia.kaisaridi/Desktop/Résultats/premiers resultats-article/final_figures/Fig3_boxplots_omegas_covariables_reviewed.png',
            dpi=1200)
plt.show()

#get the numerical differences
frame.groupby('sexe_t').median().round(3)
frame.groupby('sexe_t').quantile([0.25, 0.75]).round(2)

print('p-values for the gender')
for i in range(14):
    print('Omegas', i , '\n' )
    #print(cov_params[['sources_'+str(i)]].head())
    stat, p = mannwhitneyu(cov_params.loc[cov_params['sexe_t']=='men','w_'+str(i)], 
                           cov_params.loc[cov_params['sexe_t']=='women','w_'+str(i)])
    
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

frame.groupby('neduc_t').median().round(3)
frame.groupby('neduc_t').quantile([0.25, 0.75]).round(2)

print('p-values for the education')
print('XI', '\n' )
stat, p = mannwhitneyu(cov_params.loc[cov_params['neduc_t']=='>=13 years','xi'], 
                       cov_params.loc[cov_params['neduc_t']=='<13 years','xi'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')
for i in range(14):

    print('Omegas', i , '\n' )
    stat, p = mannwhitneyu(cov_params.loc[cov_params['neduc_t']=='>=13 years','w_'+str(i)], 
                           cov_params.loc[cov_params['neduc_t']=='<13 years','w_'+str(i)])
    
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')
    
frame.groupby('smoking_new').median().round(3)
frame.groupby('smoking_new').quantile([0.25, 0.75]).round(2)

print('p-values for smoking')
print('XI', '\n' )
stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_new']=='yes','xi'], 
                       cov_params.loc[cov_params['smoking_new']=='no','xi'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

for i in range(14):
    print('Omegas', i , '\n' )
    stat, p = mannwhitneyu(cov_params.loc[cov_params['smoking_new']=='yes','w_'+str(i)],
                       cov_params.loc[cov_params['smoking_new']=='no','w_'+str(i)])
   
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

    
frame.groupby('HTA_new').median().round(3)
frame.groupby('HTA_new').quantile([0.25, 0.75]).round(2)

print('p-values for HTA')
print('XI', '\n' )
stat, p = mannwhitneyu(cov_params.loc[cov_params['HTA_new']=='yes','xi'], 
                       cov_params.loc[cov_params['HTA_new']=='no','xi'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

for i in range(14):
    print('Omegas', i , '\n' )
    stat, p = mannwhitneyu(cov_params.loc[cov_params['HTA_new']=='yes','w_'+str(i)],
                       cov_params.loc[cov_params['HTA_new']=='no','w_'+str(i)])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')
    
frame.groupby('Domaine').median().round(3)
frame.groupby('Domaine').quantile([0.25, 0.75]).round(2)

print('p-values for mutation')
print('XI', '\n' )
stat, p = mannwhitneyu(cov_params.loc[cov_params['Domaine']=='1-6','xi'], 
                       cov_params.loc[cov_params['Domaine']=='7-34','xi'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
print('\n')

for i in range(14):
    print('Omegas', i , '\n' )
    stat, p = mannwhitneyu(cov_params.loc[cov_params['Domaine']=='1-6','w_'+str(i)],
                       cov_params.loc[cov_params['Domaine']=='7-34','w_'+str(i)])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    print('\n')

#interaction between covariables

data_cont=pd.crosstab(frame['labels'],frame['sexe_t'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['neduc_t'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['smoking_new'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['HTA_new'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['Domaine'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

cov[['FDRV_hcholes']] = 'no'
cov[['FDRV_diab']] = 'no'
cov[['FDRV_other']] = 'no'

cov.loc[cov['hcholes']==1, 'FDRV_hcholes'] = 'yes'
cov.loc[cov['diab']==1, 'FDRV_diab'] = 'yes'
cov.loc[(cov['frcardio']==1) & (cov['hta']==2) & (cov['hcholes']==2) & (cov['diab']==2) & (cov['smoking_new']=='no'),'FDRV_other'] = 'yes'

print(cov[['smoking_new']].value_counts())
print(cov[['HTA_new']].value_counts())
print(cov[['FDRV_hcholes']].value_counts())
print(cov[['FDRV_diab']].value_counts())
print(cov[['FDRV_other']].value_counts())

frame[['FDRV_hcholes']] = cov[['FDRV_hcholes']]
frame[['FDRV_diab']] = cov[['FDRV_diab']]
frame[['FDRV_other']] = cov[['FDRV_other']]

data_cont=pd.crosstab(frame['labels'],frame['FDRV_hcholes'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['FDRV_diab'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

data_cont=pd.crosstab(frame['labels'],frame['FDRV_other'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)

##sexe
print('\n GENDER \n')

data_cont=pd.crosstab(frame['sexe_t'],frame['neduc_t'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
data_cont=pd.crosstab(frame['sexe_t'],frame['smoking_new'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['sexe_t'],frame['HTA_new'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
data_cont=pd.crosstab(frame['sexe_t'],frame['Domaine'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
data_cont=pd.crosstab(frame['sexe_t'],frame['FDRV_hcholes'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['sexe_t'],frame['FDRV_diab'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['sexe_t'],frame['FDRV_other'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

##education
print('\n EDUCATION \n')

data_cont=pd.crosstab(frame['neduc_t'],frame['smoking_new'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['neduc_t'],frame['HTA_new'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
data_cont=pd.crosstab(frame['neduc_t'],frame['Domaine'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
data_cont=pd.crosstab(frame['neduc_t'],frame['FDRV_hcholes'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['neduc_t'],frame['FDRV_diab'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['neduc_t'],frame['FDRV_other'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
##smoking
print('\n SMOKING \n')

data_cont=pd.crosstab(frame['smoking_new'],frame['HTA_new'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
data_cont=pd.crosstab(frame['smoking_new'],frame['Domaine'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
data_cont=pd.crosstab(frame['smoking_new'],frame['FDRV_hcholes'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['smoking_new'],frame['FDRV_diab'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['smoking_new'],frame['FDRV_other'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
##hta
print('\n HTA \n')

data_cont=pd.crosstab(frame['HTA_new'],frame['Domaine'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
data_cont=pd.crosstab(frame['HTA_new'],frame['FDRV_hcholes'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['HTA_new'],frame['FDRV_diab'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['HTA_new'],frame['FDRV_other'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
##Domaine
print('\n DOMAINE \n')

data_cont=pd.crosstab(frame['Domaine'],frame['FDRV_hcholes'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['Domaine'],frame['FDRV_diab'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['Domaine'],frame['FDRV_other'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
##hcholes
print('\n CHOLESTEROL \n')

data_cont=pd.crosstab(frame['FDRV_hcholes'],frame['FDRV_diab'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')

data_cont=pd.crosstab(frame['FDRV_hcholes'],frame['FDRV_other'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
##diab
print('\n DIABETES \n')

data_cont=pd.crosstab(frame['FDRV_diab'],frame['FDRV_other'])
print(data_cont)
stat, p, dof, expected = chi2_contingency(data_cont)
print('p-value :',p)
print('expected \n', expected)
if p<0.05 :
    print('\n HERE !!! \n')
    
frame[['labels_bool']] = 0
frame.loc[frame['labels']=='precoce','labels_bool'] = 1

frame[['sexe_bool']] = 0
frame.loc[frame['sexe_t']=='men','sexe_bool'] = 1

frame[['neduc_bool']] = 0
frame.loc[frame['neduc_t']=='<13 years','neduc_bool'] = 1

frame[['smoking_bool']] = 0
frame.loc[frame['smoking_new']=='yes','smoking_bool'] = 1

frame[['HTA_bool']] = 0
frame.loc[frame['HTA_new']=='HTA','HTA_bool'] = 1

frame[['hcholes_bool']] = 0
frame.loc[frame['FDRV_hcholes']=='yes','hcholes_bool'] = 1

frame[['diab_bool']] = 0
frame.loc[frame['FDRV_diab']=='yes','diab_bool'] = 1

frame[['Domaine_bool']] = 0
frame.loc[frame['Domaine']=='1-6','Domaine_bool'] = 1

frame[['Risk_bool']] = 0
frame.loc[frame['Risk_new']=='1-6,8,11,26','Risk_bool'] = 1

contigency = pd.crosstab(frame['labels'], frame['labels_bool'])
print(contigency)

import statsmodels.formula.api as smf
riskmodel = smf.logit(formula = 'labels_bool ~ sexe_t', data = frame).fit()
print(riskmodel.summary(), '\n')
riskmodel = smf.logit(formula = 'labels_bool ~ sexe_bool', data = frame).fit()
print(riskmodel.summary(), '\n')
print('\n')
print('\n')

riskmodel = smf.logit(formula = 'labels_bool ~ neduc_t', data = frame).fit()
print(riskmodel.summary(), '\n')
riskmodel = smf.logit(formula = 'labels_bool ~ neduc_bool', data = frame).fit()
print(riskmodel.summary(), '\n')
print('\n')
print('\n')

riskmodel = smf.logit(formula = 'labels_bool ~ smoking_new', data = frame).fit()
print(riskmodel.summary(), '\n')
riskmodel = smf.logit(formula = 'labels_bool ~ smoking_bool', data = frame).fit()
print(riskmodel.summary(), '\n')
print('\n')
print('\n')

riskmodel = smf.logit(formula = 'labels_bool ~ HTA_new', data = frame).fit()
print(riskmodel.summary(), '\n')
riskmodel = smf.logit(formula = 'labels_bool ~ HTA_bool', data = frame).fit()
print(riskmodel.summary(), '\n')
print('\n')
print('\n')

riskmodel = smf.logit(formula = 'labels_bool ~ FDRV_hcholes', data = frame).fit()
print(riskmodel.summary(), '\n')
riskmodel = smf.logit(formula = 'labels_bool ~ hcholes_bool', data = frame).fit()
print(riskmodel.summary(), '\n')
print('\n')
print('\n')

riskmodel = smf.logit(formula = 'labels_bool ~ FDRV_diab', data = frame).fit()
print(riskmodel.summary(), '\n')
riskmodel = smf.logit(formula = 'labels_bool ~ diab_bool', data = frame).fit()
print(riskmodel.summary(), '\n')
print('\n')
print('\n')

riskmodel = smf.logit(formula = 'labels_bool ~ Domaine', data = frame).fit()
print(riskmodel.summary(), '\n')
riskmodel = smf.logit(formula = 'labels_bool ~ Domaine_bool', data = frame).fit()
print(riskmodel.summary(), '\n')
print('\n')
print('\n')

riskmodel = smf.logit(formula = 'labels_bool ~ Risk_new', data = frame).fit()
print(riskmodel.summary(), '\n')
riskmodel = smf.logit(formula = 'labels_bool ~ Risk_bool', data = frame).fit()
print(riskmodel.summary(), '\n')
print('\n')
print('\n')

#significant interactions 

riskmodel = smf.logit(formula = 'labels_bool ~ sexe_bool*Risk_bool', data = frame).fit()
print(riskmodel.summary())

riskmodel = smf.logit(formula = 'labels_bool ~ neduc_bool*hcholes_bool', data = frame).fit()
print(riskmodel.summary())

riskmodel = smf.logit(formula = 'labels_bool ~ neduc_bool*Domaine_bool', data = frame).fit()
print(riskmodel.summary())

riskmodel = smf.logit(formula = 'labels_bool ~ HTA_bool*hcholes_bool', data = frame).fit()
print(riskmodel.summary())

riskmodel = smf.logit(formula = 'labels_bool ~ HTA_bool*diab_bool', data = frame).fit()
print(riskmodel.summary())

riskmodel = smf.logit(formula = 'labels_bool ~ HTA_bool*Domaine_bool', data = frame).fit()
print(riskmodel.summary())

riskmodel = smf.logit(formula = 'labels_bool ~ HTA_bool*Risk_bool', data = frame).fit()
print(riskmodel.summary())

riskmodel = smf.logit(formula = 'labels_bool ~ hcholes_bool*diab_bool', data = frame).fit()
print(riskmodel.summary())


#Backward selection
riskmodel = smf.logit(formula = 'labels_bool ~ sexe_bool + neduc_bool + HTA_bool + hcholes_bool + diab_bool + Domaine_bool', 
                      data = frame).fit()
print(riskmodel.summary())
print('AIC : ', riskmodel.aic)
print('BIC : ', riskmodel.bic, '\n')

riskmodel = smf.logit(formula = 'labels_bool ~ sexe_bool + neduc_bool + HTA_bool + diab_bool + Domaine_bool', 
                      data = frame).fit()
print(riskmodel.summary())
print('AIC : ', riskmodel.aic)
print('BIC : ', riskmodel.bic, '\n')

riskmodel = smf.logit(formula = 'labels_bool ~ sexe_bool + neduc_bool + HTA_bool + Domaine_bool ', 
                      data = frame).fit()
print(riskmodel.summary(), '\n')


    

    
