
# coding: utf-8

# In[6]:

# Import
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import pylab as P
import Orange

from sklearn import decomposition
from sklearn import linear_model
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import preprocessing
import scipy
import math
get_ipython().magic('matplotlib inline')


#### Functions

# In[7]:

def stringTOfloat(matrix):
    """Cast string values of a matrix into float values"""
    matrix1 = []
    for pos in range(len(matrix)):
        i = matrix[pos]
        #i[i == ''] = 0.0
        i = i.astype(np.float)
        matrix1.append(i)
    return np.array( matrix1 )


# In[8]:

def NANto0(matrix):
    # Daj vse NAN na 0 -> ni ok, ampak za zacetek bo ok
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if np.isnan(matrix[i,j]):
                matrix[i,j] = 0
    return matrix


# In[9]:

def getData(name):
    file1 = open(name, "rU" )
    Y = []
    for aRow in file1:
        Y.append(aRow.split('\t'))
    file1.close()
    return Y


# In[10]:

# Parse dilution and get denominator
def get_dilution_denominator( string ):
    i = 0
    # Get rid of numerator
    while ( string[i] != '/' and i < len(string) ):
        i += 1
        
    # Get rid of '/'
    while ( string[i] != '1' and i < len(string) ):
        i += 1
        
    # Count zeros in denominator
    n_zeros = 0
    for j in range(i, len(string)):
        if string[j] == '0':
            n_zeros += 1
            
    return 10**n_zeros 


# In[11]:

def getX():
    X = getData("../data/molecular_descriptors_data.txt")
    
    # remove header
    X = np.array(X[1:])
    X = stringTOfloat(X)
    # set Nan to 0
    X = NANto0(X)
    return X


# In[12]:

def getY():
    Y = getData("../data/TrainSet-hw2.txt")
    
    # header
    head = Y[0]
    
    # remove header
    Y = np.array(Y[1:])
    
    # get CID, get dil
    cids = np.matrix(stringTOfloat(Y[:,0]))
    dils = np.matrix([get_dilution_denominator(dil) for dil in Y[:,4]])
   
    # from 
    Y_rest = np.array(Y[:, 6:]) 
    Y_rest = stringTOfloat(Y_rest)
    
    Y_cid_dil_rest = np.hstack((cids.T,dils.T, Y_rest))
    Y_cid_dil_rest = np.array(Y_cid_dil_rest)
 
    
    return Y_cid_dil_rest, head
    


# In[13]:

def avg_median(Y_cid_dil_rest):
    
    cids = Y_cid_dil_rest[:, 0]
    dils = Y_cid_dil_rest[:, 1]
    # average
    # get cids
    cids_unique = np.unique(np.array(cids))
    
    # for each cid store avg value for all attributes
    cid_avg = []
    for cid in cids_unique:
        cid_samples = Y_cid_dil_rest[ Y_cid_dil_rest[ :, 0 ] == cid ]
        dil_low = cid_samples[0][1] 
        dil_high = cid_samples[1][1] 
        cid_samples_rest = np.matrix(cid_samples[:, 2:])  # remove cid and dil  
        cid_samples_rest = np.array(cid_samples_rest[0::2].T)
        # average
        avg_low = [np.mean(column[~np.isnan(column)]) for column in cid_samples_rest]
        avg_high = [np.mean(column[~np.isnan(column)]) for column in cid_samples_rest]
        # concatenate cid with average values
        cid_avg.append( np.hstack(([cid, dil_low], avg_low)) )
        cid_avg.append( np.hstack(([cid, dil_high], avg_high)) )
        
    cid_avg = np.array( cid_avg )
    return cid_avg


# In[14]:

def getPredict():
    P = getData("../data/predict.txt")
    
    P = np.array(P)
    
    # get CID
    cids = stringTOfloat(P[:,0])
    dils = [get_dilution_denominator(dil) for dil in P[:,1]]
    
    P = []
    for i in range( len(cids) ):
        P.append((cids[i], dils[i]))
    return np.array(P)


# In[15]:

# Pravzaprav, ne uporabljam
def normalize_correct(matrix):
    size = matrix.shape
    for i in range( size[1] ):
        sum = 0
        n = 0
        # Get sum of column
        for j in range( size[0] ):
            if np.isnan(matrix[j,i]):
                pass
            else:
                sum += abs(matrix[j,i])
                n += 1
          
        # Correct each value in column
        for j in range( size[0] ):
            # NaN -> avg value
            if np.isnan(matrix[j,i]):
                matrix[j,i] = sum / n
                
            # sum = 0 -> 0
            elif sum == 0:
                matrix[j,i] = 0
                
            # else value / sum
            else:
                matrix[j,i] = matrix[j,i] / sum
                
    return matrix


# In[16]:

def select_dilution(Y, dilution = None):
    """Compute average attributes for CID. Dilution influences, which samples to take."""
    
    if dilution != None:
        # if dilution not None, take only samples with that dilution
        return Y[Y[:,1] == dilution]
    else:
        # if None, then take all Y
        return Y


# In[17]:

def select_low_high_dilution(Y, high=True):
    """Compute average attributes for CID. Based on high or low"""
    # average for each cid and dilution
    
    if high:
        return Y[1::2]  # take high dilutions
    else:
        return Y[0::2]  # take low dilutions


# In[18]:

# Ne uporabljam
def best_model(X_train, Y_train):
    best_m = None
    best_val = 100
    best_alpha = 0
    best_ratio = 0
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 5, 10]:
        for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            elastic = linear_model.ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=500)
            rez = my_cross_validation(X_train, Y_train, elastic)
            if rez < best_val:
                best_m = elastic
                best_val = rez
                best_alpha = alpha
                best_ratio = ratio
    # return best model
    return best_m


# In[19]:

def write_to_file(cid_preds, head, name):
    head = head[6:] # remove labels for attributes that are not needed
    head[-1] = (head[-1])[:-1]  # remove \n at the end of head labels
    f = open('../rezults/'+name+'.txt','w')
    for cid, pred in cid_preds:
        for i in range( len(pred) ):
            f.write(str(int(cid)))   # change float to int then to string
            f.write('\t')
            f.write(head[i])
            f.write('\t')
            f.write(str(round(float(pred[i]), 6)))
            f.write('\n')
    f.close()
    
def write_to_file2(cids, preds, head, name):
    head = head[6:] # remove labels for attributes that are not needed
    head[-1] = (head[-1])[:-1]  # remove \n at the end of head labels
    f = open('../rezults/'+name+'.txt','w')
    for i in range(len(cids)):
        cid = cids[i]
        pred = preds[i]
        for j in range( len(pred) ):
            f.write(str(int(cid)))   # change float to int then to string
            f.write('\t')
            f.write(head[j])
            f.write('\t')
            f.write(str(round(float(pred[j]), 6)))
            f.write('\n')
    f.close()


# In[20]:

NORM_STD = [ 0.18, 0.16, 0.06 ] #an average of normalizatin_costs outputs)
#means were 0 (as expected for Pearson correlation)

def pearson(x,y):
    x,y = np.array(x), np.array(y)
    anynan = np.logical_or(np.isnan(x), np.isnan(y))
    r = scipy.stats.pearsonr(x[~anynan],y[~anynan])[0]
    return 0. if math.isnan(r) else r

def final_score(rs):
    zs = rs/NORM_STD
    return np.mean(zs)

def evaluate_r(prediction, real):
    userscores = prediction
    realscores = real
    rint = pearson(userscores[:,0], realscores[:,0])
    rval = pearson(userscores[:,1], realscores[:,1])
    rdecall = [ pearson(userscores[:,i], realscores[:,i]) for i in range(2,21) ]
    rdec = np.mean(rdecall)
    return np.array([rint, rval, rdec])


# In[21]:

def get_pca(X_sel_tr, components):
    pca = decomposition.PCA(n_components=components)
    pca.fit(X_sel_tr)
    X_sel_tr = pca.transform(X_sel_tr)
    return X_sel_tr, pca


# In[22]:

def standardize(X_sel_tr):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_sel_tr) #shranimo transformacijo
    X_sel_tr = scaler.transform(X_sel_tr) #transformiramo X (standardiziramo)
    return X_sel_tr, scaler


# In[30]:

def compute_models(X_sel_tr, Y_train, dilution):
    """Create scalar, pca and 21 models for a given dilution"""

    #model = linear_model.RidgeCV()
    #model = linear_model.ElasticNetCV()
    #model = linear_model.ElasticNet()
    #model = ensemble.RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=5, random_state=0, n_jobs=1)
    model = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')    

    models_list = []
    for i in range(0,(Y_train.shape)[1]):
        #print('start learn', i)
        # learn
        lrn = model.fit(X_sel_tr, Y_train[:, i])
        models_list += [lrn]
    
    return models_list


# In[23]:

def learn_models(X_dict_by_cid, Y, unique_dils):
    dils_models = dict()
    for dil in unique_dils:

        # Get y for intensity and pleasantness
        Y_int_ple = Y[ Y[:,1] == 1000]
        # Get for rest
        Y_rest = select_low_high_dilution(Y, high=True)


        #Y_cid_dil_rest_avg = select_dilution(Y, dil)
        #Y_cid_avg = select_low_high_dilution(Y_cid_dil_rest, high=True)

        #y_train = Y_cid_dil_rest_avg[:, 2:]  # remove cid and dilution, take rest

        # Pripravi X glede na Y
        X_int_ple = np.array( [X_dict_by_cid[cid] for cid in Y_int_ple[:, 0] ] )
        X_rest = np.array( [X_dict_by_cid[cid] for cid in Y_rest[:, 0] ] )

        Y_int_ple = Y_int_ple[:, 2:4]  # remove rest
        Y_rest = Y_rest[:, 4:]   # remove rest

        # Standardize
        scalar_int_ple = None
        scalar_rest = None
        #X_int_ple, scalar_int_ple = standardize(X_int_ple)
        #X_rest, scalar_rest = standardize(X_rest)

        # Pca
        X_int_ple, pca_int_ple = get_pca(X_int_ple, 20)
        X_rest, pca_rest = get_pca(X_rest, 20)

        # Get models
        models_list_int_ple = compute_models(X_int_ple, Y_int_ple, dil)
        models_list_rest = compute_models(X_rest, Y_rest, dil)

        models_list = models_list_int_ple + models_list_rest

        dils_models[int(dil)] = (scalar_int_ple, scalar_rest, pca_int_ple, pca_rest, models_list)
        
    return dils_models


# In[98]:

# Adjusted fast cross validation
def CV_fast(X_dict_by_cid, Y_cid_dil_rest, dils_models):
    repetitions = 3
    S = 0
    kf = cross_validation.KFold(len(Y_cid_dil_rest), n_folds=repetitions)
   
    for train_index, test_index in kf:
        #X_train, X_test = X[train_index], X[test_index]
        Y_cid_dil_rest_train, Y_cid_dil_rest_test = Y_cid_dil_rest[train_index], Y_cid_dil_rest[test_index]

        # prepare examples to predict
        toPredict = [(Y_cid_dil_rest_test[i, 0], Y_cid_dil_rest_test[i, 1]) for i in range((Y_cid_dil_rest_test.shape)[0])]
        store_predictions = []
        for pr in toPredict:
            cid = pr[0]
            dil = pr[1]

            X_to_predict = X_dict_by_cid[cid]        
            scalar_int_ple, scalar_rest, pca_int_ple, pca_rest, models_list = dils_models[int(dil)]

            # Standardize
            if not scalar_int_ple == None:
                X_to_predict_int_ple = scalar_int_ple.transform(X_to_predict)
                X_to_predict_rest = scalar_rest.transform(X_to_predict)
            
            # Pca
            X_to_predict_int_ple = pca_int_ple.transform(X_to_predict)
            X_to_predict_rest = pca_rest.transform(X_to_predict)

            rezult = []
            for i in range(len(models_list)):
                model = models_list[i]
                if i < 2:
                    X_to_predict = X_to_predict_int_ple
                else:
                    X_to_predict = X_to_predict_rest

                p = model.predict(X_to_predict)
                rezult += [p[0]]
            store_predictions += [np.array(rezult)]
        store_predictions = np.array(store_predictions)
        
        # evaluate
        score = evaluate_r(store_predictions, Y_cid_dil_rest_test)
        score = final_score(score)
        print("Vmesni .................", score)
        S += score

    print("FINAL .................", S/repetitions)


# In[99]:

# General cross validation
def CV(X, Y, model):
    repetitions = 5
    S = 0
    kf = cross_validation.KFold(len(Y), n_folds=repetitions)
   
    for train_index, test_index in kf:
        print('Start new fold')
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        X_train, std = standardize(X_train)
        X_test = std.transform(X_test)
        
        X_train, pca = get_pca(X_train, 20)
        X_test = pca.transform(X_test)
        
        store_predictions = []
        for i in range((Y_train.shape)[1]):
            model.fit(X_train, Y_train[:, i])
            P = model.predict(X_test)
            store_predictions.append(P)

        store_predictions = np.array(store_predictions)
        store_predictions = np.matrix(store_predictions).T
        store_predictions = np.array(store_predictions)
        # evaluate
        score = evaluate_r(store_predictions, Y_test)
        score = final_score(score)
        print("Vmesni .................", score)
        S += score

    print("FINAL .................", S/repetitions)


# In[26]:

def prediction(X_train, Y_train, X_test, model):
    X_train, std = standardize(X_train)
    X_test = std.transform(X_test)

    X_train, pca = get_pca(X_train, 20)
    X_test = pca.transform(X_test)

    store_predictions = []
    for i in range((Y_train.shape)[1]):
        model.fit(X_train, Y_train[:, i])
        P = model.predict(X_test)
        store_predictions.append(P)

    store_predictions = np.array(store_predictions)
    store_predictions = np.matrix(store_predictions).T
    store_predictions = np.array(store_predictions)
    return store_predictions


# In[100]:

def dilution_dependant_prediction(X_train, Y_train, X_test, dils, model):
    # TODO - repair
    X_train, std = standardize(X_train)
    X_test = std.transform(X_test)

    X_train, pca = get_pca(X_train, 20)
    X_test = pca.transform(X_test)

    store_predictions = []
    for i in range((Y_train.shape)[1]):
        model.fit(X_train, Y_train[:, i])
        P = model.predict(X_test)
        store_predictions.append(P)

    store_predictions = np.array(store_predictions)
    store_predictions = np.matrix(store_predictions).T
    store_predictions = np.array(store_predictions)
    return store_predictions


## RUN

#### Load data

# In[48]:

# RUN 1
# cid, rest
X_cid_rest = getX()
# cid, dilution, rest
Y_cid_dil_rest, head = getY()
# predict info
toPredict = getPredict()
# Create access to chemical informations via CID
X_dict_by_cid = dict([(i[0], i[1:]) for i in X_cid_rest])


# In[49]:

# RUN 2
# Average
Y_cid_dil_rest_avg = avg_median(Y_cid_dil_rest)
# Unique dillutions
unique_dils = np.unique(Y_cid_dil_rest[:, 1])


#### Optional features

                # RUN OPTIONAL
# Compute models in advance based on the dilutions
dils_models = learn_models(X_dict_by_cid, Y_cid_dil_rest_avg, unique_dils)
                
                # RUN OPTIONAL
# Make a prediction with models, that are computed in advance
store_predictions = []
for pr in toPredict:
    cid = pr[0]
    dil = pr[1]

    X_to_predict = X_dict_by_cid[cid]        
    scalar_int_ple, scalar_rest, pca_int_ple, pca_rest, models_list = dils_models[int(dil)]

    # Standardize
    if not scalar_int_ple == None:
        X_to_predict_int_ple = scalar_int_ple.transform(X_to_predict)
        X_to_predict_rest = scalar_rest.transform(X_to_predict)

    # Pca
    X_to_predict_int_ple = pca_int_ple.transform(X_to_predict)
    X_to_predict_rest = pca_rest.transform(X_to_predict)

    rezult = []
    for i in range(len(models_list)):
        model = models_list[i]
        if i < 2:
            X_to_predict = X_to_predict_int_ple
        else:
            X_to_predict = X_to_predict_rest

        p = model.predict(X_to_predict)
        rezult += [p[0]]
    store_predictions += [np.array(rezult)]
store_predictions = np.array(store_predictions)
                
                # RUN OPTIONAL
write_to_file2(toPredict[:, 0], store_predictions, head, "gbr_1")
                
#### Cross validation

# In[112]:

# Start cross validation

#model = linear_model.Ridge()
model = linear_model.RidgeCV()
#model = linear_model.ElasticNetCV()
#model = linear_model.ElasticNet()
#model = ensemble.RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=5, random_state=0, n_jobs=1)
#model = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
X = np.array([X_dict_by_cid[cid] for cid in Y_cid_dil_rest_avg[:,0]])
Y = Y_cid_dil_rest_avg[:, 2:]
CV(X, Y, model)


#### Predict real data

# In[113]:

# Predict real data
X_pred = [X_dict_by_cid[cid] for cid, dil in toPredict]
rezult = prediction(X, Y, X_pred, model)


# In[114]:

# Write into file
write_to_file2(toPredict[:, 0], rezult, head, "ridge_2")


#### Null distribution of out data

# In[33]:

from scipy.stats.stats import pearsonr


# In[93]:

def remove_nan(l):
    pc = []
    for i in l:
        if not np.isnan(i):
            pc += [i]
    return pc


# In[43]:

X = np.array([X_dict_by_cid[cid] for cid in Y_cid_dil_rest_avg[:,0]])
Y = Y_cid_dil_rest_avg[:, 2]  # take one Y


# In[83]:

pc = [np.abs(pearsonr(X[:, i], Y)[0]) for i in range(X.shape[1])]
pc = remove_nan(pc)


# In[84]:

plt.hist(pc, 50, color="yellow", normed=True);


# In[94]:

y = np.copy(Y)
np.random.seed(42)
pc_null = []
for i in range(5):
    np.random.shuffle(y)
    pc_null.extend([np.abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])])
    
pc_null = remove_nan(pc_null)


# In[97]:

plt.hist(pc_null, 50, color="yellow", normed=True);
p = 0.01
threshold = np.sort(pc_null)[(1-p)*len(pc_null)]
plt.vlines(threshold, plt.ylim()[0], plt.ylim()[1], lw=3, color="k");


# In[ ]:



