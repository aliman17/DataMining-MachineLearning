__author__ = 'Rok'

import numpy as np
import math
import scipy
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold

#*******************#
# PRIPRAVA PODATKOV #
#*******************#

#Najprej odpremo in preberemo .txt datoteke ter podatke stlačimo v seznam (seznamov)
file1 = open( "data/TrainSet-hw2.txt", "rU" )
Ys = []
for aRow in file1:
    Ys.append(aRow.split('\t'))
file1.close()

file1 = open( "data/molecular_descriptors_data.txt", "rU" )
Xs = []
for aRow in file1:
    Xs.append(aRow.split('\t'))
file1.close()

file1 = open("data/predict.txt", "rU")
predict = []
for aRow in file1:
    predict.append(aRow.split('\t'))
file1.close()

#Seznam seznamov pretvorimo v np.array
Ys = np.array(Ys)
Xs = np.array(Xs)
predict = np.array(predict)

#Napišem funkcijo, ki iz matrike odstrani presledke, nove vrstice in dvojne narekovaje. Vzame np.array.
def clean(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i,j] = matrix[i,j].strip('\n').strip('""').strip(' ')
    return matrix

#Počistimo podatke
Xs = clean(Xs)
Ys = clean(Ys)
predict = clean(predict)

#Funkcija, ki vse string-e v matriki pretvori v float-e. Dela po vrsticah.
def stringTOfloat(matrix):
    matrix1 = []
    for pos in range(len(matrix)):
        i = matrix[pos]
        #i[i == ''] = 0.0
        i = i.astype(np.float)
        matrix1.append(i)
    return np.array( matrix1 )

#Funkcija, ki ustrezno pretvori vrednosti dilution-a v float-e.
#Sprejema vektorje tipe oblike [1/1000, 1/100, 1/1000000, ...]
#Dilution nastopa v 4 različnih vrednostih.
def dilutionTOfloat(matrix):
    novVektor = np.zeros(len(matrix), dtype=float)
    for x in range(len(matrix)):
        i = matrix[x]
        if i == '1/1,000':
            novVektor[x] = (1/1000)
        elif i == '1/10':
            novVektor[x] = (1/10)
        elif i == '1/10,000,000':
            novVektor[x] = (1/10000000)
        elif i == '1/100,000':
            novVektor[x] = (1/100000)
        else:
            print("Unknown dilution")
    return np.reshape(novVektor, (len(matrix),1))

#Sedaj imamo podatke v tabelah s tipom string.
#Glede na nalogo bomo tabele ustrezno spreminjali.

#Funkcija za računanje povprečne vrednosti v stolpcu, ki ignorira NaN-e.
def povprecje(vektor):
    indeksi = np.logical_not(np.isnan(vektor))
    avg = np.mean(vektor[indeksi])
    return avg

def mediana(vektor):
    indeksi = np.logical_not(np.isnan(vektor))
    avg = np.median(vektor[indeksi])
    return avg


#Napišem funkcije za krosvalidacijo. To bodo 3 različne funkcije. Vsaka vzame X in Y
#(takšna kot ju vračajo funkcije pri posamezni nalogi)
#ter k in naredi krosvalidacijo (razdeli množico na k komponent),
#kjer za scoring uporabi funkcijo iz učilnice. Skupen score je potem povprečje vseh scorov.

#************************#
# PRIPRAVA X IN Y MATRIK #
#************************#

#Vzamemo vse stolpce razen indeks 1. Še prej pa odrežemo header.
Y = Ys[1:]
Y = np.delete(Y, 1, 1)

# NALOGA 1
#---------

#Vzamemo samo vrstice, ki imajo v stolpcu "dilution" vrednost '1/1000'.
Y1 = Y[Y[:,3] == '1/1,000']
#Shranimo še IDje, ki so bili podvojeni - "replicated"
#replicated = np.unique(Y1[Y1[:,1] == 'replicate'][:,0]) #15 jih je. To je kul. :)

IDsStr = np.unique(Y1[:,0])

#Pretvorimo matriko Y1 v tip float. Dilution stolpec lahko izpustimo, ker vemo da je povsod '1/1000'.
#Prav tako ne rabimo več stolpca replicated. Pretvorimo tudi vektor replicated.
Y1 = stringTOfloat(Y1[:, [0,5]])
#replicated = stringTOfloat(replicated)

#Shranimo IDje, uporabili bomo samo te IDje pri vseh modelih.
IDs = np.unique(Y1[:,0])

#Povprečimo opažanja subjektov, tako da imamo za vsak Compound Identifier eno vrstico.
intensitys = np.zeros(len(IDs))
for i in range(len(IDs)):
    intensitys[i] = np.mean(Y1[Y1[:,0] == IDs[i]][:,1]) #Tukaj bomo poskusili še z mediano in pogledali, če bo kaj boljše.

Y1 = intensitys
    
#PRIPRAVA X MATRIKE - ta bo za vse tri naloge ista, ker uporabljamo iste CIDje pri vseh treh nalogah
#Odstranimo header pri X-u in X pretvorimo v float.
X = Xs[1:]
X = stringTOfloat(X)

#Gremo čez matriko in tam, kjer srečamo NaN ga nadomestimo s povprečjem v stolpcu, kjer se nahaja.

#Nadomestimo NaN-e s povprecji stolpcev
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if np.isnan(X[i,j]):
            X[i,j] = povprecje(X[:, j])
            #X[i,j] = 0
            
Xtest = X #shranimo za potem, da bomo iz tu povlekli podatke za testno množico.

#Iz X sedaj želimo imeti samo tiste vrstice katerih prvi(indeks 0) stolpec ima CID, ki ga najdemo tudi v IDs.
indeksi = np.zeros(X.shape[0], dtype=bool)
for i in range(X.shape[0]):
    indeksi[i] = X[i,0] in IDs
X = X[indeksi]

#Odstraniti je potrebno še prvi stolpec (CID)
X = X[:,1:]

#Na tem mestu še iz X vržemo vn konstantne stolpce
selector = VarianceThreshold()
selector.fit(X) #shranimo transformacijo
X = selector.transform(X) #transformiramo X (odstranimo konstantne stolpce)

# NALOGA 2
#---------

#PRIPRAVA Y MATRIKE

#Obdržimo samo tiste IDje, kot pri prvi nalogi.
indeksi = np.zeros(Y.shape[0], dtype=bool)
for i in range(len(Y)):
    indeksi[i] = Y[i,0] in IDsStr
Y = Y[indeksi]


###opcija a)
###Vzamimo tiste z intensity high
##Y2 = Y[Y[:,2] == 'high']

###opcija b)
###Vzamimo tiste z intensity low
##Y2 = Y[Y[:,2] == 'low']

#opcija c)
#Vzamemo kar vse, ignoriramo intensity
Y2 = Y

#Vzamemo samo stolpca CID in Valence
Y2 = Y2[:,[0,6]]
Y2 = stringTOfloat(Y2)

#Povprečimo opažanja subjektov, tako da imamo za vsak Compound Identifier eno vrstico.
valences = np.zeros(len(IDs))
for i in range(len(IDs)):
    valences[i] = mediana(Y2[Y2[:,0] == IDs[i]][:,1]) #Tukaj ne vem čisto kaj je bolš vzet. Median ali Mean

Y2 = valences
# NALOGA 3
#---------

#PRIPRAVA Y MATRIKE

#Vzamemo samo vrstice, ki imajo v stolpcu "intensity" vrednost 'high'.
Y3 = Y[Y[:,2] == 'high']

Y3 = np.hstack((Y3[:,[0,6]], Y3[:, 7:]))
Y3 = np.delete(Y3, 1, 1)
Y3 = stringTOfloat(Y3)

#Povprečimo opažanja subjektov, tako, da imamo za vsak Compound Identifier eno vrstico.
intensitys_odr = np.zeros((len(IDs), 19), dtype=float)
for j in range(19):
    for i in range(len(IDs)):
        enaGrupa = Y3[Y3[:,0] == IDs[i]][:,j+1]
        intensitys_odr[i, j] = povprecje(enaGrupa) #tuki se zdi bols vzet povprecje?

Y3 = intensitys_odr

Y = np.hstack((np.reshape(Y1, (len(Y1), 1)), np.reshape(Y2, (len(Y2), 1)), Y3))

#Da bom lahko napisal preprosto prečno preverjanje, rabim pr vseh 3 modelih
#iste CID-je v učni množici.

#*******************#
# MODELS/TECHNIQUES #
#*******************#                       

class rf:
    """Class za random forest z n drevesi"""
    def __init__(self, n):
        self.model = RandomForestRegressor(n_estimators=n, min_samples_split=1, n_jobs=-1)
    
    def fit(self, Xtrain, Ytrain):
        """Se nauči na train"""
        self.model.fit(Xtrain, Ytrain)

    def predict(self, Xtest):
        """Napove vrednosti za X test"""
        return self.model.predict(Xtest)

class rf_pca:
    """Class za random forest z n drevesi in s pca na n_pca glavnih komponent"""
    def __init__(self, n, n_pca):
        self.pca = PCA(n_components=n_pca)
        self.model = RandomForestRegressor(n_estimators=n, min_samples_split=1, n_jobs=-1)
    
    def fit(self, Xtrain, Ytrain):
        self.pca.fit(Xtrain)
        self.pca.transform(Xtrain)
        self.model.fit(Xtrain, Ytrain)

    def predict(self, Xtest):
        self.pca.transform(Xtest)
        return self.model.predict(Xtest)

class ridge(rf):
    """Class za ridge z regularizacijskim parametrom alpha"""
    def __init__(self, alpha):
        self.model = Ridge(alpha=alpha, normalize=True)

class ridge_pca(rf_pca):
    """Class za ridge z regularizacijskim parametrom alpha s pca na n_pca glavnih komponent"""
    def __init__(self, alpha, n_pca):
        self.pca = PCA(n_components=n_pca)
        self.model = Ridge(alpha=alpha, normalize=True)

class lasso:
    """Class za lasso z regularizacijskim parametrom alpha"""
    def __init__(self, alpha):
        self.model = MultiTaskLasso(alpha=alpha)
        self.scaler = StandardScaler()
    
    def fit(self, Xtrain, Ytrain):
        """Neredi standardizacijo in napove"""
        self.scaler.fit(Xtrain)
        self.scaler.transform(Xtrain)
        self.model.fit(Xtrain, Ytrain)

    def predict(self, Xtest):
        self.scaler.transform(Xtest)
        return self.model.predict(Xtest)

class lasso_pca:
    """Class za lasso z regularizacijskim parametrom alpha s pca na n_pca glavnih komponent"""
    def __init__(self, alpha, n_pca):
        self.model = MultiTaskLasso(alpha=alpha)
        self.pca = PCA(n_components=n_pca)
        self.scaler = StandardScaler()
        
    def fit(self, Xtrain, Ytrain):
        """Neredi pca, standardizacijo in se nauči"""
        self.pca.fit(Xtrain)
        self.pca.transform(Xtrain)
        self.scaler.fit(Xtrain)
        self.scaler.transform(Xtrain)
        self.model.fit(Xtrain, Ytrain)
    
    def predict(self, Xtest):
        """Neredi pca, standardizacijo in napove"""
        self.pca.transform(Xtest)
        self.scaler.transform(Xtest)
        return self.model.predict(Xtest)

class el_net(lasso):
    """Class za elastic net z regularizacijskim parametrom alpha in razmerjem regularizacij ration"""
    def __init__(self, alpha, ratio):
        self.model = MultiTaskElasticNet(alpha=alpha, l1_ratio=ratio)
        self.scaler = StandardScaler()

class el_net_pca(lasso_pca):
    """Class za elastic net z regularizacijskim parametrom alpha in razmerjem regularizacij ratio"""
    """ter s pca na n_pca glavnih komponent"""
    def __init__(self, alpha, ratio, n_pca):
        self.model = MultiTaskElasticNet(alpha=alpha, l1_ratio=ratio)
        self.pca = PCA(n_components=n_pca)
        self.scaler = StandardScaler()

#****************************#
# PRIREJENA SCORING FUNKCIJA #
#****************************#
def pearson(x,y):
    x,y = np.array(x), np.array(y)
    anynan = np.logical_or(np.isnan(x), np.isnan(y))
    r = scipy.stats.pearsonr(x[~anynan],y[~anynan])[0]
    return 0. if math.isnan(r) else r

def evaluate_r(prediction, real):
    userscores = prediction
    realscores = real
    rint = pearson(userscores[:,0], realscores[:,0])
    rval = pearson(userscores[:,1], realscores[:,1])
    rdecall = [ pearson(userscores[:,i], realscores[:,i]) for i in range(2,21) ]
    rdec = np.mean(rdecall)
    return np.array([rint, rval, rdec])

#permutacijski test
def normalization_consts(real):
    """ Obtain the gold-standard deviation for our test and
    leaderboard data sets. """
    permres = []
    permuted = real
    for i in range(10000):
        permuted = np.random.permutation(permuted)
        permres.append(evaluate_r(real, permuted))
    permres = np.array(permres)
    return np.mean(permres, axis=0), np.std(permres, axis=0)

def final_score(rs, NORM_STD):
    zs = rs/NORM_STD
    return np.mean(zs)

#********************#
# PREČNO PREVERJANJE #
#********************#

#Vsako tehniko bom testiral tako, da bom s for zanko šel čez različne vrednosti parametrov,
#ki lahko vplivajo na uspešnost metode. To bom nekajkrat ponovil, pri vsaki iteraciji pa bom
#glede na prejšnjo ustrezno zgostil in zožal prostor parametrov.


#splošen CV, ki sprejme podatke, št foldov in model (npr. rf_pca(10,25))
#-----------------------------------------------------------------------
def cv(X, Y, fold, model):
    data = np.hstack((X, Y))
    np.random.shuffle(data)
    X = data[:, :-21]
    Y = data[:, -21:]
    foldsX = np.array(np.array_split(X, fold))
    foldsY = np.array(np.array_split(Y, fold))
    scores = np.zeros(fold)
    NORM_STD = normalization_consts(foldsY[0])[1]
    for i in range(fold):
        indeksi = np.arange(0,fold)
        indeksi = indeksi != i
        Xucna = np.vstack(foldsX[indeksi])
        Yucna = np.vstack(foldsY[indeksi])
        Xtestna = foldsX[i]
        Ytestna = foldsY[i]
        model.fit(Xucna, Yucna)
        napoved = model.predict(Xtestna)
        score = final_score(evaluate_r(napoved, Ytestna), NORM_STD)
        scores[i] = score
    return(np.mean(scores))

#*************#
# POVPREČENJE #
#*************#
class povprecenje:
    def __init__(self, learners):
        self.learners = learners

    def fit(self, Xtrain, Ytrain):
        for i in range(len(self.learners)):
            learner = learners[i]
            learner.fit(Xtrain, Ytrain)

    def predict(self, Xtest):
        ocene = np.zeros((len(Xtest), 21, len(self.learners)))
        for i in range(len(self.learners)):
            learner = learners[i]
            ocene[:,:,i] = learner.predict(Xtest)
        return np.mean(ocene, axis=2)

#**********#
# STACKING #
#**********#
class stacking:
    """Class za stacking"""
    """learners je seznam learnerjev npr. [ridge_pca(0.5, 25), rf(50)]"""
    def __init__(self, lev0_learners, meta_learner, fold):
        self.learners = lev0_learners
        self.meta_learner = meta_learner
        self.fold = fold

    def fit(self, Xtrain, Ytrain):
        """Nafitaj meta-learner"""
        kf = KFold(len(Xtrain), self.fold)
        meta_data_Xtrain = np.zeros((len(Xtrain), len(self.learners)*21))
        
        for train_indeks, test_indeks in kf:
            Xucna, Xtestna = Xtrain[train_indeks], Xtrain[test_indeks]
            Yucna, Ytestna = Ytrain[train_indeks], Ytrain[test_indeks]
            for i in range(len(self.learners)):
                l0_learner = self.learners[i]
                l0_learner.fit(Xucna, Yucna)
                meta_data_Xtrain[test_indeks, i*21:(i+1)*21] = l0_learner.predict(Xtestna)
        for i in range(len(self.learners)):
            l0_learner = self.learners[i]                                  
            l0_learner.fit(Xtrain, Ytrain)
        self.meta_learner.fit(meta_data_Xtrain, Ytrain)

    def predict(self, Xtest):
        """Oceni testno množico"""
        meta_data_Xtest = np.zeros((len(Xtest), len(self.learners)*21))
        for i in range(len(self.learners)):
            l0_learner = self.learners[i]
            meta_data_Xtest[:,i*21:(i+1)*21] = l0_learner.predict(Xtest)
        return self.meta_learner.predict(meta_data_Xtest)

#***********************************************************************#
# PRIMERJAVA STACKINGA IN POVPREČENJA S POSAMEZNIMI MODELI Z UPORABO CV #
#***********************************************************************#
###10 fold CV
##learners = [rf(50), rf_pca(50, 50), ridge(100), ridge_pca(25, 100), lasso(0.0006),
##            lasso_pca(0.001, 100), el_net(0.001, 0.9), el_net_pca(0.01, 0.75, 25)]
##
##for i in learners:
##    print(i ,cv(X, Y, 10, i))
##
##print("stacking", cv(X, Y, 10, stacking(learners, rf(50), 3)))
##
##print("averaging", cv(X, Y, 10, povprecenje(learners)))
##
#*******************************#
# NAPOVED ZA ODDAJO NA STREŽNIK #
#*******************************#

#Potrebujemo "molecular values" za CID-je v prvem(indeks 0) stolpcu tabele predict. Za njih napovemo vrednosti.
indeksi = np.zeros(Xtest.shape[0], dtype=bool)
for i in range(len(Xtest)):
    indeksi[i] = Xtest[i, 0] in stringTOfloat(predict[:,0])
    
Xtest = Xtest[indeksi]
CIDs = Xtest[:,0]
Xtest = Xtest[:,1:]     #prvi stolpec izpustimo ker so to CID-ji

Xtest = selector.transform(Xtest)

learners = [rf(100), rf_pca(100, 50), ridge(100), ridge_pca(25, 100), lasso(0.0006),
            lasso_pca(0.001, 100), el_net(0.001, 0.9), el_net_pca(0.01, 0.75, 25)]

##learners = [rf(100), rf_pca(100, 50), ridge(15), ridge_pca(15, 25)]

stacker = stacking(learners, rf(100), 10)
stacker.fit(X, Y)
ocena = stacker.predict(Xtest)

#*******************************#
# UREDIMO ZA ODDAJO NA STREŽNIK #
#*******************************#
#najprej napišem funkcijo, ki popravi vse vrednosti v matriki tako, da so na intervalu [0,100]
#potem pa napovedi stlačim v matriko 1680x3 in uporabim np.savetxt

napoved = np.zeros((1680,3), dtype=object)

descriptor = Ys[0,6:]
odID = CIDs.astype(int)
IDs = np.repeat(odID[0], 21)
matrika = np.hstack((np.reshape(IDs, (len(IDs),1)), np.reshape(descriptor, (len(descriptor),1))))

for i in range(1,80):
    IDs = np.repeat(odID[i], 21)
    enDelcek = np.hstack((np.reshape(IDs, (len(IDs),1)), np.reshape(descriptor, (len(descriptor),1))))
    matrika = np.vstack((matrika,enDelcek))
    
napoved[:,:2] = matrika

def popravi_matriko(m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i,j] < 0:
                m[i,j] = 0
            elif m[i,j] > 100:
                m[i,j] = 100
    return m

ocena = popravi_matriko(ocena)

for i in range(ocena.shape[0]):
    for j in range(ocena.shape[1]):
        ocena[i,j] = round(float(ocena[i,j]), 6)

for k in range(80):
    napoved[21*k: 21*k + 21, 2] = ocena[k, :]

np.savetxt("napoved_3_10.txt", napoved, delimiter="\t", fmt="%s")


