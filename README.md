# Total-Perspective-Vortex

# Chapitre II : Introduction

Ce sujet vise à créer une interface cerveau-ordinateur basée sur des données électroencéphalographiques (EEG) à l'aide d'algorithmes d'apprentissage automatique. En utilisant la lecture EEG d'un sujet, vous devrez inférer ce qu'il ou elle est en train de penser ou de faire - (mouvement) A ou B dans un intervalle de temps de t0 à tn.

---

# Chapitre III : Objectifs

- Traiter les données EEG (parsing et filtrage)
- Implémenter un algorithme de réduction de dimensionnalité
- Utiliser l'objet pipeline de scikit-learn
- Classifier un flux de données en "temps réel"

---

# Chapitre IV : Instructions générales

Vous devrez traiter des données provenant de l'activité cérébrale, avec des algorithmes d'apprentissage automatique. Les données ont été mesurées lors d'une expérience d'imagerie motrice, où les participants devaient effectuer ou imaginer un mouvement de la main ou du pied. Les participants ont été invités à penser ou à effectuer un mouvement correspondant à un symbole affiché à l'écran. Les résultats sont des signaux cérébraux avec des étiquettes indiquant les moments où le sujet devait effectuer une certaine tâche.

Vous devrez coder en Python, car il offre MNE, une bibliothèque spécialisée dans le traitement des données EEG, et scikit-learn, une bibliothèque spécialisée dans l'apprentissage automatique. Le sujet se concentre sur l'implémentation de l'algorithme de réduction de dimensionnalité, afin de transformer davantage les données filtrées avant la classification. Cet algorithme devra être intégré à sklearn afin que vous puissiez utiliser les outils de classification et de validation des scores de sklearn.

---

# Chapitre V : Partie obligatoire

## V.1 Structure

Vous devrez écrire un programme Python implémentant les trois phases du traitement des données :

### V.1.1 Prétraitement, parsing et formatage

Tout d'abord, vous devrez analyser et explorer les données EEG avec MNE, à partir de physionet. Vous devrez écrire un script pour visualiser les données brutes puis les filtrer afin de ne conserver que les bandes de fréquences utiles, et visualiser à nouveau après ce prétraitement. 

C'est dans cette partie que vous déciderez quelles caractéristiques vous extrayez des signaux pour les fournir à votre algorithme. Vous devrez donc être minutieux dans le choix des éléments importants pour le résultat souhaité. Un exemple est d'utiliser la puissance du signal par fréquence et par canal comme entrée du pipeline.

La plupart des algorithmes liés au filtrage et à l'obtention du spectre du signal utilisent la transformée de Fourier ou la transformée en ondelettes (cf. bonus).

### V.1.2 Pipeline de traitement

Ensuite, le pipeline de traitement doit être mis en place :
- Algorithme de réduction de dimensionnalité (par exemple : PCA, ICA, CSP, CSSP...).
- Algorithme de classification, il y a beaucoup de choix parmi ceux disponibles dans sklearn, pour déterminer quel morceau de données correspond à quel type de mouvement.
- Lecture "playback" sur le fichier pour simuler un flux de données.

Il est conseillé de tester d'abord votre architecture de programme avec les algorithmes sklearn et MNE, avant d'implémenter votre propre algorithme CSP ou tout autre algorithme de votre choix.

Le programme devra contenir un script pour l'entraînement et un autre pour la prédiction. Le script de prédiction devra effectuer sa tâche sur un flux de données, avec un délai de 2 secondes après l'envoi du morceau de données dans le pipeline de traitement. (vous ne devez pas utiliser mne-realtime)

Vous devez utiliser l'objet pipeline de sklearn (utilisez les classes baseEstimator et transformerMixin de sklearn).

---

## V.1.3 Implémentation

L'objectif est d'implémenter l'algorithme de réduction de dimensionnalité. Cela signifie exprimer les données avec les caractéristiques les plus significatives, en déterminant une matrice de projection. 

Cette matrice projettera les données sur un nouvel ensemble d'axes qui exprimeront les variations les plus "importantes". Cela s'appelle un changement de base, et il s'agit d'une transformation composée de rotations, translations et opérations de mise à l'échelle.

Ainsi, le PCA considère votre ensemble de données et détermine de nouveaux composants de base, classés en fonction de l'importance des variations qu'ils expliquent dans les données.

Le CSP (Common Spatial Patterns) analyse les données en fonction des classes de sortie et tente de maximiser les variations entre elles.

Le PCA est un algorithme plus général, mais le CSP est plus utilisé dans les interfaces cerveau-ordinateur EEG. Prenons l'expression formelle d'un signal EEG :

$$
\{E_n\}_{n=1}^{N} \in \mathbb{R}^{ch \times time} \quad (V.1)
$$

Nous avons :
- N : le nombre d'événements de chaque classe,
- ch : le nombre de canaux (électrodes),
- time : la longueur de l'enregistrement d'un événement.

En considérant la matrice du signal extrait $X \in \mathbb{R}^{d \times N}$, sachant que $d = ch \times time$ est la dimension d'un vecteur de signal pour un enregistrement d'événement.

Votre objectif sera de trouver une matrice de transformation $W$ telle que :

$$
W^T X = X_{CSP}
$$

où $X_{CSP}$ correspond aux données transformées par l'algorithme CSP (ou $X_{PCA}$, $X_{ICA}$, ... selon le choix que vous ferez).

Les fonctions Numpy ou scipy sont également autorisées pour trouver les valeurs propres, les valeurs singulières et l'estimation de la matrice de covariance.

---

## V.1.4 Entraînement, validation et test

- Vous devez utiliser `cross_val_score` sur l'ensemble du pipeline de traitement pour évaluer votre classification.
- Vous devez choisir comment diviser votre ensemble de données entre les ensembles d'entraînement, de validation et de test (ne pas trop ajuster, avec des séparations différentes à chaque fois).
- Vous devez obtenir une précision moyenne de 60% sur tous les sujets utilisés dans vos données de test (correspondant aux six types d'expériences et sur des données jamais apprises).
- Vous pouvez entraîner/prédire sur le sujet et la tâche de votre choix.

Exemples :

