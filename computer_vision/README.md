# calcul_perf_PREPROC.py

Il s'agit d'un copié-collé du code de PREPROC.py afin de calculer le temps de calcul de ce dernier et d'évaluer les fonctions les plus gourmandes en ressources.
Ici, le module line-profiler est utilisé. Bien que cela n'ait pas d'impact critique sur le temps de calcul du raspberry Pi, il a été évalué que cv2.Canny() est responsable de l'utilisation de 95% du temps de PREPROC.py

# dataset_analyse.ipynb

Code permettant d'évaluer la distribution des données d'un dataset.

# test_ML.ipynb

L'objectif de ce code fut de créer un modèle permettant de donner comme primitive le centre_piste pour chaque image en entrée.
Les images brutes subissent un préprocessing simple sous forme de cv2.inRange().

Le modèle est un CNN et est entrainé par régression.

