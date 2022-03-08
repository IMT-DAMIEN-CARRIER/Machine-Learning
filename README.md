# Guide d'utilisation



[TOC]

## Projet GitLab

Si vous clonez le projet tel quel, il vous suffit de vous rendre dans le répertoire **src/** et dans lancer le script **ml.py** pour visualiser ce que fais notre code :

```bash
cd ./src
python ml.py
```



## Sources uniquement

Si vous avez récupéré nos sources uniquement (fichiers .py et rapport), il vous faut télécharger le dataset suivant :  https://www.kaggle.com/mczielinski/bitcoin-historical-data

Une fois cela fait, placez le fichier .csv obtenu dans le répertoire où vous avez stocker nos sources. Effectuez ensuite les manipulations suivantes :

```bash
mkdir src/
cp *.py src/
cd src/
python setup.py #Creer un nouveau fichier sans les valeurs nulles
python ml.py
```

