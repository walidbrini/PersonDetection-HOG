

%% Gradient d'une image
close all ; 
clear all ; 
clc ; 

% Lire une image en entre 
image = imread('Database\pos\B10_crop001119a.png');
ing = im2gray(image);

% Calcul du gradient
[Or, G] = gradient(ing); 


%% Extraction Histogramme des gradients

% paramtres d'extraction et de visualisation
[H,W]=size(ing); hCell=8; wCell=8;
nbhCell=H/hCell; nbwCell=W/wCell;
nbBins=9;

% paramtres pour la visualisation
param.ImageSize=[H W];
param.WindowSize=[H W];
param.CellSize=[hCell wCell];
param.BlockSize=[1 1];
param.BlockOverlap=[0 0];
param.NumBins=9;
param.UseSignedOrientation=0;

% Extraction
hogfeat=hogfeatures(double(ing),[1 0 -1],hCell,nbBins);

% Visualisation
visu=vision.internal.hog.Visualization(hogfeat, param);
figure; imshow(uint8(image)); 
hold on;
plot(visu)
title('HOGs manual')
pause(0.1)


%% Extraction de tous les features de la base de donnees 


% Il faut se placer dans la bonne directory pwd 'O:\TraitementImage\TpDetectionPieton'
imagefilesPos = 'Database\pos';  
imagefilesNeg = 'Database\neg';      


% Verifier la disponibilite du dossier contenant les images
if ~isdir(imagefilesPos)
  errorMessage = sprintf('Error: Ce dossier existe pas \n%s', imagefilesPos);
  uiwait(warndlg(errorMessage));
  return;
end

filePatternPos = fullfile(imagefilesPos, '*.png');
filePatternNeg = fullfile(imagefilesNeg, '*.png');

% Extraire le nombre d'images positives et d'images negatives
pngFilesP = dir(filePatternPos)
pngFilesN = dir(filePatternNeg)

Np = length(pngFilesP);
Ng = length(pngFilesN);

M = nbBins * H/hCell*W/wCell;

% Initialisation de la matrice des features positives
train_matrix_pos = zeros(Np,M);

% Lecture des images positives et creation de la train_matrix_pos resultat
% de la concetenation de tous les HOG features
for k = 1:Np
  baseFileName = pngFilesP(k).name;
  fullFileName = fullfile(imagefilesPos, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  imageArray = imread(fullFileName);
  ing = rgb2gray(imageArray);
  %imshow(imageArray);  % Display image.
  %drawnow; % Force display to update immediately.
  hogfeat  = hogfeatures(double(ing),[1 0 -1],hCell,nbBins);
  train_matrix_pos(k,:)= hogfeat' ; 
end


% Initialisation de la matrice des features negatives
train_matrix_neg = zeros(Ng,M);

% Lecture des images Negative et creation de la train_matrix_neg resultat
% de la concetenation de tous les HOG features

for k = 1:Ng
  baseFileName = pngFilesN(k).name;
  fullFileName = fullfile(imagefilesNeg, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  imageArray = imread(fullFileName);
  ing = rgb2gray(imageArray);
  %imshow(imageArray);  % Display image.
  %drawnow; % Force display to update immediately.
  hogfeat  = hogfeatures(double(ing),[1 0 -1],hCell,nbBins);
  train_matrix_neg(k,:)= hogfeat' ; 
end

% Creation de la train_matrix, la concetenation des deux train_matrix pos
% et neg, labels pour chaque donnee 1 si positive 0 si negative
train_matrix = [train_matrix_pos ; train_matrix_neg]; 
labels =[ones(Np,1);zeros(Ng, 1 )];

%% Analyse en Composantes Principales et Visualisation des features

% Initialisation des variables 
X = train_matrix;
[p, n] = size(X);

% Centrage des donnees mean par rapport a chaque feature / alignement a
% lorigne ce qui explique les valeurs negative
Xm = X - mean(X);

% Normaliser les donnees centrees  (1600x1152)*(1152 x 1152)
Xs = Xm * diag(1 ./ std(Xm)); 

% Calcule de la matrice de covariance des donnees valables si les variables xik sont centrees
% covV de taille(1152*1152)
% Calculer la matrice de covariance permet de quantifier les relations entre les differentes features des donnees
covV = Xs' * Xs / p;

% Calcule des vecteurs propres et valeurs propores
% Les valeurs propres indiquent l'importance de chaque nouvelle dimension dans la representation des donnees
% Calculer les valeurs propres et vecteurs propres permet d'identifier les directions dans lesquelles les donnees varient le plus 
[U, D, V] = eig(covV);


% Tri descendant des valeurs propres et des vecteur propres
[Ds, Isort] = sort(diag(D), 'descend');
% Les vecteurs propres de la matrice de covariance representent les nouvelles bases dans lesquelles les donnees peuvent etre projetees
V = V(:, Isort);


% La projection des donnees dans l'espace des composantes principales
Xp = Xs * V;

% Visualisation de l'evolution des lamdas (les valeurs propres) 
figure, plot(Ds);


%% Trace en 3D des premieres composantes principales

% Plot 3D 3 first principal components
figure, hold on;
for i = 0:1
    Il = find(labels == i);
    scatter3(Xp(Il, 1), Xp(Il, 2), Xp(Il, 3), 'filled');
end
title('Principal Components 1-2-3');

figure, hold on;
nlig = 2;
ncol = 3;
for j = 1:nlig * ncol
    subplot(nlig, ncol, j); hold on;
    for i = 0:1
        Il = find(labels == i);
        scatter(Xp(Il, 2 * j - 1), Xp(Il, 2 * j), 'filled');
    end
    title(['Principal Component ', num2str(2 * j - 1), '-', num2str(2 * j)]);
    pause;
end

%% Test de classification 


% On choisit nf pour reduire la dimensionnalite
nf = 309;

% Le choix de nf est base sur les valeurs propres triees par ordre decroissant
% On conserve les 309 premieres directions en utilisant la methode du coude

% La fonction classify cherche la courbe quadratique qui separe les deux ensembles
label_test = classify(Xp(:, 1:nf), Xp(:, 1:nf), labels);

% Il faut prendre en compte qu'on classifie sur les donees d'entrainement
% (Le modele a ete deja entraine sur ce type d'image)

% Classification 
tst = labels - label_test;
FP = sum(tst == -1); % Faux positifs
FN = sum(tst == 1);  % Faux negatifs
TP = sum((labels == 1) & (label_test == 1)); % Vrais positifs
TN = sum((labels == 0) & (label_test == 0)); % Vrais negatifs


%% Evaluation de la classification 

% Calcul de l'accuracy, la precision, le rappel et le F1-score
accuracy = (TP + TN) / length(labels);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision * recall);

% Afficher les resultats
fprintf('Accuracy: %.2f\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1_score);


% Matrice de confusion
confusionmat(labels, label_test)

% Affichage de la matrice de confusion
figure;
confusionchart(labels, label_test);
title('Matrice de confusion');
