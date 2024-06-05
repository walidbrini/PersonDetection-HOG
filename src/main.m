

imagefilesPos = 'Database\pos';  
imagefilesNeg = 'Database\neg';   

[train_matrix,labels] = train (imagefilesPos,imagefilesNeg);


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