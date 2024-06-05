% Fonction qui calcul le gradient d'une image 
% Retourne Or, Grad 

function [Or, Grad] = gradient(I)
    hy = [1 0 -1];
    hx = [1 0 -1].';
    
    val = 0.15;
    
    % Ajouter un padding al'image pour gerer les bords
    padSize = 1; 
    I_padded = padarray(I, [padSize, padSize], 'replicate');
    
    % Calculer les gradients 
    GradE = conv2(I_padded, hy, 'same'); 
    GradN = conv2(I_padded, hx, 'same'); 

    % Enlever le padding 
    GradE = GradE(padSize+1:end-padSize, padSize+1:end-padSize);
    GradN = GradN(padSize+1:end-padSize, padSize+1:end-padSize);

    % Magnitude du gradient
    Grad_sqrt = sqrt(GradE.^2 + GradN.^2); 
    
    % Orientation du gradient
    Or = atan2(GradN, GradE);

    % Normaliser La Magnitude
    Grad = Grad_sqrt / max(Grad_sqrt(:));
  
    % Normaliser l'orientation 
    Or = Or / (2 * pi) + 0.5;
    Or(Grad < val) = 0;

    % Afficher
    montage({Grad, Or});
    


end

