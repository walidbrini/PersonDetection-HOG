function hogs = hogfeatures(I, h, B, nbins)

    GradE = conv2(I, h, 'same');
    GradN = conv2(I, h.', 'same');
    
    % Calcul gradient 
    Grad_sqrt = sqrt(GradE.^2 + GradN.^2); 
    Or = atan2(GradN, GradE);
    Or(Or < 0) = Or(Or < 0) + pi;
    
    [H, W] = size(I);
    
    % Calcul N et M bloc val entiere
    blocH = floor(H / B);
    blocW = floor(W / B);
    
    % Init
    hogs = [];
    
    % Definir les bornes des bins
    bin_edges = linspace(0, pi, nbins+1);
    
    for j = 1: blocW
        for i = 1: blocH
            % Extraire fenetre taille BxB
            window_norme = Grad_sqrt((i-1)*B+1 : i*B, (j-1)*B+1 : j*B);
            window_orientation = Or((i-1)*B+1:i*B, (j-1)*B+1:j*B);
            
            % Init histogram pour la fenetere courante 
            hist = zeros(1, nbins);
            
            % Calcul histogram pour la fenetre courante 
            for bin = 1:nbins
                % Trouvez les pixel qui tombe dans la bin courante 
                bin_mask = (window_orientation >= bin_edges(bin)) & (window_orientation < bin_edges(bin+1));
                % Somme des magnitudes
                hist(bin) = sum(window_norme(bin_mask));
            end
            
            % Normaliser l'histogram
            hist_norm = norm(hist);
            if hist_norm > 0
                hist = hist / hist_norm;
            end
            
            % Ajouter l'histogram au HOG feature vector
            hogs = [hogs, hist]; 
            
        end
    end
end
