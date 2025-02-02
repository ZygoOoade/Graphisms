# Suppression des filigranes sur des vidéos et augmentation de leur résolution

## Suppression des filigranes sur des vidéos
[Video Watermark Removal.ipynb](https://github.com/ZygoOoade/Graphisms/blob/master/Video_Watermark_Removal.ipynb) permet d'enlever le watermark d'une vidéo sur une zone ciblée rectangulaire $(x_1, x_2, x_3, x_4)$ . $x_1, x_2$ sont les coordonnées haut-gauche et $x_3, x_4$ les coordonnées bas-droite  .
Vous pouvez utiliser des sites [comme celui-ci](https://pixspy.com/) pour identifier des coordonnées précises sur une frame de votre vidéo.<br>
Le modèle utilisé pour enlever le watermark est un Spatio-Temporal Trajectory Network (STTN).<br>
Ce notebook est une version fonctionnelle de [ce repo github](https://github.com/chenwr727/KLing-Video-WatermarkRemover-Enhancer) .<br>
Il y a deux paramètres supplémentaires du modèle expliqués sur sa page (`mask_expand: 30` et `neighbor_stride: 10`). <br>
`mask_expand: 30` sert à agrandir la zone du filigrane d'un nombre donné de pixels.<br>
`neighbor_stride: 10` sert à contrôler la taille de la foulée lors du calcul des images voisines à l'aide du réseau de trajectoires spatio-temporelles. Une petite valeur augmente *a priori* le temps de calcul mais permet une meilleure qualité.

**Limitations**<br>
En raison de sa méthode, le modèle ne marche pas si le watermark bouge considérablement durant la vidéo.<br>
Par ailleurs, à moins que le filigrane (watermark) soit rectangulaire, le fait d'utiliser une zone $(x_1, x_2, x_3, x_4)$ ne **cible** pas avec finesse le watermark parce que cela conduit à prendre en compte des pixels qu'il vaudrait mieux ne pas écraser.
A cet égard, d'autres méthodes utilisent un fichier 'mask' qui permet de cibler exactement le filigrane, qui est souvent un logo.

## Amélioration de la qualité de la vidéo
[Video Watermark Removal.ipynb](https://github.com/ZygoOoade/Graphisms/blob/master/Video_Watermark_Removal.ipynb) permet également d'augmenter la résolution / la qualité de la vidéo en utilisant **RealESRGAN_model** et **GFPGANer_model_path**.


# Génération d'image

[Flux_opti.ipynb](https://github.com/ZygoOoade/Graphisms/blob/master/flux_opti.ipynb) est une version optimisée du notebook de [Camenduru](https://github.com/camenduru/flux-jupyter).
Grâce à l'optimisation mise en oeuve, il est possible de générer des images Full-HD via Flux Dev (https://blackforestlabs.ai) sur Google Colab.
