# Anleitung

Um trainingsergebnisse anzuschauen plot_results_1.ipynb, plot_results_2.ipynb ausführen.

Um den Versuch nachzumachen, im constants.py File, folgende variabeln setzen:
main_folder = '../'
dataset1_folder = main_folder + 'dataset_tfrecord' + '/'
dataset2_folder = main_folder + 'dataset2' + '/'

Diese Variabeln müssen auf existente Folders zeigen. Hier werden die Datensätze gespeichert.

Zusätzlich müssen folgende Folders erstellt werden:
SECT1_CHECKPOINT_FOLDER = main_folder + 'checkpoints_sect1' + '/'
SECT2_CHECKPOINT_FOLDER = main_folder + 'checkpoints_sect2' + '/'
SINGLE_CHECKPOINT_FOLDER = main_folder + 'checkpoints_single' + '/'
OPTIMIZER_FOLDER = main_folder + 'optimizers' + '/'

In diesen Foldern werden Trainingsdaten gespeichert.

Als Beispiel habe ich meine Folders als Bild eingefügt.
![alt text](Folders.png)