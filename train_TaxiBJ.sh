# total 7 baselines, the results of ST-3DNet is already obtained. 
# STAR
cd STAR
python main_taxiBJ.py
# python transfer_learning_evaluation2.py
cd ..
# St-ResNet
cd ST-ResNet
python main_taxiBJ.py
# python transfer_learning_evaluation2.py
cd ..
# # MST3D
cd MST3D
# python transfer_learning_evaluation2.py
python main_taxiBJ.py
cd ..
# # Pred-CNN
cd Pred-CNN
# python transfer_learning_evaluation2.py
python main_taxiBJ.py
cd ..
# # # ST3DNet
# cd ST3DNet
# python prepareData.py
# # python transfer_learning_evaluation2.py
# python main_taxiBJ.py

# # 3D-CLoST
cd 3D-CLoST
python main_taxiBJ.py
cd ..

cd MST3D
python main_taxiBJ.py
cd ..

cd Autoencoder
python main_taxiBJ.py
cd ..